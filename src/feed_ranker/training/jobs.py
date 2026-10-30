"""Optional local composition of the independent data/model/training interfaces.

This convenience job owns paths and export. Trainer itself only consumes instances.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from feed_ranker.artifacts import save_cosine, save_mf, save_popularity, save_ranker
from feed_ranker.data.dataset import EventDataset, make_loader
from feed_ranker.data.parquet import ParquetReader
from feed_ranker.evaluation.predictions import PredictionEvaluator
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.persistence import collect_run_provenance, write_json
from feed_ranker.training.bpr import ObservedSampler, fit_mf, histories
from feed_ranker.training.checkpoint import load_model_weights, save_checkpoint
from feed_ranker.training.objectives import Objective, TaskLossConfig
from feed_ranker.training.optimizer import OptimizerConfig, build_optimizer
from feed_ranker.training.trainer import TrainConfig, Trainer


@dataclass(frozen=True)
class TrainingJobConfig:
    data: Path
    model: DeepConfig
    tasks: tuple[TaskLossConfig, ...]
    training: TrainConfig
    batch_size: int = 1024
    eval_batch_size: int = 8192
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if len(self.tasks) != 1 or self.tasks[0].output != self.model.output:
            raise ValueError("Deep requires one objective matching its output")
        if any(type(n) is not int or n < 1 for n in (self.batch_size, self.eval_batch_size)):
            raise ValueError("Loader batch sizes must be positive")


def train_job(
    config: TrainingJobConfig, directory: Path, *, resume_from: Path | None = None
) -> dict[str, Any]:
    data = ParquetReader(config.data)
    task = config.tasks[0]
    if task.target not in data.targets:
        raise ValueError("Objective target does not match prepared inputs")
    if task.mask is not None or task.weight is not None:
        raise ValueError("Current observed-event preparation has no mask/weight columns")
    if (
        set(config.model.feature_names) - data.feature_dtypes.keys()
        or any(data.vocab_sizes.get(n) != size for n, size in config.model.sparse.items())
        or any(n in data.vocab_sizes for n in config.model.dense)
    ):
        raise ValueError("Model inputs and materialized feature schema disagree")
    targets = (task.target,)
    columns = [*data.metadata_dtypes, *config.model.feature_names, *targets]
    train = EventDataset(data.read_split("train", columns), config.model.feature_names, targets)
    validation = EventDataset(
        data.read_split("validation", columns), config.model.feature_names, targets
    )
    train_loader = make_loader(train, config.batch_size, config.training.seed)
    validation_loader = make_loader(validation, config.eval_batch_size)

    torch.manual_seed(config.training.seed)
    model = DeepModel(config.model).to(config.training.device)
    optimizer = build_optimizer(model, config.optimizer)
    objective = Objective(config.tasks)
    evaluator = PredictionEvaluator({task.output: task.target})
    trainer = Trainer(model, optimizer, objective, config.training)
    run_identity = {
        "data_manifest_sha256": data.identity,
        "model": config.model.to_dict(),
        "objective": [asdict(task)],
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
    }
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "config.json",
        {
            "data": str(config.data),
            **run_identity,
            "training": {**asdict(config.training), "optimizer": asdict(config.optimizer)},
        },
    )
    provenance = {
        **collect_run_provenance(),
        "contract": run_identity,
        "seed": config.training.seed,
    }
    write_json(directory / "provenance.json", provenance)
    result = trainer.fit(
        train_loader,
        validation_loader,
        evaluator,
        checkpoint_dir=directory / "checkpoints",
        run_identity=run_identity,
        resume_from=None if resume_from is None else resume_from / "checkpoints",
    )

    load_model_weights(model, Path(result["best_checkpoint"]))
    provenance.update(
        selected_epoch=result["best_epoch"], checkpoint_policy=result["checkpoint_policy"]
    )
    save_ranker(directory / "model", model, data.load_feature_state(), provenance)
    result["artifact"] = str(directory / "model")
    write_json(directory / "result.json", result)
    return result


def job_config(value: dict[str, Any]) -> TrainingJobConfig:
    """Resolve an experiment's input list once against fitted ETL vocabularies."""
    data = Path(value["data"])
    reader = ParquetReader(data)
    model = dict(value["model"])
    names = tuple(model.pop("features"))
    if not names or len(set(names)) != len(names) or set(names) - reader.feature_dtypes.keys():
        raise ValueError("Model inputs must be fitted features, never targets/metadata")
    model["dense"] = tuple(n for n in names if n not in reader.vocab_sizes)
    model["sparse"] = {n: reader.vocab_sizes[n] for n in names if n in reader.vocab_sizes}
    training = dict(value["training"])
    optimizer = OptimizerConfig(**training.pop("optimizer", {}))
    return TrainingJobConfig(
        data,
        DeepConfig(**model),
        tuple(TaskLossConfig(**t) for t in value["tasks"]),
        TrainConfig(**training),
        optimizer=optimizer,
        **value.get("loading", {}),
    )


def fit_recall_job(
    data_path: Path, directory: Path, *, epochs: int, seed: int, target: str = "long_view"
) -> dict[str, Any]:
    if type(epochs) is not int or epochs < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Require positive MF epochs and a valid seed")
    data = ParquetReader(data_path)
    if target not in data.manifest["targets"]:
        raise ValueError("Recall target must be a prepared target")
    state = data.load_feature_state()
    users = state.user_ids
    items = state.item_ids[state.item_features["item_code"][1:] > 0]
    train = data.read_split("train", ["user_id", "video_id", target])
    positive, negative = histories(train, users, items, target)
    cosine = SparseCosineCooccurrence(positive, len(items))
    directory.mkdir(parents=True, exist_ok=False)
    sampler = ObservedSampler(positive, negative)
    torch.manual_seed(seed)
    mf = BPRMF(len(users), len(items), 32)
    optimizer = build_optimizer(mf, OptimizerConfig(lr=1e-3))
    history = fit_mf(mf, sampler, optimizer, epochs=epochs, seed=seed)
    mf.eval()
    (directory / "checkpoints").mkdir()
    save_checkpoint(
        directory / "checkpoints" / "mf.pt",
        {
            "model": mf.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs - 1,
            "policy": "fixed final epoch; no validation selection",
            "seed": seed,
            "dimension": mf.user.embedding_dim,
        },
    )
    write_json(directory / "history.json", history)
    provenance = {
        **collect_run_provenance(),
        "data_manifest_sha256": data.identity,
        "seed": seed,
        "target": target,
        "mf_epochs": epochs,
        "checkpoint_policy": "fixed final epoch",
    }
    write_json(
        directory / "config.json",
        {
            "data": str(data_path),
            "target": target,
            "epochs": epochs,
            "seed": seed,
            "dimension": 32,
            "optimizer": asdict(OptimizerConfig(lr=1e-3)),
        },
    )
    write_json(directory / "provenance.json", provenance)
    models = directory / "models"
    save_mf(models / "mf", mf, users, items, sampler.supported, provenance)
    save_popularity(models / "popularity", cosine.counts, users, items, provenance)
    save_cosine(models / "cosine", cosine, users, items, provenance)
    result = {
        "artifacts": {kind: str(models / kind) for kind in ("mf", "popularity", "cosine")},
        "history": history,
        "supported_users": int(sampler.supported.sum()),
        "data_manifest_sha256": data.identity,
    }
    write_json(directory / "result.json", result)
    return result
