"""Optional local composition of the independent data/model/training interfaces.

This convenience job owns paths and export. Trainer itself only consumes instances.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from feed_ranker.artifacts import (
    CosineArtifact,
    MFArtifact,
    PopularityArtifact,
    RankerArtifact,
    save_artifact,
)
from feed_ranker.data.dataset import EventDataset, make_loader
from feed_ranker.data.parquet import ParquetReader
from feed_ranker.evaluation.predictions import PredictionEvaluator
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.persistence import collect_run_provenance, write_json
from feed_ranker.training.bpr import ObservedSampler, fit_mf, histories
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

    def __post_init__(self) -> None:
        if len(self.tasks) != 1 or self.tasks[0].output != self.model.output:
            raise ValueError("Deep requires one objective matching its output")
        if any(type(n) is not int or n < 1 for n in (self.batch_size, self.eval_batch_size)):
            raise ValueError("Loader batch sizes must be positive")


def train_job(
    config: TrainingJobConfig, directory: Path, *, resume_from: Path | None = None
) -> dict[str, Any]:
    data = ParquetReader(config.data)
    for task in config.tasks:
        if task.target not in data.targets or task.output != config.model.output:
            raise ValueError("Objective target/output does not match prepared/model inputs")
        if task.mask is not None or task.weight is not None:
            raise ValueError("Current observed-event preparation has no mask/weight columns")
    tables = data.load_feature_state()
    targets = tuple(dict.fromkeys(t.target for t in config.tasks))
    columns = [*data.metadata_dtypes, *config.model.feature_names, *targets]
    train, validation = [
        EventDataset(data.read_split(split, columns), config.model.feature_names, targets)
        for split in ("train", "validation")
    ]
    torch.manual_seed(config.training.seed)
    model = DeepModel(config.model)
    fitted = RankerArtifact(model, tables, "", {})  # Validate ETL/model agreement before fitting.
    evaluator = PredictionEvaluator(
        {t.output: t.target for t in config.tasks},
        {t.output: t.mask for t in config.tasks if t.mask is not None},
    )
    validation_batches = make_loader(validation, config.eval_batch_size)
    trainer = Trainer(model, config.training, loss=Objective(config.tasks))
    contract = {
        "data_manifest_sha256": data.identity,
        "model": config.model.to_dict(),
        "objective": [asdict(t) for t in config.tasks],
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
    }
    result = trainer.fit(
        lambda epoch: make_loader(train, config.batch_size, config.training.seed + epoch),
        lambda current: evaluator.evaluate(current, validation_batches),
        directory,
        {"resume_contract": contract},
        resume_from=resume_from,
    )
    fitted.provenance = {
        **collect_run_provenance(),
        "contract": contract,
        "seed": config.training.seed,
        "selected_epoch": result["best_epoch"],
        "checkpoint_policy": result["checkpoint_policy"],
    }
    save_artifact(directory / "model", fitted)
    result["artifact"] = str(directory / "model")
    write_json(directory / "result.json", result)
    write_json(directory / "provenance.json", fitted.provenance)
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
    training["optimizer"] = OptimizerConfig(**training.get("optimizer", {}))
    return TrainingJobConfig(
        data,
        DeepConfig(**model),
        tuple(TaskLossConfig(**t) for t in value["tasks"]),
        TrainConfig(**training),
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
    torch.save(
        {
            "model": mf.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs - 1,
            "policy": "fixed final epoch; no validation selection",
            "seed": seed,
            "dimension": mf.user.embedding_dim,
        },
        directory / "mf.pt",
    )
    write_json(directory / "mf_history.json", history)
    provenance = {
        **collect_run_provenance(),
        "data_manifest_sha256": data.identity,
        "seed": seed,
        "target": target,
        "mf_epochs": epochs,
        "checkpoint_policy": "fixed final epoch",
    }
    for artifact in (
        MFArtifact(mf, users, items, sampler.supported, "", provenance),
        PopularityArtifact(cosine.counts, users, items, "", provenance),
        CosineArtifact(cosine, users, items, "", provenance),
    ):
        save_artifact(directory / artifact.kind, artifact)
    result = {
        "artifacts": {kind: str(directory / kind) for kind in ("mf", "popularity", "cosine")},
        "history": history,
        "supported_users": int(sampler.supported.sum()),
        "data_manifest_sha256": data.identity,
    }
    write_json(directory / "result.json", result)
    return result
