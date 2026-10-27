"""One configured preparation-consuming train/validate/select/export lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from feed_ranker.artifacts import collect_run_provenance, write_json
from feed_ranker.data.prepared import EventDataset, make_loader, open_prepared
from feed_ranker.evaluation.predictions import evaluate_predictions
from feed_ranker.models.artifacts import ModelArtifact, save_artifact
from feed_ranker.models.deep import ModelConfig, build_model
from feed_ranker.training.objective import Objective, TaskLossConfig
from feed_ranker.training.trainer import TrainConfig, Trainer


@dataclass(frozen=True)
class TrainingJobConfig:
    data: Path
    model: ModelConfig
    tasks: tuple[TaskLossConfig, ...]
    training: TrainConfig


class PredictionEvaluator:
    def __init__(
        self, dataset: EventDataset, tasks: tuple[TaskLossConfig, ...], batch_size: int
    ) -> None:
        self.dataset, self.tasks, self.batch_size = dataset, tasks, batch_size

    def evaluate(self, model: nn.Module, split: str) -> dict[str, Any]:
        if split != "val":
            raise ValueError("Only validation is available for checkpoint selection")
        return evaluate_predictions(
            model,
            make_loader(self.dataset, self.batch_size),
            {t.output: t.target for t in self.tasks},
            {t.output: t.mask for t in self.tasks if t.mask is not None},
        )


def train_job(
    config: TrainingJobConfig, directory: Path, *, resume_from: Path | None = None
) -> dict[str, Any]:
    data = open_prepared(config.data)
    for task in config.tasks:
        if (
            task.target not in data.schema
            or data.schema[task.target].role != "target"
            or task.output != config.model.output
        ):
            raise ValueError("Objective target/output does not match prepared/model schema")
        if task.mask is not None or task.weight is not None:
            raise ValueError("Current observed-event preparation has no mask/weight columns")
    state = data.load_feature_state()
    targets = tuple(dict.fromkeys(t.target for t in config.tasks))
    columns = ["event_id", "user_id", "video_id", "time_ms", *targets]
    train, validation = [
        EventDataset(data.read_split(split, columns), state, config.model.features, targets)
        for split in ("train", "validation")
    ]
    torch.manual_seed(config.training.seed)
    model = build_model(config.model, data.schema)
    objective = Objective(config.tasks)
    trainer = Trainer(
        model,
        lambda epoch: make_loader(train, config.training.batch_size, config.training.seed + epoch),
        config.training,
        objective=objective,
        evaluator=PredictionEvaluator(validation, config.tasks, config.training.eval_batch_size),
    )
    contract = {
        "data_manifest_sha256": data.identity,
        "model": asdict(config.model),
        "objective": [asdict(t) for t in config.tasks],
        "schema": {n: asdict(s) for n, s in data.schema.items()},
    }
    result = trainer.fit(directory, {"resume_contract": contract}, resume_from=resume_from)
    provenance = {
        **collect_run_provenance(),
        "contract": contract,
        "seed": config.training.seed,
        "selected_epoch": result["best_epoch"],
        "checkpoint_policy": result["checkpoint_policy"],
    }
    save_artifact(
        directory / "model", ModelArtifact("deep", model, "", provenance, config.model, state)
    )
    result["artifact"] = str(directory / "model")
    write_json(directory / "result.json", result)
    write_json(directory / "provenance.json", provenance)
    return result
