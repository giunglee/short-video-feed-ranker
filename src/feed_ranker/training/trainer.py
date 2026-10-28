"""Optimization and checkpoint selection over supplied training instances.

The trainer knows no dataset paths, feature recipes, artifact formats or funnel
policies. It consumes batches, a supplied objective, and explicit evaluation callbacks.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from feed_ranker.data.instances import Batch
from feed_ranker.persistence import save_checkpoint, write_json
from feed_ranker.training.optimizer import OptimizerConfig, build_optimizer

LossFunction = Callable[
    [dict[str, torch.Tensor], Batch], tuple[torch.Tensor, dict[str, dict[str, float | int]]]
]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15  # reasonable default, not from any specific production config
    seed: int = 42  # recorded initialization/sampling seed; orchestration applies it
    device: str = "cpu"
    steps: int | None = None
    selection_metric: str = "long_view/roc_auc"
    selection_mode: str = "max"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.steps is not None and (type(self.steps) is not int or self.steps < 1):
            raise ValueError("steps must be positive or None")
        if not self.selection_metric or self.selection_mode not in ("min", "max"):
            raise ValueError("Require a selection metric and min/max mode")


class Trainer:
    """Own model/optimizer execution state; consume instances through fit().

    loss returns a scalar and per-task numerator/denominator/counts/task_weight.
    Training loss reflects changing weights. Optional train_evaluate measures a
    fixed checkpoint. record_metrics can persist epoch records independently.
    """

    def __init__(self, model: nn.Module, config: TrainConfig, *, loss: LossFunction) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.loss = loss
        self.optimizer = build_optimizer(self.model, config.optimizer)

    def fit(
        self,
        train_batches: Callable[[int], Iterable[Batch]],
        validate: Callable[[nn.Module], dict[str, Any]],
        directory: Path,
        metadata: dict[str, Any],
        *,
        train_evaluate: Callable[[nn.Module], dict[str, Any]] | None = None,
        record_metrics: Callable[[dict[str, Any]], None] | None = None,
        resume_from: Path | None = None,
    ) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", {"training": asdict(self.config), **metadata})
        history: list[dict[str, Any]] = []
        best_score = math.inf if self.config.selection_mode == "min" else -math.inf

        if resume_from is not None:
            history, best_score = self._resume(resume_from, directory, metadata)

        for epoch in range(len(history), self.config.epochs):
            started = time.monotonic()
            training = self._train_epoch(train_batches(epoch), epoch)
            training_seconds = time.monotonic() - started
            eval_started = time.monotonic()
            train_evaluation = None if train_evaluate is None else train_evaluate(self.model)
            train_eval_seconds = time.monotonic() - eval_started
            eval_started = time.monotonic()
            validation = validate(self.model)
            validation_seconds = time.monotonic() - eval_started
            score = validation[self.config.selection_metric]
            if score is None or not math.isfinite(score):
                raise ValueError(f"Selection metric {self.config.selection_metric!r} is undefined")
            history.append(
                {
                    "epoch": epoch,
                    **training,
                    "val": validation,
                    "train_eval": train_evaluation,
                    "training_seconds": training_seconds,
                    "train_eval_seconds": train_eval_seconds,
                    "validation_seconds": validation_seconds,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "seconds": time.monotonic() - started,
                }
            )
            checkpoint_started = time.monotonic()
            checkpoint = self._checkpoint(epoch, validation, metadata)
            save_checkpoint(directory / "last.pt", checkpoint)
            # Strict improvement retains the earliest checkpoint on an exact tie.
            improved = (
                score < best_score if self.config.selection_mode == "min" else score > best_score
            )
            if improved:
                best_score = score
                save_checkpoint(directory / "best.pt", checkpoint)
            history[-1]["checkpoint_seconds"] = time.monotonic() - checkpoint_started
            history[-1]["seconds"] = time.monotonic() - started
            write_json(directory / "history.json", history)
            if record_metrics is not None:
                record_metrics(history[-1])
            print(
                f"epoch={epoch} train={training['train_loss']:.6f} "
                f"val_{self.config.selection_metric}={score:.6f}",
                flush=True,
            )

        result = self._restore_best(directory, history)
        write_json(directory / "result.json", result)
        return result

    def _train_epoch(self, batches: Iterable[Batch], epoch: int) -> dict[str, Any]:
        self.model.train()
        row_count = 0
        gradient_norms: list[float] = []
        task_totals: dict[str, dict[str, float]] = {}
        for step, batch in enumerate(batches):
            if self.config.steps is not None and step >= self.config.steps:
                break
            self.optimizer.zero_grad(set_to_none=True)
            moved = {
                role: {n: v.to(self.config.device) for n, v in values.items()}
                for role, values in batch.items()
            }
            outputs = self.model(moved["features"])
            loss, diagnostics = self.loss(outputs, moved)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}, step {step}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf"), error_if_nonfinite=True
            )
            self.optimizer.step()
            row_count += len(next(iter(outputs.values())))
            gradient_norms.append(float(norm))
            for name, values in diagnostics.items():
                total = task_totals.setdefault(
                    name,
                    {
                        "numerator": 0.0,
                        "denominator": 0.0,
                        "valid_count": 0.0,
                        "effective_count": 0.0,
                        "task_weight": values["task_weight"],
                    },
                )
                if total["task_weight"] != values["task_weight"]:
                    raise ValueError("Task weights cannot change within an epoch")
                for key in ("numerator", "denominator", "valid_count", "effective_count"):
                    total[key] += values[key]
        if row_count == 0 or not task_totals:
            raise ValueError("Empty training epoch or missing loss reductions")
        return {
            "tasks": task_totals,
            "train_loss": sum(
                v["task_weight"] * v["numerator"] / v["denominator"]
                for v in task_totals.values()
                if v["denominator"] > 0
            ),
            "train_rows": row_count,
            "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
        }

    def _resume(
        self, source: Path, destination: Path, resume_identity: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], float]:
        """Continue CPU training into a fresh directory, preserving prior selection."""
        if self.config.device != "cpu":
            raise ValueError("Exact continuation currently supports CPU RNG only")
        last = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
        previous_config = dict(last["training"])
        current_config = asdict(self.config)
        previous_config.pop("epochs")
        current_config.pop("epochs")
        if previous_config != current_config:
            raise ValueError("Continuation may change only the epoch cap")
        if last["resume_identity"] != resume_identity:
            raise ValueError("Continuation resume_contract/metadata mismatch")
        history = json.loads((source / "history.json").read_text())
        if [row["epoch"] for row in history] != list(range(last["epoch"] + 1)):
            raise ValueError("History must contain every epoch through the last checkpoint")
        if self.config.epochs <= len(history):
            raise ValueError("Continuation requires a larger epoch cap")
        if last["val"] != history[-1]["val"]:
            raise ValueError("Last checkpoint disagrees with validation history")
        best = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        best_config = dict(best["training"])
        best_config.pop("epochs")  # The best epoch may predate an earlier continuation.
        if best_config != current_config or best["resume_identity"] != resume_identity:
            raise ValueError("Best checkpoint training/resume contract mismatch")
        metric = self.config.selection_metric
        scores = [row["val"][metric] for row in history]
        if any(score is None or not math.isfinite(score) for score in scores):
            raise ValueError("History contains an undefined selection metric")
        expected = min(scores) if self.config.selection_mode == "min" else max(scores)
        if best["epoch"] != scores.index(expected) or best["val"] != history[best["epoch"]]["val"]:
            raise ValueError("Best checkpoint disagrees with selection history")
        self.model.load_state_dict(last["model"])
        self.optimizer.load_state_dict(last["optimizer"])
        torch.set_rng_state(last["torch_rng"])
        shutil.copy2(source / "best.pt", destination / "best.pt")
        return history, expected

    def _checkpoint(
        self,
        epoch: int,
        validation: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "val": validation,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training": asdict(self.config),
            "metadata": metadata,
            "resume_identity": metadata,
            "torch_rng": torch.get_rng_state(),
        }

    def _restore_best(self, directory: Path, history: list[dict[str, Any]]) -> dict[str, Any]:
        selected = torch.load(
            directory / "best.pt", map_location=self.config.device, weights_only=True
        )
        self.model.load_state_dict(selected["model"])
        best_epoch = selected["epoch"]
        neighbors = history[max(0, best_epoch - 1) : best_epoch + 2]
        return {
            "best_epoch": best_epoch,
            "val": selected["val"],
            "seed": self.config.seed,
            "checkpoint_policy": {
                "metric": self.config.selection_metric,
                "mode": self.config.selection_mode,
                "tie_break": "earliest epoch",
            },
            "neighbor_selection_values": [
                row["val"][self.config.selection_metric] for row in neighbors
            ],
            "last_selection_value": history[-1]["val"][self.config.selection_metric],
            "exploratory": self.config.steps is not None,
        }
