"""Optimization and checkpoint selection over supplied training instances.

The trainer knows no dataset paths, feature recipes, artifact formats or funnel
policies. It iterates supplied loaders, evaluates metrics, and saves checkpoints.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from feed_ranker.data.dataset import Batch, EpochBatchSampler
from feed_ranker.evaluation.predictions import PredictionEvaluator
from feed_ranker.persistence import write_json
from feed_ranker.training.checkpoint import resume_training, save_checkpoint
from feed_ranker.training.objectives import Objective
from feed_ranker.training.optimizer import optimizer_contract


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15  # reasonable default, not from any specific production config
    seed: int = 42  # recorded initialization/sampling seed; orchestration applies it
    device: str = "cpu"
    steps: int | None = None
    selection_metric: str = "long_view/roc_auc"
    selection_mode: str = "max"

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
    """Optimize the supplied model, leaving it at the final epoch on return.

    The caller builds the model, optimizer and objective. fit() owns the epoch
    loop, history and checkpoints; the job explicitly loads best weights before
    export and writes its result once. No data/artifact builders or callbacks.
    """

    def __init__(
        self, model: nn.Module, optimizer: Optimizer, objective: Objective, config: TrainConfig
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.config = config

    def fit(
        self,
        train_loader: Iterable[Batch],
        validation_loader: Iterable[Batch],
        evaluator: PredictionEvaluator,
        *,
        checkpoint_dir: Path,
        run_identity: dict[str, Any],
        train_eval_loader: Iterable[Batch] | None = None,
        resume_from: Path | None = None,
    ) -> dict[str, Any]:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        training_config = {
            **asdict(self.config),
            "optimizer": optimizer_contract(self.model, self.optimizer),
        }
        history = []
        if resume_from is not None:
            history = resume_training(
                resume_from,
                checkpoint_dir,
                self.model,
                self.optimizer,
                training_config,
                run_identity,
            )
        metric, mode = self.config.selection_metric, self.config.selection_mode
        scores = [row["val"][metric] for row in history]
        best_score = math.inf if mode == "min" else -math.inf
        if scores:
            best_score = min(scores) if mode == "min" else max(scores)

        for epoch in range(len(history), self.config.epochs):
            started = time.monotonic()
            if isinstance(train_loader, DataLoader) and isinstance(
                train_loader.sampler, EpochBatchSampler
            ):
                train_loader.sampler.set_epoch(epoch)
            training = self._train_epoch(train_loader, epoch)
            training_seconds = time.monotonic() - started
            eval_started = time.monotonic()
            train_evaluation = (
                None
                if train_eval_loader is None
                else evaluator.evaluate(self.model, train_eval_loader)
            )
            train_eval_seconds = time.monotonic() - eval_started
            eval_started = time.monotonic()
            validation = evaluator.evaluate(self.model, validation_loader)
            validation_seconds = time.monotonic() - eval_started
            score = validation[metric]
            if score is None or not math.isfinite(score):
                raise ValueError(f"Selection metric {metric!r} is undefined")
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
                }
            )
            checkpoint_started = time.monotonic()
            selected_state = {
                "epoch": epoch,
                "val": validation,
                "model": self.model.state_dict(),
                "training": training_config,
                "run_identity": run_identity,
            }
            save_checkpoint(
                checkpoint_dir / "last.pt",
                {
                    **selected_state,
                    "optimizer": self.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                },
            )
            # Strict improvement retains the earliest checkpoint on an exact tie.
            improved = score < best_score if mode == "min" else score > best_score
            if improved:
                best_score = score
                save_checkpoint(checkpoint_dir / "best.pt", selected_state)
            history[-1]["checkpoint_seconds"] = time.monotonic() - checkpoint_started
            history[-1]["seconds"] = time.monotonic() - started
            write_json(checkpoint_dir.parent / "history.json", history)
            print(
                f"epoch={epoch} train={training['train_loss']:.6f} val_{metric}={score:.6f}",
                flush=True,
            )

        scores = [row["val"][metric] for row in history]
        best_epoch = scores.index(best_score)
        return {
            "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "val": history[best_epoch]["val"],
            "seed": self.config.seed,
            "checkpoint_policy": {"metric": metric, "mode": mode, "tie_break": "earliest epoch"},
            "neighbor_selection_values": scores[max(0, best_epoch - 1) : best_epoch + 2],
            "last_selection_value": scores[-1],
            "exploratory": self.config.steps is not None,
        }

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
            loss, diagnostics = self.objective(outputs, moved)
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
