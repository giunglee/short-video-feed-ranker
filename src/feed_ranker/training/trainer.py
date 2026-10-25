"""Scalar-loss training with explicit validation metrics and checkpoint selection.

The runner supplies the loss and evaluator. Test is never read during fitting.
Training loss is measured during updates, not by rescoring a frozen checkpoint.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import torch
from torch import nn

from feed_ranker.artifacts import save_checkpoint, write_json
from feed_ranker.data.schema import RecommendationBatch
from feed_ranker.evaluation.ranking import predict_regression
from feed_ranker.training.optimizer import OptimizerConfig, build_optimizer


class ModelEvaluator(Protocol):
    """An evaluator owns metric definitions and split-specific prediction inputs."""

    def evaluate(self, model: nn.Module, split: str) -> dict[str, Any]: ...


@dataclass
class TrainConfig:
    # reasonable default, not from any specific production config
    epochs: int = 15
    batch_size: int = 1024
    eval_batch_size: int = 8192
    train_eval_rows: int = 100_000
    train_eval_seed: int = 2026
    seed: int = 42
    metric_seed: int = 0
    xauc_pairs: int = 100_000
    user_pairs: int = 256
    device: str = "cpu"
    steps: int | None = None
    loss: str = "mae"
    weighting: str = "event"
    metrics: tuple[str, ...] = ("mae", "xauc", "user_xauc")
    selection_metric: str = "mae"
    selection_mode: str = "min"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if type(self.train_eval_rows) is not int or self.train_eval_rows < 0:
            raise ValueError("train_eval_rows must be nonnegative; zero disables the diagnostic")
        self.metrics = tuple(self.metrics)
        if self.weighting not in ("event", "pair"):
            raise ValueError("weighting must be event or pair")
        if self.loss not in ("mae", "mse", "smooth_l1", "bce_logits"):
            raise ValueError("loss must be mae, mse, smooth_l1, or bce_logits")
        if (
            not self.metrics
            or len(set(self.metrics)) != len(self.metrics)
            or any(not isinstance(name, str) or not name.strip() for name in self.metrics)
        ):
            raise ValueError(
                "metrics must contain nonempty unique names; evaluator validates support"
            )
        if self.selection_metric not in self.metrics:
            raise ValueError("selection_metric must be included in metrics")
        if self.selection_mode not in ("min", "max"):
            raise ValueError("selection_mode must be min or max")
        positive_fields = ("epochs", "batch_size", "eval_batch_size", "xauc_pairs", "user_pairs")
        for name in positive_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps is not None and (type(self.steps) is not int or self.steps <= 0):
            raise ValueError("steps must be a positive integer or None")
        for name in ("seed", "metric_seed", "train_eval_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")


class Trainer:
    """Own one run; loss_fn must return a scalar mean loss over the batch."""

    def __init__(
        self,
        model: nn.Module,
        train_batches: Callable[[int], Iterable[RecommendationBatch]],
        config: TrainConfig,
        *,
        loss_fn: nn.Module,
        evaluator: ModelEvaluator,
    ) -> None:
        self.model = model.to(config.device)
        self.train_batches = train_batches
        self.config = config
        self.loss_fn = loss_fn.to(config.device)
        self.evaluator = evaluator
        self.optimizer = build_optimizer(self.model, config.optimizer)

    def fit(
        self, directory: Path, metadata: dict[str, Any], *, resume_from: Path | None = None
    ) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", {"training": asdict(self.config), **metadata})
        history: list[dict[str, Any]] = []
        best_score = math.inf if self.config.selection_mode == "min" else -math.inf

        if resume_from is not None:
            history, best_score = self._resume(resume_from, directory, metadata)

        for epoch in range(len(history), self.config.epochs):
            started = time.monotonic()
            training = self._train_epoch(epoch)
            training_seconds = time.monotonic() - started
            eval_started = time.monotonic()
            train_evaluation = None
            if "train_diagnostic" in getattr(self.evaluator, "loaders", {}):
                train_evaluation = self.evaluator.evaluate(self.model, "train_diagnostic")
            train_eval_seconds = time.monotonic() - eval_started
            eval_started = time.monotonic()
            validation = self.evaluator.evaluate(self.model, "val")
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
            print(
                f"epoch={epoch} train={training['train_loss']:.6f} "
                f"val_{self.config.selection_metric}={score:.6f}",
                flush=True,
            )

        result = self._restore_best(directory, history)
        write_json(directory / "result.json", result)
        return result

    def _resume(
        self, source: Path, destination: Path, metadata: dict[str, Any]
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
        for key in (
            "data_manifest_sha256",
            "model_name",
            "hidden",
            "dropout",
            "use_ids",
            "feature_set",
            "feature_config",
            "side_features_sha256",
        ):
            if last["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"Continuation metadata mismatch: {key}")
        history = json.loads((source / "history.json").read_text())
        if [row["epoch"] for row in history] != list(range(last["epoch"] + 1)):
            raise ValueError("History must contain every epoch through the last checkpoint")
        if self.config.epochs <= len(history):
            raise ValueError("Continuation requires a larger epoch cap")
        best = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        metric = self.config.selection_metric
        scores = [row["val"][metric] for row in history]
        expected = min(scores) if self.config.selection_mode == "min" else max(scores)
        if best["val"][metric] != expected or best["epoch"] != scores.index(expected):
            raise ValueError("Best checkpoint disagrees with selection history")
        self.model.load_state_dict(last["model"])
        self.optimizer.load_state_dict(last["optimizer"])
        torch.set_rng_state(last["torch_rng"])
        shutil.copy2(source / "best.pt", destination / "best.pt")
        return history, expected

    def _train_epoch(self, epoch: int) -> dict[str, float | int]:
        self.model.train()
        loss_sum = 0.0
        row_count = 0
        weight_total = 0.0
        gradient_norms: list[float] = []
        batches = self.train_batches(epoch)
        for step, batch in enumerate(batches):
            if self.config.steps is not None and step >= self.config.steps:
                break
            self.optimizer.zero_grad(set_to_none=True)
            labels = batch["label"].to(self.config.device)
            predicted = predict_regression(
                self.model, batch["feature"], len(labels), self.config.device
            )
            loss = self.loss_fn(predicted, labels)
            batch_weight = float(len(labels))
            if self.config.weighting == "pair":
                if loss.shape != labels.shape or "pair_weight" not in batch:
                    raise ValueError(
                        "Pair weighting requires per-event losses and prepared pair weights"
                    )
                weights = batch["pair_weight"].to(self.config.device)
                if (
                    weights.shape != labels.shape
                    or not torch.isfinite(weights).all()
                    or (weights <= 0).any()
                ):
                    raise ValueError("Invalid pair weights")
                batch_weight = float(weights.sum())
                loss = (loss * weights).sum() / weights.sum()
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}, step {step}")
            loss.backward()
            # Infinity measures the norm and checks finiteness without clipping.
            norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf"), error_if_nonfinite=True
            )
            self.optimizer.step()
            loss_sum += loss.item() * batch_weight
            weight_total += batch_weight
            row_count += len(labels)
            gradient_norms.append(float(norm))

        if row_count == 0:
            raise ValueError("Empty training epoch")
        return {
            "train_loss": loss_sum / weight_total,
            "train_rows": row_count,
            "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
        }

    def _checkpoint(
        self, epoch: int, validation: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "val": validation,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training": asdict(self.config),
            "metadata": metadata,
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
