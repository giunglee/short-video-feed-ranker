"""
Generic train/eval loop — forward, loss, backward, step, metric logging.

Batch contract (v1, single-task):
  feature: dict[str, Tensor]
  label:   Tensor [B]

Metrics:
  - Always logs loss + MAE
  - AUC only when TrainConfig.compute_auc=True (binary / CTR tasks)
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchmetrics.functional.classification import binary_auroc

from training.metrics import xauc as compute_xauc_metric
from training.optimizer import build_optimizer, OptimizerConfig


@dataclass
class TrainConfig:
    batch_size: int = 100
    epochs: int = 100
    # None = one full shuffled pass over the train split per epoch (steps
    # auto-computed as len(train) // batch_size). Set to an int to cap each
    # epoch at fewer batches (e.g. for a quick smoke test on a big dataset).
    steps: int | None = 10
    eval_batch_size: int = 4096
    device: torch.device = (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    optimizer_config: OptimizerConfig = OptimizerConfig()
    # Regression (watch_ratio): compute_auc=False. Binary CTR: compute_auc=True.
    compute_auc: bool = False
    # Pairwise ranking-concordance metric for continuous regression targets
    # (see training/metrics.py). Cheap (fixed sample size), safe to leave on.
    compute_xauc: bool = False
    xauc_num_pairs: int = 100_000
    xauc_seed: int = 0
    primary_metric: Literal["mae", "auc"] = "mae"
    show_plots: bool = False
    # If True, restore the model's weights from the epoch with the best
    # primary_metric on val before returning from fit() (and before any
    # subsequent evaluate_split("test") call). Without this, self.model ends
    # fit() holding whichever epoch happened to run *last* -- which is fine
    # if val is still improving at the final epoch, but silently returns an
    # overfit/worse checkpoint if val degrades in later epochs (e.g. after
    # increasing `epochs` without also re-checking for overfitting).
    restore_best_weights: bool = True


@dataclass
class EvalMetrics:
    loss: float
    mae: float
    auc: float | None = None
    xauc: float | None = None


@dataclass
class FitSummary:
    train_losses: list[float]
    val_losses: list[float]
    grad_norms: list[float]
    weight_norms: list[float]
    maes: list[float]
    aucs: list[float]
    xaucs: list[float] = None  # type: ignore[assignment]
    best_epoch: int = -1
    restored_best_weights: bool = False
    # AUC/XAUC recorded *at* best_epoch (the restored checkpoint) -- not the
    # same as best_val_auc/best_val_xauc below when primary_metric="mae",
    # since those track each metric's own best epoch, which may differ from
    # the epoch actually selected/restored by primary_metric.
    best_epoch_auc: float | None = None
    best_epoch_xauc: float | None = None

    def __post_init__(self) -> None:
        if self.xaucs is None:
            self.xaucs = []

    @property
    def best_val_mae(self) -> float:
        return min(self.maes)

    @property
    def final_val_mae(self) -> float:
        return self.maes[-1]

    @property
    def best_val_auc(self) -> float | None:
        return max(self.aucs) if self.aucs else None

    @property
    def final_val_auc(self) -> float | None:
        return self.aucs[-1] if self.aucs else None

    @property
    def best_val_xauc(self) -> float | None:
        return max(self.xaucs) if self.xaucs else None

    @property
    def final_val_xauc(self) -> float | None:
        return self.xaucs[-1] if self.xaucs else None

    @property
    def final_val_loss(self) -> float:
        return self.val_losses[-1]


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        data_loader: Any,
        loss_fn: nn.Module,
    ):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.optimizer: torch.optim.Optimizer | None = None
        self._val_data: dict[str, Any] | None = None

    def build(self) -> Trainer:
        self.optimizer, _ = build_optimizer(self.model, self.config.optimizer_config)
        self.model.to(self.config.device)
        self._val_data = self.data_loader.get_full(split="val")
        return self

    def fit(self) -> FitSummary:
        if self.optimizer is None:
            self.build()

        train_losses: list[float] = []
        val_losses: list[float] = []
        grad_norms: list[float] = []
        weight_norms: list[float] = []
        maes: list[float] = []
        aucs: list[float] = []
        xaucs: list[float] = []

        best_epoch = -1
        best_score: float | None = None
        best_state_dict: dict[str, Any] | None = None
        best_epoch_auc: float | None = None
        best_epoch_xauc: float | None = None

        def _is_better(score: float, best: float | None) -> bool:
            if best is None:
                return True
            return score < best if self.config.primary_metric == "mae" else score > best

        for epoch in range(self.config.epochs):
            start_time = datetime.now()
            train_loss_accum = 0.0
            grad_norm_accum = 0.0
            weight_norm_accum = 0.0

            self.model.train()
            # One shuffled pass over train per epoch (no replacement within
            # the epoch) unless config.steps caps it shorter -- guarantees
            # full data coverage per epoch, unlike resampling a fresh random
            # batch each step (IID with replacement across the whole run,
            # which under-covers the data at a fixed step budget).
            train_iter = self.data_loader.epoch_batches(
                self.config.batch_size, split="train"
            )
            if self.config.steps is not None:
                train_iter = itertools.islice(train_iter, self.config.steps)

            n_steps = 0
            for batch in train_iter:
                assert self.optimizer is not None
                self.optimizer.zero_grad()

                feature = self._to_device(batch["feature"])
                label = self._to_device(batch["label"]).view(-1).float()
                logits = self.model(feature)
                loss = self.loss_fn(logits.view(-1), label)
                loss.backward()
                train_loss_accum += loss.item()

                grad_norm_accum += torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float("inf")
                ).item()

                self.optimizer.step()

                weight_norm_accum += torch.nn.utils.get_total_norm(
                    list(self.model.parameters())
                ).item()
                n_steps += 1

            end_time = datetime.now()
            avg_train_loss = train_loss_accum / n_steps
            avg_grad_norm = grad_norm_accum / n_steps
            avg_weight_norm = weight_norm_accum / n_steps
            train_losses.append(avg_train_loss)
            grad_norms.append(avg_grad_norm)
            weight_norms.append(avg_weight_norm)

            val_metrics = self.evaluate_split("val")
            val_losses.append(val_metrics.loss)
            maes.append(val_metrics.mae)
            if val_metrics.auc is not None:
                aucs.append(val_metrics.auc)
            if val_metrics.xauc is not None:
                xaucs.append(val_metrics.xauc)

            score = val_metrics.mae if self.config.primary_metric == "mae" else val_metrics.auc
            if score is not None and _is_better(score, best_score):
                best_score = score
                best_epoch = epoch
                best_epoch_auc = val_metrics.auc
                best_epoch_xauc = val_metrics.xauc
                if self.config.restore_best_weights:
                    best_state_dict = copy.deepcopy(self.model.state_dict())

            metric_bits = f"val MAE: {val_metrics.mae:.4f}"
            if val_metrics.auc is not None:
                metric_bits += f", val AUC: {val_metrics.auc:.4f}"
            if val_metrics.xauc is not None:
                metric_bits += f", val XAUC: {val_metrics.xauc:.4f}"

            print(
                f"epoch: {epoch}, batch_size: {self.config.batch_size}, "
                f"steps: {n_steps}, elapsed_time:{end_time - start_time}, "
                f"train loss: {avg_train_loss}, val loss: {val_metrics.loss}, "
                f"{metric_bits}, grad norm: {avg_grad_norm}, "
                f"weight norm: {avg_weight_norm}"
            )

        restored = False
        if self.config.restore_best_weights and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            restored = True
            print(
                f"Restored best checkpoint: epoch {best_epoch} "
                f"(val {self.config.primary_metric}={best_score:.4f}); "
                f"final epoch was {self.config.epochs - 1}."
            )

        if self.config.show_plots:
            self._plot_eval_metrics(
                train_losses=train_losses,
                val_losses=val_losses,
                grad_norms=grad_norms,
                weight_norms=weight_norms,
                maes=maes,
                aucs=aucs,
            )
        return FitSummary(
            train_losses=train_losses,
            val_losses=val_losses,
            grad_norms=grad_norms,
            weight_norms=weight_norms,
            maes=maes,
            aucs=aucs,
            xaucs=xaucs,
            best_epoch=best_epoch,
            restored_best_weights=restored,
            best_epoch_auc=best_epoch_auc,
            best_epoch_xauc=best_epoch_xauc,
        )

    @torch.no_grad()
    def evaluate_split(
        self,
        split: str = "val",
    ) -> EvalMetrics:
        self.model.eval()
        if split == "val" and self._val_data is None:
            self._val_data = self.data_loader.get_full(split="val")

        eval_data = (
            self._val_data if split == "val" else self.data_loader.get_full(split=split)
        )
        assert eval_data is not None

        features = eval_data["feature"]
        labels = eval_data["label"]
        n = len(labels)
        eval_bs = self.config.eval_batch_size

        loss_accum = 0.0
        mae_accum = 0.0
        logits_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        for start in range(0, n, eval_bs):
            end = min(start + eval_bs, n)
            batch_feature = self._to_device(self._slice_dict(features, start, end))
            batch_label = self._to_device(labels[start:end]).view(-1).float()
            batch_len = end - start

            preds = self.model(batch_feature).view(-1)
            loss = self.loss_fn(preds, batch_label)
            loss_accum += loss.item() * batch_len
            mae_accum += torch.abs(preds - batch_label).sum().item()

            if self.config.compute_auc or self.config.compute_xauc:
                logits_chunks.append(preds.detach().cpu())
                label_chunks.append(batch_label.detach().cpu())

        auc: float | None = None
        xauc: float | None = None
        if self.config.compute_auc or self.config.compute_xauc:
            all_preds = torch.cat(logits_chunks)
            all_labels = torch.cat(label_chunks)
            if self.config.compute_auc:
                auc = binary_auroc(all_preds, all_labels.long()).item()
            if self.config.compute_xauc:
                xauc = compute_xauc_metric(
                    all_preds,
                    all_labels,
                    num_pairs=self.config.xauc_num_pairs,
                    seed=self.config.xauc_seed,
                )

        return EvalMetrics(loss=loss_accum / n, mae=mae_accum / n, auc=auc, xauc=xauc)

    def _to_device(self, obj: Any) -> Any:
        """Recursively move nested dicts of tensors to TrainConfig.device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.config.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _slice_dict(
        data: dict[str, torch.Tensor], start: int, end: int
    ) -> dict[str, torch.Tensor]:
        return {k: v[start:end] for k, v in data.items()}

    def _plot_eval_metrics(
        self,
        train_losses: list[float],
        val_losses: list[float],
        grad_norms: list[float],
        weight_norms: list[float],
        maes: list[float],
        aucs: list[float],
    ) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="red")
        plt.title("Training and Validation Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 5))
        plt.plot(maes, label="Validation MAE", color="purple")
        plt.title("Validation MAE")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)

        if aucs:
            plt.figure(figsize=(10, 5))
            plt.plot(aucs, label="Validation AUCs", color="red")
            plt.title("Validation AUCs")
            plt.xlabel("Epochs")
            plt.ylabel("AUCs")
            plt.legend()
            plt.grid(True)

        plt.figure(figsize=(10, 5))
        plt.plot(grad_norms, label="Grad Norm", color="green")
        plt.title("Gradient Norm Curve")
        plt.xlabel("Epochs")
        plt.ylabel("L2 Norm")
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 5))
        plt.plot(weight_norms, label="Weight Norm", color="orange")
        plt.title("Weight Norm Curve")
        plt.xlabel("Epochs")
        plt.ylabel("L2 Norm")
        plt.legend()
        plt.grid(True)

        plt.show()
