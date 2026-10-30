"""Explicit named binary supervision, independent of model computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from feed_ranker.data.dataset import Batch


@dataclass(frozen=True)
class TaskLossConfig:
    output: str
    target: str
    loss: str = "bce_logits"
    mask: str | None = None
    weight: str | None = None
    task_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.output or not self.target or self.loss != "bce_logits":
            raise ValueError("Require output/target names and the supported BCE loss")
        if not math.isfinite(self.task_weight) or self.task_weight <= 0:
            raise ValueError("Task weight must be finite and positive")


class Objective(nn.Module):
    def __init__(self, tasks: tuple[TaskLossConfig, ...]) -> None:
        super().__init__()
        if not tasks or len({t.output for t in tasks}) != len(tasks):
            raise ValueError("Require unique configured task outputs")
        self.tasks = tasks

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: Batch
    ) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
        total = None
        diagnostics = {}
        for task in self.tasks:
            if task.output not in outputs or task.target not in batch.get("targets", {}):
                raise ValueError(f"Missing output/target for {task.output}")
            prediction, target = outputs[task.output], batch["targets"][task.target]
            if (
                prediction.ndim != 1
                or target.shape != prediction.shape
                or not target.is_floating_point()
            ):
                raise ValueError(f"{task.output}: require aligned floating [B] outputs/targets")
            mask = (
                torch.ones_like(target, dtype=torch.bool)
                if task.mask is None
                else batch.get("target_masks", {}).get(task.mask)
            )
            weight = (
                torch.ones_like(target)
                if task.weight is None
                else batch.get("target_weights", {}).get(task.weight)
            )
            if mask is None or mask.dtype != torch.bool or mask.shape != target.shape:
                raise ValueError(f"{task.output}: invalid validity mask")
            if (
                weight is None
                or weight.shape != target.shape
                or not torch.isfinite(weight).all()
                or (weight < 0).any()
            ):
                raise ValueError(f"{task.output}: invalid example weights")
            active = mask & weight.gt(0)
            denominator = weight[active].sum()
            valid_count = int(mask.sum().item())
            if active.any():
                y, p, w = target[active], prediction[active], weight[active]
                if not torch.isfinite(p).all() or not ((y == 0) | (y == 1)).all():
                    raise ValueError(f"{task.output}: invalid active binary targets/logits")
                numerator = (F.binary_cross_entropy_with_logits(p, y, reduction="none") * w).sum()
                loss = numerator / denominator
                term = loss * task.task_weight
                total = term if total is None else total + term
                numerator_value, loss_value = float(numerator.detach()), float(loss.detach())
            else:
                numerator_value = loss_value = 0.0
            diagnostics[task.output] = {
                "task_weight": task.task_weight,
                "numerator": numerator_value,
                "denominator": float(denominator),
                "valid_count": valid_count,
                "effective_count": int(active.sum()),
                "loss": loss_value,
            }
        if total is None:
            raise ValueError("All configured tasks are inactive")
        if total.ndim != 0 or not torch.isfinite(total):
            raise ValueError("Objective must return one finite scalar")
        return total, diagnostics
