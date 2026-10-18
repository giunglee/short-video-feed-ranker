"""Target transforms for scalar regression; predictions remain unconstrained."""

from __future__ import annotations

import math

import torch
from torch import nn


class TargetMAE(nn.Module):
    """Transform targets only; raw batch labels and unconstrained predictions survive."""

    def __init__(self, cap: float | None = None) -> None:
        super().__init__()
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("Target cap must be finite and positive")
        self.cap = cap

    def forward(self, predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if (
            predicted.shape != labels.shape
            or not torch.isfinite(labels).all()
            or (labels < 0).any()
        ):
            raise ValueError("MAE requires matching shapes and finite nonnegative labels")
        target = labels if self.cap is None else labels.clamp(max=self.cap)
        return torch.nn.functional.l1_loss(predicted, target)
