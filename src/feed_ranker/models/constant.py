"""Training-label mean and median predictors for scalar regression."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConstantRegressor(nn.Module):
    """Intercept-only baseline: mean minimizes squared error, median absolute error.

    Reference: Hastie, Tibshirani & Friedman, The Elements of Statistical Learning,
    2nd ed., section 2.4 (squared-error and absolute-error prediction).
    Inputs follow the ranker feature contract; output has shape [B, 1].
    """

    def __init__(self, prediction: float) -> None:
        super().__init__()
        if not math.isfinite(prediction):
            raise ValueError("Constant prediction must be finite")
        self.register_buffer("prediction", torch.tensor(prediction, dtype=torch.float32))
        if not torch.isfinite(self.prediction):
            raise ValueError("Constant prediction must be representable as float32")

    @classmethod
    def fit(cls, labels: torch.Tensor, statistic: str) -> ConstantRegressor:
        """Fit once on training labels; even-sized medians average the middle values."""
        if labels.ndim != 1 or labels.numel() == 0 or not torch.isfinite(labels).all():
            raise ValueError("Expected a nonempty finite vector of training labels")
        values = labels.detach().double()
        if statistic == "mean":
            prediction = values.mean()
        elif statistic == "median":
            prediction = torch.quantile(values, 0.5)
        else:
            raise ValueError("statistic must be mean or median")
        return cls(prediction.item())

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("At least one feature is needed to determine batch size")
        rows = len(next(iter(features.values())))
        return self.prediction.expand(rows, 1)
