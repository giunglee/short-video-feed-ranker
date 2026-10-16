"""Scalar regression evaluation; metric sampling is independent of training RNG.

Error metrics weight rows equally; user XAUC weights eligible users equally.
Only ranking metrics retain all predictions; MAE/MSE accumulate by batch.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import nn

from feed_ranker.data.schema import RecommendationBatch
from feed_ranker.training.metrics import user_xauc, xauc

METRICS = ("mae", "mse", "xauc", "user_xauc")


class RegressionEvaluator:
    def __init__(
        self,
        loaders: Mapping[str, Iterable[RecommendationBatch]],
        metrics: tuple[str, ...],
        device: str,
        metric_seed: int,
        xauc_pairs: int,
        user_pairs: int,
    ) -> None:
        if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(METRICS):
            raise ValueError(f"metrics must be a nonempty unique subset of {METRICS}")
        self.loaders = loaders
        self.metrics = metrics
        self.device = device
        self.metric_seed = metric_seed
        self.xauc_pairs = xauc_pairs
        self.user_pairs = user_pairs

    @torch.no_grad()
    def evaluate(self, model: nn.Module, split: str) -> dict[str, Any]:
        model.eval()
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        users: list[torch.Tensor] = []
        error_sums = {name: 0.0 for name in self.metrics if name in ("mae", "mse")}
        retain_predictions = bool(set(self.metrics) & {"xauc", "user_xauc"})
        row_count = 0

        for batch in self.loaders[split]:
            labels = batch["label"]
            predicted = predict_regression(model, batch["feature"], len(labels), self.device).cpu()
            residual = predicted.double() - labels.double()
            if "mae" in error_sums:
                error_sums["mae"] += residual.abs().sum().item()
            if "mse" in error_sums:
                error_sums["mse"] += residual.square().sum().item()
            row_count += len(labels)
            if retain_predictions:
                predictions.append(predicted)
                targets.append(labels)
            if "user_xauc" in self.metrics:
                users.append(batch["raw_user_id"])

        if row_count == 0:
            raise ValueError(f"Empty {split} split")
        result: dict[str, Any] = {name: total / row_count for name, total in error_sums.items()}
        if retain_predictions:
            predicted, labels = torch.cat(predictions), torch.cat(targets)
            if "xauc" in self.metrics:
                result["xauc"] = xauc(predicted, labels, self.xauc_pairs, self.metric_seed)
            if "user_xauc" in self.metrics:
                result.update(
                    user_xauc(
                        predicted, labels, torch.cat(users), self.user_pairs, self.metric_seed
                    )
                )
        result["rows"] = row_count
        return result


def predict_regression(
    model: nn.Module, features: dict[str, torch.Tensor], rows: int, device: str
) -> torch.Tensor:
    """Accept [B] or [B, 1] scalar predictions; preserve gradients and model mode."""
    inputs = {name: value.to(device) for name, value in features.items()}
    predicted = model(inputs)
    if predicted.shape not in ((rows,), (rows, 1)):
        raise ValueError(f"Expected predictions [B] or [B, 1]; got {tuple(predicted.shape)}")
    if not torch.isfinite(predicted).all():
        raise ValueError("Nonfinite model predictions")
    return predicted.reshape(-1)
