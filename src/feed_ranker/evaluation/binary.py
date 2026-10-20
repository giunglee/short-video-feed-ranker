"""Exact event-level binary metrics for logits, separate from watch-ratio metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from feed_ranker.data.schema import RecommendationBatch
from feed_ranker.evaluation.ranking import predict_regression


def binary_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """ROC AUC uses average ranks for ties; undefined one-class AUC is None."""
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != logits.shape
        or not len(labels)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Require aligned nonempty binary labels and finite logits")
    probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    ranks = pd.Series(logits).rank(method="average").to_numpy()
    auc = (
        ((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        if positives and negatives
        else None
    )
    return {
        "rows": len(labels),
        "positive_rate": float(labels.mean()),
        "mean_probability": float(probability.mean()),
        "logloss": float((np.logaddexp(0, logits) - labels * logits).mean()),
        "brier": float(np.square(probability - labels).mean()),
        "roc_auc": None if auc is None else float(auc),
    }


@torch.no_grad()
def predict_binary(
    model: nn.Module,
    batches: Iterable[RecommendationBatch],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, logits = [], []
    for batch in batches:
        labels.append(batch["label"].numpy())
        logits.append(
            predict_regression(model, batch["feature"], len(batch["label"]), "cpu").numpy()
        )
    if not labels:
        raise ValueError("Cannot evaluate empty binary dataset")
    return np.concatenate(labels), np.concatenate(logits)


class BinaryEvaluator:
    """CPU evaluator consumed by the existing Trainer; does not read a test split."""

    def __init__(self, validation: Callable[[], Iterable[RecommendationBatch]]) -> None:
        self.validation = validation

    def evaluate(self, model: nn.Module, split: str) -> dict[str, Any]:
        if split != "val":
            raise ValueError("This development runner exposes validation only")
        return binary_metrics(*predict_binary(model, self.validation()))


def candidate_metrics(
    ordered: np.ndarray,
    positive: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, float | None]:
    """Full warm-observed-pool denominators, including positives missed by retrieval.

    Empty returns score zero when positives exist; zero-positive queries are undefined.
    The oracle can reorder only the returned candidates, with the same full-pool IDCG.
    """
    if k < 1 or len(np.unique(ordered)) != len(ordered):
        raise ValueError("Require positive k and unique candidates")
    total = len(np.unique(positive))
    if not total:
        return {"recall": None, "ndcg": None, "oracle_ndcg": None}
    relevance = np.isin(ordered, positive)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[: min(total, k)].sum()
    hits = int(relevance.sum())
    return {
        "recall": hits / total,
        "ndcg": float((relevance[:k] * discounts[: min(k, len(ordered))]).sum() / ideal),
        "oracle_ndcg": float(discounts[: min(hits, k)].sum() / ideal),
    }
