"""Prediction collection and metrics usable independently of a training job."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.data.instances import Batch
from feed_ranker.evaluation.metrics import binary_metrics, user_auc_metrics


@dataclass(frozen=True)
class Predictions:
    """Aligned records for one binary output, including explicit excluded rows."""

    labels: np.ndarray
    logits: np.ndarray
    users: np.ndarray
    warm: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or any(
            v.shape != self.labels.shape for v in (self.logits, self.users, self.warm, self.valid)
        ):
            raise ValueError("Predictions and labels must align as [N]")
        if self.valid.dtype != np.bool_ or self.warm.dtype != np.bool_:
            raise ValueError("Validity and cohort flags must be boolean")
        if self.users.dtype.kind not in "iu" or (self.users[self.valid] < 0).any():
            raise ValueError("Active user IDs must be nonnegative integers")
        if (
            not np.isfinite(self.logits[self.valid]).all()
            or not np.isin(self.labels[self.valid], [0, 1]).all()
        ):
            raise ValueError("Active predictions must be finite and labels binary")


def evaluate_predictions(predictions: Mapping[str, Predictions]) -> dict[str, Any]:
    """Compute full-population metrics from records, without a model or loader.

    Labels/logits may come from loader inference or persisted offline predictions.
    Never average batch AUCs. Optimization weights do not redefine this population.
    """
    if not predictions:
        raise ValueError("Require prediction records")
    result = {}
    for output, records in predictions.items():
        if not len(records.labels):
            raise ValueError("Cannot evaluate an empty split")
        y, p, users, warm = (
            v[records.valid] for v in (records.labels, records.logits, records.users, records.warm)
        )
        result[f"{output}/rows_total"] = len(records.labels)
        for cohort, selected in (
            ("", np.ones(len(y), dtype=bool)),
            ("warm_item/", warm),
            ("cold_item/", ~warm),
        ):
            prefix = f"{output}/{cohort}"
            if selected.any():
                metrics = {
                    **binary_metrics(y[selected], p[selected]),
                    **user_auc_metrics(users[selected], y[selected], p[selected]),
                }
            else:
                metrics = {
                    "rows": 0,
                    "roc_auc": None,
                    "macro_user_auc": None,
                    "logloss": None,
                    "brier": None,
                    "users_all": 0,
                    "users_with_both_classes": 0,
                }
            result.update({prefix + n: v for n, v in metrics.items()})
    return result


class PredictionEvaluator:
    """Own output-to-label interpretation; callers supply models and instances.

    No dataset paths, model selection, optimizer state or metric storage lives here.
    collect() exposes prediction records for reuse outside the training process.
    """

    def __init__(self, targets: Mapping[str, str], masks: Mapping[str, str] | None = None) -> None:
        if not targets or any(not k or not v for k, v in targets.items()):
            raise ValueError("Configure named prediction targets")
        self.targets = dict(targets)
        self.masks = dict(masks or {})
        if self.masks.keys() - self.targets.keys():
            raise ValueError("Masks must refer to configured outputs")

    @torch.no_grad()
    def collect(self, model: nn.Module, batches: Iterable[Batch]) -> dict[str, Predictions]:
        chunks: dict[str, list[tuple[np.ndarray, ...]]] = {n: [] for n in self.targets}
        device = next(model.parameters(), torch.empty(0)).device
        was_training = model.training
        model.eval()
        try:
            for batch in batches:
                outputs = model({n: v.to(device) for n, v in batch["features"].items()})
                for output, target in self.targets.items():
                    labels, logits = batch["targets"][target].cpu(), outputs[output].detach().cpu()
                    mask_name = self.masks.get(output)
                    valid = (
                        torch.ones_like(labels, dtype=torch.bool)
                        if mask_name is None
                        else batch["target_masks"][mask_name].cpu()
                    )
                    metadata = batch.get("metadata", {})
                    if "user_id" not in metadata:
                        raise ValueError("Raw user IDs required for macro-user metrics")
                    warm = metadata.get("warm_item", torch.ones_like(valid)).cpu()
                    record = Predictions(
                        *(
                            v.numpy()
                            for v in (labels, logits, metadata["user_id"].cpu(), warm, valid)
                        )
                    )
                    chunks[output].append(
                        (record.labels, record.logits, record.users, record.warm, record.valid)
                    )
        finally:
            model.train(was_training)
        if any(not rows for rows in chunks.values()):
            raise ValueError("Cannot evaluate an empty split")
        return {
            name: Predictions(*(np.concatenate([row[i] for row in rows]) for i in range(5)))
            for name, rows in chunks.items()
        }

    def evaluate(self, model: nn.Module, batches: Iterable[Batch]) -> dict[str, Any]:
        return evaluate_predictions(self.collect(model, batches))
