"""Complete aligned named prediction metrics; no fitting or checkpoint policy."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.data.prepared import Batch
from feed_ranker.evaluation.binary import binary_metrics, user_auc_metrics


@torch.no_grad()
def evaluate_predictions(
    model: nn.Module,
    batches: Iterable[Batch],
    targets: Mapping[str, str],
    masks: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Map output names to target names; report unweighted observational metrics.

    Loss weights do not silently redefine the evaluation population. Masked rows
    are excluded explicitly, with row/user inclusion counts returned per target.
    """
    if not targets:
        raise ValueError("Configure at least one prediction target")
    model.eval()
    collected: dict[str, list[tuple[np.ndarray, ...]]] = {n: [] for n in targets}
    totals = dict.fromkeys(targets, 0)
    device = next(model.parameters()).device
    for batch in batches:
        outputs = model({"features": {n: v.to(device) for n, v in batch["features"].items()}})
        for output, target in targets.items():
            y, p = batch["targets"][target], outputs[output].detach().cpu()
            if y.ndim != 1 or p.shape != y.shape:
                raise ValueError("Predictions and targets must align as [B]")
            mask_key = (masks or {}).get(output)
            mask = (
                torch.ones_like(y, dtype=torch.bool)
                if mask_key is None
                else batch["target_masks"][mask_key]
            )
            if mask.shape != y.shape or mask.dtype != torch.bool:
                raise ValueError("Invalid evaluation validity mask")
            metadata = batch.get("metadata", {})
            if "user_id" not in metadata or metadata["user_id"].shape != y.shape:
                raise ValueError("Aligned raw user IDs required for macro-user metrics")
            warm = metadata.get("warm_item", torch.ones_like(mask))
            if warm.shape != y.shape or warm.dtype != torch.bool:
                raise ValueError("Invalid warm-item cohort mask")
            totals[output] += len(y)
            collected[output].append(
                tuple(v[mask].numpy() for v in (y, p, metadata["user_id"], warm))
            )
    result = {}
    for output, chunks in collected.items():
        if not chunks:
            raise ValueError("Cannot evaluate an empty split")
        y, p, users, warm = (np.concatenate([c[i] for c in chunks]) for i in range(4))
        result[f"{output}/rows_total"] = totals[output]
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
