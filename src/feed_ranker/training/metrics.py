"""Sampled concordance excludes label ties and gives prediction ties half credit.

Global XAUC weights comparable ordered row pairs uniformly. User concordance
weights eligible users equally. Neither metric uses the model's random generator.
"""

from __future__ import annotations

import numpy as np
import torch


def xauc(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_pairs: int = 100_000,
    seed: int = 0,
) -> float | None:
    """Return None when no unequal-label pair exists; otherwise sample with replacement."""
    predictions, targets = _metric_arrays(preds, labels, num_pairs)
    pairs = _sample_unequal_pairs(targets, num_pairs, seed)
    if pairs is None:
        return None
    left, right = pairs
    tied = predictions[left] == predictions[right]
    concordant = (predictions[left] > predictions[right]) == (targets[left] > targets[right])
    return float(np.where(tied, 0.5, concordant.astype(float)).mean())


def user_xauc(
    preds: torch.Tensor,
    labels: torch.Tensor,
    users: torch.Tensor,
    num_pairs: int = 256,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Macro-average per-user XAUC; count users without comparable labels as excluded."""
    predictions, targets = _metric_arrays(preds, labels, num_pairs)
    user_ids = users.detach().cpu().numpy().reshape(-1)
    if len(user_ids) != len(targets) or not np.isfinite(user_ids).all():
        raise ValueError("User IDs must be finite and aligned with metric inputs")
    if (user_ids != np.floor(user_ids)).any():
        raise ValueError("User IDs must be integers")

    order = np.argsort(user_ids, kind="stable")
    unique_users, starts, counts = np.unique(user_ids[order], return_index=True, return_counts=True)
    predictions_tensor = torch.from_numpy(predictions)
    targets_tensor = torch.from_numpy(targets)
    scores: list[float] = []
    for user, start, count in zip(unique_users, starts, counts):
        indices = torch.from_numpy(order[start : start + count])
        user_seed = (seed + int(user)) % (2**32)
        score = xauc(predictions_tensor[indices], targets_tensor[indices], num_pairs, user_seed)
        if score is not None:
            scores.append(score)
    return {
        "user_xauc": float(np.mean(scores)) if scores else None,
        "eligible_users": len(scores),
        "excluded_users": len(unique_users) - len(scores),
    }


def _metric_arrays(
    preds: torch.Tensor, labels: torch.Tensor, num_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    predictions = preds.detach().cpu().numpy().reshape(-1)
    targets = labels.detach().cpu().numpy().reshape(-1)
    if len(predictions) != len(targets):
        raise ValueError("Predictions and labels must have the same number of rows")
    if type(num_pairs) is not int or num_pairs <= 0:
        raise ValueError("num_pairs must be a positive integer")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Metric inputs must be finite")
    return predictions, targets


def _sample_unequal_pairs(
    labels: np.ndarray, num_pairs: int, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(labels) < 2:
        return None
    order = np.argsort(labels, kind="stable")
    _, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
    if len(counts) < 2:
        return None

    rng = np.random.default_rng(seed)
    # A label group of size k contributes k * (N-k) eligible ordered pairs.
    weights = counts.astype(float) * (len(labels) - counts)
    groups = rng.choice(len(counts), num_pairs, p=weights / weights.sum())
    left = starts[groups] + (rng.random(num_pairs) * counts[groups]).astype(int)
    right = (rng.random(num_pairs) * (len(labels) - counts[groups])).astype(int)
    # Skip the selected label block instead of repeatedly rejecting tied labels.
    right += (right >= starts[groups]) * counts[groups]
    return order[left], order[right]
