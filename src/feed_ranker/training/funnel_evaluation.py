"""Linear capped-gain NDCG on fixed, observed held-out user-item pools.

The full eligible pool determines IDCG, even for short candidate lists. Labels
never enter prediction inputs or candidate ordering. This is an offline exposure-
conditional metric, not a temporal or causal estimate of recommendation utility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from feed_ranker.training.recall_evaluation import tie_keys


class TargetMAE(nn.Module):
    """Transform targets only; raw batch labels and unconstrained predictions survive."""

    def __init__(self, cap: float | None = None) -> None:
        super().__init__()
        if cap is not None and (not np.isfinite(cap) or cap <= 0):
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


@dataclass
class GainPool:
    """One immutable first-event gain per eligible pair, grouped by raw user."""

    users: np.ndarray
    items: np.ndarray
    offsets: np.ndarray
    ratios: np.ndarray
    duration_ms: np.ndarray
    k: int = 10
    cap: float = 2.0

    def __post_init__(self) -> None:
        n = len(self.items)
        if (
            self.k <= 0
            or not np.isfinite(self.cap)
            or self.cap <= 0
            or self.offsets.shape != (len(self.users) + 1,)
            or self.offsets[0] != 0
            or self.offsets[-1] != n
            or (np.diff(self.offsets) < 0).any()
            or self.ratios.shape != (n,)
            or self.duration_ms.shape != (n,)
            or not np.isfinite(self.ratios).all()
            or (self.ratios < 0).any()
        ):
            raise ValueError("Invalid gain pool, offsets, or nonnegative finite ratios")
        self.gains = np.minimum(self.ratios, self.cap)
        self.discounts = 1 / np.log2(np.arange(2, self.k + 2))
        self.ties = np.empty(n, dtype=np.uint64)
        self.ideals = np.empty(len(self.users))
        for u, user in enumerate(self.users):
            lo, hi = self.offsets[u : u + 2]
            if len(np.unique(self.items[lo:hi])) != hi - lo:
                raise ValueError("Duplicate item within user pool")
            self.ties[lo:hi] = tie_keys(int(user), self.items[lo:hi])
            self.ideals[u] = self.dcg(np.sort(self.gains[lo:hi])[::-1])

    def dcg(self, gains: np.ndarray) -> float:
        return float(np.dot(gains[: self.k], self.discounts[: min(self.k, len(gains))]))

    def order(self, scores: np.ndarray, positions: np.ndarray) -> np.ndarray:
        valid = positions[np.isfinite(scores[positions])]
        return valid[np.lexsort((self.items[valid], self.ties[valid], -scores[valid]))]

    def candidates(self, scores: np.ndarray, budget: int) -> list[np.ndarray]:
        if scores.shape != self.items.shape or np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError("Candidate scores must match pool; only -inf marks unsupported")
        return [
            self.order(scores, np.arange(lo, hi))[:budget]
            for lo, hi in zip(self.offsets[:-1], self.offsets[1:], strict=True)
        ]

    def evaluate(
        self, predictions: np.ndarray, candidates: list[np.ndarray] | None = None
    ) -> tuple[dict[str, float | int | None], pd.DataFrame]:
        if predictions.shape != self.items.shape or not np.isfinite(predictions).all():
            raise ValueError("Require finite predictions for every eligible pair")
        if candidates is not None and len(candidates) != len(self.users):
            raise ValueError("Candidate user count differs from pool")
        rows = []
        for u, user in enumerate(self.users):
            lo, hi = self.offsets[u : u + 2]
            selected = np.arange(lo, hi) if candidates is None else candidates[u]
            if (
                (selected < lo).any()
                or (selected >= hi).any()
                or len(np.unique(selected)) != len(selected)
            ):
                raise ValueError("Candidates must be unique positions in this user's pool")
            final = self.order(predictions, selected)[: self.k]
            ideal = self.ideals[u]
            row = {
                "user_id": int(user),
                "pool_size": int(hi - lo),
                "returned": len(selected),
                "final_slots": len(final),
                "idcg": ideal,
                "cap2_ndcg10": self.dcg(self.gains[final]) / ideal if ideal else np.nan,
                "candidate_oracle_ndcg10": self.dcg(np.sort(self.gains[selected])[::-1]) / ideal
                if ideal
                else np.nan,
                "utility_per_slot10": float(self.gains[final].sum() / self.k),
            }
            rows.append(row)
        frame = pd.DataFrame(rows)
        eligible = frame.idcg.gt(0)
        metrics = ["cap2_ndcg10", "candidate_oracle_ndcg10"]
        result = {
            name: float(frame.loc[eligible, name].mean()) if eligible.any() else None
            for name in metrics
        }
        result.update(
            {
                "users_all": len(frame),
                "users_quality": int(eligible.sum()),
                "users_zero_idcg": int((~eligible).sum()),
                "utility_per_slot10_all_users": float(frame.utility_per_slot10.mean()),
                "mean_returned": float(frame.returned.mean()),
                "mean_final_slots": float(frame.final_slots.mean()),
                "fraction_prediction_gt2": float((predictions > 2).mean()),
            }
        )
        return result, frame

    def duration_report(
        self, predictions: np.ndarray, candidates: list[np.ndarray]
    ) -> pd.DataFrame:
        """Global top-K exposure and captured utility, grouped by observed duration.

        Group ideals come from the full pool; group ceilings need not be attainable
        simultaneously under one K budget. Duration is diagnostic only.
        """
        seconds = self.duration_ms / 1000
        masks = {
            "le10s": seconds <= 10,
            "10_30s": (seconds > 10) & (seconds <= 30),
            "30_60s": (seconds > 30) & (seconds <= 60),
            "gt60s": seconds > 60,
        }
        rows = []
        for u, user in enumerate(self.users):
            lo, hi = self.offsets[u : u + 2]
            final = self.order(predictions, candidates[u])[: self.k]
            for name, mask in masks.items():
                ideal = self.dcg(np.sort(self.gains[lo:hi] * mask[lo:hi])[::-1])
                actual = self.dcg(self.gains[final] * mask[final])
                rows.append(
                    {
                        "user_id": int(user),
                        "duration": name,
                        "selected_slots": int(mask[final].sum()),
                        "utility_per_slot10": float(
                            (self.gains[final] * mask[final]).sum() / self.k
                        ),
                        "duration_ndcg10": actual / ideal if ideal else np.nan,
                    }
                )
        return pd.DataFrame(rows)
