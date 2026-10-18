"""Shared-pool binary retrieval metrics; no per-method denominator shrinkage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.data.recall import RecallData
from feed_ranker.recommendation.ordering import (
    ordered_positions as ordered_positions,
)
from feed_ranker.recommendation.ordering import (
    tie_keys as tie_keys,
)


def evaluate(
    data: RecallData,
    scores: np.ndarray,
    rank_scores: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Every row uses its full eligible pool; output top-100 padded internal IDs."""
    if scores.shape != data.pools.items.shape:
        raise ValueError("Pool score length mismatch")
    if rank_scores is not None and (
        rank_scores.shape != scores.shape or not np.isfinite(rank_scores).all()
    ):
        raise ValueError("Ranker must score every eligible pair finitely")
    discounts = 1 / np.log2(np.arange(2, 12))
    selected = np.full((len(data.users), 100), -1, dtype=np.int64)
    rows = []
    for u, raw_user in enumerate(data.users):
        lo, hi = data.pools.offsets[u : u + 2]
        items = data.pools.items[lo:hi]
        raw_items = data.items[items]
        rel = data.pools.relevant[lo:hi]
        count = int(rel.sum())
        ideal = discounts[: min(10, count)].sum()
        order = ordered_positions(scores[lo:hi], int(raw_user), raw_items)[:100]
        selected[u, : len(order)] = items[order]
        row: dict[str, Any] = {
            "user_id": int(raw_user),
            "pool_size": len(items),
            "positives": count,
            "train_seen": len(data.training.seen[u]),
            "train_positive": len(data.training.positive[u]),
            "train_negative": len(data.training.negative[u]),
            "mf_supported": bool(data.training.supported[u]),
        }
        for m in (20, 50, 100):
            candidates = order[:m]
            hits = int(rel[candidates].sum())
            row[f"returned@{m}"] = len(candidates)
            row[f"pool_short@{m}"] = len(items) < m
            row[f"method_short@{m}"] = len(candidates) < min(m, len(items))
            row[f"recall@{m}"] = hits / count if count else np.nan
            row[f"hit@{m}"] = float(hits > 0) if count else np.nan
            row[f"oracle_ndcg@10_m{m}"] = (
                discounts[: min(10, hits)].sum() / ideal if count else np.nan
            )
            if rank_scores is None:
                final = candidates[:10]
            else:
                ranking = ordered_positions(
                    rank_scores[lo:hi][candidates], int(raw_user), raw_items[candidates]
                )
                final = candidates[ranking[:10]]
            final_hits = int(rel[final].sum())
            row[f"ndcg@10_m{m}"] = (
                float((rel[final] * discounts[: len(final)]).sum() / ideal) if count else np.nan
            )
            row[f"recall@10_m{m}"] = final_hits / count if count else np.nan
            row[f"hit@10_m{m}"] = float(final_hits > 0) if count else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    quality = [c for c in frame if c.startswith(("recall@", "hit@", "ndcg@", "oracle_"))]
    summary: dict[str, Any] = {
        "users_all": len(frame),
        "users_quality": int(frame.positives.gt(0).sum()),
        "users_zero_positive": int(frame.positives.eq(0).sum()),
        "metrics": {c: float(frame[c].mean()) if frame[c].notna().any() else None for c in quality},
        "coverage": {
            c: float(frame[c].mean())
            for c in frame
            if c.startswith(("returned@", "pool_short@", "method_short@"))
        },
        "ordering": "ranker" if rank_scores is not None else "retrieval",
    }
    return summary, frame, selected
