"""Shared-pool binary retrieval metrics; no per-method denominator shrinkage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.data.recall import RecallData


def tie_keys(user: int, items: np.ndarray, seed: int = 2026) -> np.ndarray:
    """Stable SplitMix64 keys, independent of labels, Python hash, or M."""
    with np.errstate(over="ignore"):
        x = (
            items.astype(np.uint64)
            ^ np.uint64(user) * np.uint64(0x9E3779B97F4A7C15)
            ^ np.uint64(seed)
        )
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> 31)


def ordered_positions(scores: np.ndarray, raw_user: int, raw_items: np.ndarray) -> np.ndarray:
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Nonfinite prediction other than explicitly unsupported -inf")
    valid = np.flatnonzero(np.isfinite(scores))
    return valid[
        np.lexsort((raw_items[valid], tie_keys(raw_user, raw_items[valid]), -scores[valid]))
    ]


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
            "train_seen": len(data.seen[u]),
            "train_positive": len(data.positive[u]),
            "train_negative": len(data.negative[u]),
            "mf_supported": bool(data.supported[u]),
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
