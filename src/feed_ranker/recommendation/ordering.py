"""Deterministic raw-ID ordering shared by inference and offline metrics."""

import numpy as np


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
