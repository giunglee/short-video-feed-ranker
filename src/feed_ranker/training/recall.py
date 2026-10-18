"""Offline observed-pair sampling and flattened-pool scoring for recall experiments."""

import numpy as np
import torch

from feed_ranker.data.recall import RecallData, RecallTrainingState
from feed_ranker.models.recall import BPRMF, CosineCooccurrence


def incidence(data: RecallTrainingState) -> torch.Tensor:
    """[users, items] binary positive incidence; full histories, no truncation."""
    matrix = torch.zeros((len(data.users), len(data.items)), dtype=torch.float32)
    for u, items in enumerate(data.positive):
        matrix[u, torch.from_numpy(items)] = 1
    return matrix


def baseline_scores(data: RecallData, method: str) -> tuple[np.ndarray, dict[str, int]]:
    """Scores aligned with flattened eligible pools; -inf means unsupported."""
    h = incidence(data.training)
    counts = h.sum(0)
    if method == "popularity":
        scores = counts.numpy()[data.pools.items].copy()
    elif method == "cooccurrence":
        model = CosineCooccurrence(h)
        scores = np.empty(len(data.pools.items), dtype=np.float32)
        for start in range(0, len(data.users), 64):
            stop = min(start + 64, len(data.users))
            values = model.score_users(start, stop)
            lo, hi = data.pools.offsets[[start, stop]]
            scores[lo:hi] = values[data.pools.users[lo:hi] - start, data.pools.items[lo:hi]]
    else:
        raise ValueError(f"Unknown baseline {method}")
    # Zero evidence is not an implicit popularity/random fallback.
    scores[scores <= 0] = -np.inf
    return scores, {"items_with_positive_support": int(counts.gt(0).sum())}


@torch.no_grad()
def pool_scores(model: BPRMF, data: RecallData) -> np.ndarray:
    model.eval()
    scores = np.empty(len(data.pools.items), dtype=np.float32)
    for start in range(0, len(scores), 32768):
        stop = min(start + 32768, len(scores))
        users = torch.from_numpy(data.pools.users[start:stop])
        items = torch.from_numpy(data.pools.items[start:stop])
        scores[start:stop] = model.score_pairs(users, items).numpy()
    allowed = (
        data.training.supported[data.pools.users] & model.item_updated.numpy()[data.pools.items]
    )
    scores[~allowed] = -np.inf
    return scores


class ObservedSampler:
    """One observed comparison per positive, holding clear-negative user support fixed.

    The optional neutral policy changes only the comparison pool, not the positive
    labels or supported user cohort. Uniform sampling is over the pooled items,
    not a 50/50 mixture of outcome classes.
    """

    def __init__(self, data: RecallTrainingState, negative_policy: str = "negative_only") -> None:
        if negative_policy not in ("negative_only", "negative_and_neutral"):
            raise ValueError("Unknown observed comparison policy")
        if not data.supported.any():
            raise ValueError("No users have both positive and observed-negative training pairs")
        self.users = np.concatenate(
            [
                np.full(len(p), u, dtype=np.int64)
                for u, p in enumerate(data.positive)
                if data.supported[u]
            ]
        )
        self.positive = np.concatenate(
            [p for u, p in enumerate(data.positive) if data.supported[u]]
        )
        comparison = data.negative
        neutral = [np.empty(0, dtype=np.int64) for _ in data.users]
        if negative_policy == "negative_and_neutral":
            neutral = data.neutral
            comparison = [np.union1d(n, z) for n, z in zip(data.negative, neutral, strict=True)]
        self.offsets = np.r_[0, np.cumsum([len(n) for n in comparison])]
        self.negative = np.concatenate(comparison)
        self.is_neutral = np.concatenate(
            [np.isin(n, z) for n, z in zip(comparison, neutral, strict=True)]
        )
        self.last_neutral_draws = 0
        self.lengths = np.diff(self.offsets)
        if not len(self.users):
            raise ValueError("No users have both positive and observed-negative training pairs")

    def sample(self, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # floor(U*n) gives independent uniform offsets, including one-item pools.
        positions = self.offsets[self.users] + (
            rng.random(len(self.users)) * self.lengths[self.users]
        ).astype(np.int64)
        order = rng.permutation(len(self.users))
        self.last_neutral_draws = int(self.is_neutral[positions].sum())
        return (
            torch.from_numpy(self.users[order]),
            torch.from_numpy(self.positive[order]),
            torch.from_numpy(self.negative[positions[order]]),
        )
