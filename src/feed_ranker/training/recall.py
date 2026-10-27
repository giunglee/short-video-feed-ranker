"""Training-only baseline fitting and observed-pair sampling."""

import numpy as np
import torch

from feed_ranker.data.recall import RecallTrainingState
from feed_ranker.models.recall import CosineCooccurrence
from feed_ranker.recommendation.retrievers import CooccurrenceRetriever, PopularityRetriever


def incidence(data: RecallTrainingState) -> torch.Tensor:
    """[users, items] binary positive incidence; full histories, no truncation."""
    matrix = torch.zeros((len(data.users), len(data.items)), dtype=torch.float32)
    for u, items in enumerate(data.positive):
        matrix[u, torch.from_numpy(items)] = 1
    return matrix


def fit_baseline(
    data: RecallTrainingState, method: str, artifact_id: str
) -> PopularityRetriever | CooccurrenceRetriever:
    """Fit positive-user counts or cosine similarities using training histories only."""
    if method == "popularity":
        counts = np.bincount(np.concatenate(data.positive), minlength=len(data.items)).astype(
            np.float32
        )
        return PopularityRetriever(data.items, counts, artifact_id)
    if method == "cooccurrence":
        return CooccurrenceRetriever(
            data.users, data.items, CosineCooccurrence(incidence(data)), artifact_id
        )
    raise ValueError(f"Unknown baseline {method}")


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
