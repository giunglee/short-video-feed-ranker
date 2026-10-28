"""Exact recall baselines and BPR MF.

BPR: Rendle et al., UAI 2009, https://arxiv.org/abs/1205.2618.
Cosine similarity operates on binary positive-user incidence vectors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BPRMF(nn.Module):
    """User/item dot products, BPR margins. Citation: https://arxiv.org/abs/1205.2618."""

    def __init__(self, users: int, items: int, dimension: int = 32) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        # reasonable default, not from any specific production config
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        self.register_buffer("item_updated", torch.zeros(items, dtype=torch.bool))

    def forward(
        self, users: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        return (self.user(users) * (self.item(positive) - self.item(negative))).sum(-1)

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score aligned internal IDs; eligibility belongs to the retrieval adapter."""
        return (self.user(users) * self.item(items)).sum(-1)


class SparseCosineCooccurrence:
    """Exact cosine with storage linear in positive graph edges.

    Sarwar et al., WWW 2001, https://doi.org/10.1145/371920.372071.
    Evaluate h_u D^-1 H.T H D^-1 - h_u without constructing H.T H.
    H is binary [users,items]; D contains square roots of positive-user counts.
    """

    def __init__(self, positive: list[np.ndarray], items: int) -> None:
        if items < 1 or not positive:
            raise ValueError("Require users and a positive item vocabulary size")
        self.positive = [np.asarray(p, dtype=np.int64).copy() for p in positive]
        for p in self.positive:
            if p.ndim != 1 or (p < 0).any() or (p >= items).any() or len(np.unique(p)) != len(p):
                raise ValueError("Histories must contain unique in-vocabulary item IDs")
        edge_items = np.concatenate(self.positive)
        edge_users = np.repeat(np.arange(len(positive)), [len(p) for p in positive])
        self.counts = np.bincount(edge_items, minlength=items)
        self.offsets = np.r_[0, self.counts.cumsum()]
        self.edge_users = edge_users[np.argsort(edge_items, kind="stable")]
        self.norms = np.sqrt(np.maximum(self.counts, 1))

    def _edges(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lengths = self.counts[items]
        groups = np.repeat(np.arange(len(items)), lengths)
        starts = self.offsets[items]
        positions = np.repeat(starts - np.r_[0, lengths.cumsum()[:-1]], lengths)
        positions += np.arange(int(lengths.sum()))
        return groups, self.edge_users[positions]

    def score(self, user: int, items: np.ndarray) -> np.ndarray:
        if not 0 <= user < len(self.positive):
            raise ValueError("User outside fitted vocabulary")
        if items.ndim != 1 or (items < 0).any() or (items >= len(self.counts)).any():
            raise ValueError("Items outside fitted vocabulary")
        history = self.positive[user]
        if not len(items) or not len(history):
            return np.zeros(len(items), dtype=np.float64)
        groups, neighbors = self._edges(history)
        weights = np.bincount(
            neighbors, weights=1 / self.norms[history[groups]], minlength=len(self.positive)
        )
        groups, neighbors = self._edges(items)
        scores = np.bincount(groups, weights=weights[neighbors], minlength=len(items))
        scores = scores / self.norms[items] - np.isin(items, history)
        # Removing the diagonal can leave roundoff at zero, never positive evidence.
        scores[np.abs(scores) < 1e-12] = 0
        # Canonicalize arithmetic-level ties before the shared raw-ID tie breaker.
        return np.round(scores, decimals=12)

    @property
    def storage_bytes(self) -> int:
        return sum(p.nbytes for p in self.positive) + sum(
            a.nbytes for a in [self.counts, self.offsets, self.edge_users, self.norms]
        )
