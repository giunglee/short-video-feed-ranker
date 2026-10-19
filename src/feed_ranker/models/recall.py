"""Exact recall baselines and BPR MF.

BPR: Rendle et al., UAI 2009, https://arxiv.org/abs/1205.2618.
Cosine similarity operates on binary positive-user incidence vectors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CosineCooccurrence:
    """Exact cosine item similarities fitted from binary positive-user incidence.

    Intended for the bounded KuaiRec catalog: the similarity matrix is quadratic
    in catalog size. Both batch evaluation and request retrieval use this scorer.
    """

    def __init__(
        self, positive_incidence: torch.Tensor, *, similarity: torch.Tensor | None = None
    ) -> None:
        if (
            positive_incidence.ndim != 2
            or not ((positive_incidence == 0) | (positive_incidence == 1)).all()
        ):
            raise ValueError("Require binary positive-user incidence")
        self.incidence = positive_incidence.detach().cpu().float().clone()
        if similarity is None:
            norms = self.incidence.sum(0).sqrt().clamp_min(1)
            self.similarity = self.incidence.T @ self.incidence
            self.similarity.div_(norms[:, None]).div_(norms[None, :])
            self.similarity.fill_diagonal_(0)
        else:
            n = self.incidence.shape[1]
            if (
                similarity.shape != (n, n)
                or not torch.isfinite(similarity).all()
                or (similarity < 0).any()
                or similarity.diag().ne(0).any()
            ):
                raise ValueError("Invalid fitted cosine similarity")
            self.similarity = similarity.detach().cpu().float().clone()

    def score_users(self, start: int, stop: int) -> np.ndarray:
        return (self.incidence[start:stop] @ self.similarity).numpy()


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
