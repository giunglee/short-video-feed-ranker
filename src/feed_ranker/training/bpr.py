"""BPR optimization for the temporal binary bridge, reusing the observed sampler.

BPR: Rendle et al., https://arxiv.org/abs/1205.2618.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer

from feed_ranker.models.recall import BPRMF


def fit_mf(
    model: BPRMF,
    sampler: ObservedSampler,
    optimizer: Optimizer,
    *,
    epochs: int,
    seed: int,
    batch_size: int = 4096,
) -> list[dict[str, Any]]:
    """Update the supplied model; return the fixed-final-epoch loss history.

    The caller owns initialization, optimizer construction and persistence.
    Sampling uses a local NumPy RNG; this CPU loop does not reset the torch RNG.
    """
    if any(type(n) is not int or n < 1 for n in (epochs, batch_size)):
        raise ValueError("Require positive epochs and batch size")
    model.train()
    rng = np.random.default_rng(seed)
    curve = []
    for epoch in range(epochs):
        users, pos, neg = sampler.sample(rng)
        total = 0.0
        for start in range(0, len(users), batch_size):
            u, p, n = (
                users[start : start + batch_size],
                pos[start : start + batch_size],
                neg[start : start + batch_size],
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-model(u, p, n)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite BPR loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            model.item_updated[p] = True
            model.item_updated[n] = True
            total += float(loss.detach()) * len(u)
        curve.append({"epoch": epoch, "train_loss": total / len(users), "pairs": len(users)})
        print(f"mf epoch={epoch} loss={curve[-1]['train_loss']:.6f}", flush=True)
    return curve


class ObservedSampler:
    """One uniform observed-negative comparison per positive; unsupported users omitted.

    Positive-ever and never-positive histories define the observed training comparisons. Sampling retains
    the original per-user ordering and RNG draws used by the KuaiRand BPR study.
    """

    def __init__(self, positive: list[np.ndarray], negative: list[np.ndarray]) -> None:
        self.supported = np.array(
            [bool(len(p) and len(n)) for p, n in zip(positive, negative, strict=True)]
        )
        if not self.supported.any():
            raise ValueError("No users have both positive and observed-negative training pairs")
        self.users = np.concatenate(
            [
                np.full(len(p), u, dtype=np.int64)
                for u, p in enumerate(positive)
                if self.supported[u]
            ]
        )
        self.positive = np.concatenate([p for u, p in enumerate(positive) if self.supported[u]])
        self.offsets = np.r_[0, np.cumsum([len(n) for n in negative])]
        self.negative = np.concatenate(negative)
        self.lengths = np.diff(self.offsets)

    def sample(self, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = self.offsets[self.users] + (
            rng.random(len(self.users)) * self.lengths[self.users]
        ).astype(np.int64)
        order = rng.permutation(len(self.users))
        return (
            torch.from_numpy(self.users[order]),
            torch.from_numpy(self.positive[order]),
            torch.from_numpy(self.negative[positions[order]]),
        )


def histories(
    train: pd.DataFrame,
    users: np.ndarray,
    items: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Positive-ever vs observed never-positive items; no contradictory BPR pair."""
    pairs = train.groupby(["user_id", "video_id"], as_index=False)[label].max()
    positive, negative = [], []
    grouped = {int(u): f for u, f in pairs.groupby("user_id")}
    for user in users:
        rows = grouped[int(user)]
        ids = pd.Index(items).get_indexer(rows.video_id)
        positive.append(ids[rows[label].to_numpy() == 1])
        negative.append(ids[rows[label].to_numpy() == 0])
    return positive, negative
