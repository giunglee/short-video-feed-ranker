"""Budgeted merging of ranked retrieval channels, without mixing score scales.

Round-robin gives each active channel one unique admission per cycle; duplicates
do not consume its turn and exhausted channels yield their remaining capacity.
Reciprocal rank fusion sums rank contributions to favor channel agreement.
Both are deterministic, label-free policies, not learned models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass
class MergedCandidates:
    """Unique IDs plus the admitting channel/rank, not causal attribution."""

    items: np.ndarray
    channels: tuple[str, ...]
    owner: np.ndarray
    source_rank: np.ndarray
    duplicate_skips: np.ndarray


def _validated_lists(
    ranked: Mapping[str, np.ndarray],
    budget: int,
) -> tuple[tuple[str, ...], list[np.ndarray]]:
    if not ranked or not isinstance(budget, int) or budget < 1:
        raise ValueError("Require at least one channel and a positive integer budget")
    if not all(isinstance(name, str) and name for name in ranked):
        raise ValueError("Channel names must be nonempty strings")
    names = tuple(sorted(ranked))
    lists = []
    for name in names:
        values = np.asarray(ranked[name])
        if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError("Channel candidates must be nonnegative integer vectors")
        if len(np.unique(values)) != len(values):
            raise ValueError("A channel must not contain duplicate candidates")
        lists.append(values)
    return names, lists


def merge_channels(
    ranked: Mapping[str, np.ndarray], budget: int, user_id: int, seed: int = 2026
) -> MergedCandidates:
    """Merge unique integer-ID lists in rank order, with at most ``budget`` IDs.

    Callers enforce eligibility before supplying lists. No labels or cross-channel
    scores enter this function. A channel can never contribute an item it did not
    retrieve. Identical inputs yield identical prefix lists for all budgets.
    """
    names, lists = _validated_lists(ranked, budget)
    order = sorted(
        range(len(names)),
        key=lambda i: hashlib.sha256(f"{seed}:{user_id}:{names[i]}".encode()).digest(),
    )
    cursors = np.zeros(len(names), dtype=np.int64)
    skipped = np.zeros(len(names), dtype=np.int64)
    chosen: list[int] = []
    owners: list[int] = []
    ranks: list[int] = []
    seen: set[int] = set()
    while len(chosen) < budget:
        progressed = False
        for channel in order:
            values = lists[channel]
            while cursors[channel] < len(values) and int(values[cursors[channel]]) in seen:
                cursors[channel] += 1
                skipped[channel] += 1
            if cursors[channel] == len(values):
                continue
            item = int(values[cursors[channel]])
            chosen.append(item)
            seen.add(item)
            owners.append(channel)
            ranks.append(int(cursors[channel]) + 1)
            cursors[channel] += 1
            progressed = True
            if len(chosen) == budget:
                break
        if not progressed:
            break
    return MergedCandidates(
        np.asarray(chosen, dtype=np.int64),
        names,
        np.asarray(owners, dtype=np.int64),
        np.asarray(ranks, dtype=np.int64),
        skipped,
    )


@dataclass
class FusedCandidates:
    """RRF results; source_ranks has shape [items, channels], with 0 for absent."""

    items: np.ndarray
    channels: tuple[str, ...]
    scores: np.ndarray
    source_ranks: np.ndarray


def anchored_candidates(
    ranked: Mapping[str, np.ndarray],
    budget: int,
    user_id: int,
    anchor: str = "mf",
    fraction: float = 0.8,
    seed: int = 2026,
) -> np.ndarray:
    """Protect floor(fraction * budget) anchor items, then admit unique supplements.

    Supplements use round-robin in their original order, skipping protected IDs.
    Unused anchor slots pass to supplements; exhausted supplements yield to the
    anchor tail. The requested budget sets the quota, not the eligible pool size.
    Different budgets need not produce nested lists. No engagement labels enter.
    """
    _validated_lists(ranked, budget)
    if anchor not in ranked or len(ranked) < 2:
        raise ValueError("Require an anchor and at least one supplemental channel")
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Anchor fraction must be finite and strictly between zero and one")
    protected = ranked[anchor][: int(np.floor(fraction * budget))]
    supplemental = {
        name: items[~np.isin(items, protected)] for name, items in ranked.items() if name != anchor
    }
    added = merge_channels(supplemental, budget - len(protected), user_id, seed).items
    selected = np.concatenate([protected, added])
    tail = ranked[anchor][~np.isin(ranked[anchor], selected)]
    return np.concatenate([selected, tail[: budget - len(selected)]])


def reciprocal_rank_fusion(
    ranked: Mapping[str, np.ndarray],
    budget: int,
    user_id: int,
    seed: int = 2026,
    rank_constant: float = 60.0,  # reasonable default, not from any specific production config
) -> FusedCandidates:
    """Sum equal-weight 1/(constant + one-based rank); absent items contribute zero.

    Cormack, Clarke and Buettcher, SIGIR 2009, doi:10.1145/1571941.1572114.
    https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf

    Supply raw video IDs for stable seeded user/video tie-breaking. All supplied
    ranks contribute; the caller declares source depth. Budget limits output only.
    Fixed input lists produce nested prefixes. No label or raw score is consumed.
    """
    names, lists = _validated_lists(ranked, budget)
    if not np.isfinite(rank_constant) or rank_constant < 0:
        raise ValueError("RRF rank constant must be finite and nonnegative")
    items = np.unique(np.concatenate(lists))
    ranks = np.zeros((len(items), len(names)), dtype=np.int64)
    for channel, values in enumerate(lists):
        ranks[np.searchsorted(items, values), channel] = np.arange(1, len(values) + 1)
    contributions = np.zeros(ranks.shape, dtype=float)
    supported = ranks > 0
    contributions[supported] = 1.0 / (rank_constant + ranks[supported])
    # Canonical summation preserves exact ties when channel ranks are permuted.
    scores = np.sort(contributions, axis=1).sum(axis=1)
    order = sorted(
        range(len(items)),
        key=lambda i: (
            -scores[i],
            hashlib.sha256(f"{seed}:{user_id}:{int(items[i])}".encode()).digest(),
            int(items[i]),
        ),
    )[:budget]
    return FusedCandidates(items[order], names, scores[order], ranks[order])
