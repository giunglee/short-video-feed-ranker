"""Compose existing retrievers with a fixed, label-free channel allocation.

Source depth is explicit: None requests full support; an integer caps each
source before fusion. Support counts the union available at that chosen depth.
"""

import hashlib
import json
from collections.abc import Sequence

import numpy as np

from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Request,
    RetrievalResult,
    Retriever,
    ScoredItems,
)
from feed_ranker.recommendation.merging import (
    anchored_candidates,
    merge_channels,
    reciprocal_rank_fusion,
)


class MergedRetriever:
    """A merged channel implements the same Retriever contract as a singleton."""

    def __init__(
        self,
        channels: Sequence[Retriever],
        policy: str = "round_robin",
        *,
        source_depth: int | None,
        seed: int = 2026,
        rank_constant: float = 60.0,
        anchor: str = "mf",
        anchor_fraction: float = 0.8,
    ) -> None:
        if not channels or len({c.name for c in channels}) != len(channels):
            raise ValueError("Require nonempty, distinct retrieval channels")
        if policy not in ("round_robin", "rrf", "anchored"):
            raise ValueError("Unknown channel merging policy")
        if not np.isfinite(rank_constant) or rank_constant < 0:
            raise ValueError("Require a nonnegative finite RRF constant")
        if source_depth is not None and (type(source_depth) is not int or source_depth < 1):
            raise ValueError("Source depth must be a positive integer or None for full support")
        self.source_depth = source_depth
        self.channels = tuple(sorted(channels, key=lambda c: c.name))
        self.policy, self.seed, self.rank_constant = policy, seed, rank_constant
        self.anchor, self.anchor_fraction = anchor, anchor_fraction
        if policy == "anchored" and (
            anchor not in {c.name for c in channels}
            or len(channels) < 2
            or not np.isfinite(anchor_fraction)
            or not 0 < anchor_fraction < 1
        ):
            raise ValueError("Require an anchor, supplemental channels and fraction in (0, 1)")
        self.name = policy + ":" + "+".join(c.name for c in self.channels)
        self.config = {
            "policy": policy,
            "tie_seed": seed,
            "rank_constant": rank_constant if policy == "rrf" else None,
            "source_depth": source_depth,
            "channels": {c.name: c.artifact_id for c in self.channels},
        }
        if policy == "anchored":
            self.config.update(anchor=anchor, anchor_fraction=anchor_fraction)
        self.artifact_id = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        if type(limit) is not int or limit < 1:
            raise ValueError("Require a positive integer retrieval limit")
        ranked = {}
        for channel in self.channels:
            # Positive limits also allow adapters to handle an empty scope.
            depth = self.source_depth if self.source_depth is not None else max(1, len(scope.items))
            result = channel.retrieve(request, scope, depth)
            items = result.candidates.items
            if (
                type(result.supported_count) is not int
                or result.supported_count < 0
                or len(items) != min(depth, result.supported_count)
                or (self.source_depth is None and result.supported_count != len(items))
                or not np.isin(items, scope.items).all()
            ):
                raise ValueError(
                    "Channel must return complete supported candidates at the requested depth"
                )
            ranked[channel.name] = items
        supported = len(np.unique(np.concatenate(list(ranked.values()))))
        if self.policy == "round_robin":
            merged = merge_channels(ranked, limit, request.user_id, self.seed)
            items = merged.items
            # Ordinal allocation scores, not comparable engagement predictions.
            scores = np.arange(len(items), 0, -1, dtype=float)
        elif self.policy == "anchored":
            items = anchored_candidates(
                ranked, limit, request.user_id, self.anchor, self.anchor_fraction, self.seed
            )
            scores = np.arange(len(items), 0, -1, dtype=float)
        else:
            merged = reciprocal_rank_fusion(
                ranked, limit, request.user_id, self.seed, self.rank_constant
            )
            items, scores = merged.items, merged.scores
        return RetrievalResult(ScoredItems(items, scores), supported)
