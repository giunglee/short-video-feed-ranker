"""Fixed-budget, label-free merging through the shared retrieval interface."""

from __future__ import annotations

import numpy as np

from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Request,
    RetrievalResult,
    Retriever,
    ScoredItems,
)
from feed_ranker.recommendation.merging import merge_channels, reciprocal_rank_fusion


class MergedRetriever:
    """Each source returns at most source_depth; the final limit is shared across sources."""

    def __init__(self, sources: list[Retriever], policy: str, source_depth: int = 50) -> None:
        if policy not in ("rr_all", "rrf_all") or source_depth < 1 or len(sources) < 2:
            raise ValueError("Require a fixed RR/RRF policy and at least two sources")
        if len({s.name for s in sources}) != len(sources):
            raise ValueError("Source names must be unique")
        self.sources, self.name, self.source_depth = sources, policy, source_depth
        self.artifact_id = f"{policy}:depth{source_depth}:rrf60:seed2026:" + ":".join(
            s.artifact_id for s in sources
        )

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        ranked = {
            source.name: source.retrieve(request, scope, self.source_depth).candidates.items
            for source in self.sources
        }
        if any(not np.isin(items, scope.items).all() for items in ranked.values()):
            raise ValueError("Source returned ineligible candidates")
        support = len(np.unique(np.concatenate(list(ranked.values()))))
        if self.name == "rr_all":
            items = merge_channels(ranked, limit, request.user_id).items
        else:
            items = reciprocal_rank_fusion(ranked, limit, request.user_id, rank_constant=60.0).items
        # Preserve the policy's exact order without comparing source score scales.
        return RetrievalResult(ScoredItems(items, -np.arange(len(items), dtype=float)), support)
