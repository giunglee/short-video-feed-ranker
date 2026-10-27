"""KuaiRand adapter for the shared request pipeline; no outcomes or split readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import CandidateScope, Request, RetrievalResult, Retriever
from feed_ranker.recommendation.rankers import ModelRanker
from feed_ranker.recommendation.retrievers import select


class SparseCooccurrenceRetriever:
    name = "cooccurrence"

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        model: SparseCosineCooccurrence,
        artifact_id: str,
    ) -> None:
        if len(users) != len(model.positive) or len(items) != len(model.counts):
            raise ValueError("Cooccurrence vocabulary mismatch")
        self.users, self.items = pd.Index(users), pd.Index(items)
        self.model, self.artifact_id = model, artifact_id

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        user = self.users.get_indexer([request.user_id])[0]
        ids = self.items.get_indexer(scope.items)
        known = ids >= 0
        items, ids = scope.items[known], ids[known]
        if user < 0:
            return select(request, items[:0], np.empty(0), limit)
        scores = self.model.score(int(user), ids)
        supported = scores > 0
        return select(request, items[supported], scores[supported], limit)


def load_ranker(path: Path) -> ModelRanker:
    from feed_ranker.models.legacy_kuairand import load_legacy_ranker

    loaded = load_legacy_ranker(path)
    return ModelRanker(loaded.model, loaded.features, loaded.identity)


def load_retrievers(directory: Path) -> list[Retriever]:
    from feed_ranker.models.legacy_kuairand import load_legacy_recall
    from feed_ranker.recommendation.retrievers import MFRetriever, PopularityRetriever

    pop, cosine, mf = load_legacy_recall(directory)
    return [
        PopularityRetriever(pop.items, pop.model, pop.identity),
        SparseCooccurrenceRetriever(cosine.users, cosine.items, cosine.model, cosine.identity),
        MFRetriever(mf.model, mf.users, mf.items, mf.supported_users, mf.identity),
    ]
