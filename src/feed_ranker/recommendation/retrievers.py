"""Exact query retrievers, independent of held-out pools and labels.

MF uses Rendle et al., UAI 2009 (https://arxiv.org/abs/1205.2618).
Cooccurrence sums cosine-normalized positive-user incidence similarities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import load_artifact
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Request,
    RetrievalResult,
    Retriever,
    ScoredItems,
    item_ids,
)
from feed_ranker.recommendation.ordering import ordered_positions


def select(request: Request, items: np.ndarray, scores: np.ndarray, limit: int) -> RetrievalResult:
    order = ordered_positions(scores, request.user_id, items)
    chosen = order[:limit]
    return RetrievalResult(ScoredItems(items[chosen], scores[chosen]), len(order))


class PopularityRetriever:
    name = "popularity"

    def __init__(self, items: np.ndarray, positive_counts: np.ndarray, artifact_id: str) -> None:
        self.items = pd.Index(item_ids(items))
        self.counts = np.asarray(positive_counts).copy()
        if (
            self.counts.shape != (len(items),)
            or not np.isfinite(self.counts).all()
            or (self.counts < 0).any()
        ):
            raise ValueError("Invalid positive-user counts")
        self.artifact_id = artifact_id

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        indices = self.items.get_indexer(scope.items)
        valid = indices >= 0
        ids, counts = scope.items[valid], self.counts[indices[valid]]
        evidence = counts > 0
        return select(request, ids[evidence], counts[evidence], limit)


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


class MFRetriever:
    name = "mf"

    def __init__(
        self,
        model: BPRMF,
        users: np.ndarray,
        items: np.ndarray,
        supported_users: np.ndarray,
        artifact_id: str,
    ) -> None:
        self.users, self.items = pd.Index(item_ids(users)), pd.Index(item_ids(items))
        self.supported = np.asarray(supported_users, dtype=bool).copy()
        if (
            self.supported.shape != (len(users),)
            or model.user.num_embeddings != len(users)
            or model.item.num_embeddings != len(items)
        ):
            raise ValueError("MF vocabulary/support shape mismatch")
        self.model = model.cpu().eval()
        self.artifact_id = artifact_id

    @torch.no_grad()
    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        u = self.users.get_indexer([request.user_id])[0]
        if u < 0 or not self.supported[u]:
            return select(request, scope.items[:0], np.empty(0), limit)
        indices = self.items.get_indexer(scope.items)
        known = indices >= 0
        ids, indices = scope.items[known], indices[known]
        updated = self.model.item_updated[torch.from_numpy(indices)].numpy()
        ids, indices = ids[updated], indices[updated]
        users = torch.full((len(ids),), int(u), dtype=torch.long)
        values = self.model.score_pairs(users, torch.from_numpy(indices)).numpy()
        return select(request, ids, values, limit)


def load_retriever(path: Path) -> Retriever:
    artifact = load_artifact(path)
    if artifact.kind == "popularity":
        return PopularityRetriever(artifact.items, artifact.model, artifact.identity)
    if artifact.kind == "cosine":
        return SparseCooccurrenceRetriever(
            artifact.users, artifact.items, artifact.model, artifact.identity
        )
    if artifact.kind == "mf":
        return MFRetriever(
            artifact.model,
            artifact.users,
            artifact.items,
            artifact.supported_users,
            artifact.identity,
        )
    raise ValueError("Expected a fitted recall artifact")
