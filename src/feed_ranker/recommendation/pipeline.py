"""Single-retriever execution. Offline labels and report writing stay outside."""

import numpy as np

from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Ranker,
    Recommendation,
    Request,
    Retriever,
)
from feed_ranker.recommendation.ordering import ordered_positions


class RecommendationPipeline:
    def __init__(self, retriever: Retriever, ranker: Ranker, budget: int = 50, k: int = 10) -> None:
        if type(budget) is not int or type(k) is not int or budget < 1 or k < 1:
            raise ValueError("Budget and K must be positive integers")
        self.retriever, self.ranker = retriever, ranker
        self.budget, self.k = budget, k

    def recommend(self, request: Request, scope: CandidateScope) -> Recommendation:
        result = self.retriever.retrieve(request, scope, self.budget)
        candidates = result.candidates
        if type(result.supported_count) is not int or not 0 <= result.supported_count <= len(
            scope.items
        ):
            raise ValueError("Invalid retriever support count")
        if len(candidates.items) != min(self.budget, result.supported_count):
            raise ValueError("Retriever must fill the budget up to its declared support")
        if not np.isin(candidates.items, scope.items).all():
            raise ValueError("Retriever returned ineligible items")
        ranked = self.ranker.score(request, candidates.items)
        if not np.array_equal(ranked.items, candidates.items):
            raise ValueError("Ranker changed candidate identity or alignment")
        final = ordered_positions(ranked.scores, request.user_id, ranked.items)[: self.k]
        return Recommendation(
            request,
            self.retriever.name,
            self.retriever.artifact_id,
            self.ranker.artifact_id,
            len(scope.items),
            self.budget,
            result.supported_count,
            candidates,
            ranked,
            final,
        )
