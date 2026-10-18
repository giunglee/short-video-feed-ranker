"""Keyed prediction replay through the same stage contracts as model inference."""

import numpy as np
import pandas as pd

from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Request,
    RetrievalResult,
    ScoredItems,
)
from feed_ranker.recommendation.ordering import ordered_positions


class ScoreTable:
    def __init__(
        self, frame: pd.DataFrame, artifact_id: str, *, allow_unsupported: bool = False
    ) -> None:
        required = ["query_id", "video_id", "score"]
        if not set(required).issubset(frame.columns) or frame.duplicated(required[:2]).any():
            raise ValueError("Require unique query_id/video_id score keys")
        if (
            frame.query_id.isna().any()
            or not frame.query_id.map(lambda x: isinstance(x, str) and bool(x)).all()
        ):
            raise ValueError("Query IDs must be nonempty strings")
        if frame.video_id.dtype.kind not in "iu" or frame.video_id.lt(0).any():
            raise ValueError("Require nonnegative raw video IDs")
        values = frame.score.to_numpy()
        invalid = np.isnan(values) | np.isposinf(values)
        if invalid.any() or (not allow_unsupported and not np.isfinite(values).all()):
            raise ValueError("Invalid cached predictions")
        self.artifact_id = artifact_id
        self.rows = {
            q: group.set_index("video_id").score.copy() for q, group in frame.groupby("query_id")
        }

    def lookup(self, request: Request, items: np.ndarray) -> np.ndarray:
        if not len(items):
            return np.empty(0, dtype=float)
        if request.query_id not in self.rows:
            raise ValueError(f"Missing cached query: {request.query_id}")
        values = self.rows[request.query_id].reindex(items).to_numpy()
        if np.isnan(values).any():
            raise ValueError("Cached predictions do not cover requested candidates")
        return values


class ReplayRetriever:
    def __init__(self, name: str, table: ScoreTable) -> None:
        self.name, self.table, self.artifact_id = name, table, table.artifact_id

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        values = self.table.lookup(request, scope.items)
        order = ordered_positions(values, request.user_id, scope.items)
        return RetrievalResult(
            ScoredItems(scope.items[order[:limit]], values[order[:limit]]), len(order)
        )


class ReplayRanker:
    def __init__(self, table: ScoreTable) -> None:
        self.table, self.artifact_id = table, table.artifact_id

    def score(self, request: Request, items: np.ndarray) -> ScoredItems:
        return ScoredItems(items, self.table.lookup(request, items))
