"""Raw IDs at stage boundaries; no labels or evaluation-pool positions."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def item_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
        raise ValueError("Item IDs must be a nonnegative integer vector")
    if len(np.unique(values)) != len(values):
        raise ValueError("Duplicate item IDs")
    result = values.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Request:
    query_id: str
    user_id: int

    def __post_init__(self) -> None:
        if not self.query_id or not isinstance(self.query_id, str):
            raise ValueError("Require a nonempty query ID")
        if not isinstance(self.user_id, (int, np.integer)) or self.user_id < 0:
            raise ValueError("Require a nonnegative raw user ID")


@dataclass(frozen=True)
class CandidateScope:
    """Already eligible IDs: catalog/seen-item policy belongs to the caller."""

    items: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", item_ids(self.items))


@dataclass(frozen=True)
class ScoredItems:
    """Aligned item IDs and finite scores. Unsupported items are omitted explicitly."""

    items: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", item_ids(self.items))
        values = np.asarray(self.scores)
        if values.shape != self.items.shape or not np.isfinite(values).all():
            raise ValueError("Require one finite score per item")
        values = values.copy()
        values.flags.writeable = False
        object.__setattr__(self, "scores", values)


@dataclass(frozen=True)
class RetrievalResult:
    candidates: ScoredItems
    supported_count: int


class Retriever(Protocol):
    name: str
    artifact_id: str

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult: ...


class Ranker(Protocol):
    artifact_id: str

    def score(self, request: Request, items: np.ndarray) -> ScoredItems: ...


@dataclass(frozen=True)
class Recommendation:
    """Response plus compact trace; final_positions indexes both score vectors."""

    request: Request
    channel: str
    retrieval_artifact: str
    ranking_artifact: str
    pool_size: int
    budget: int
    supported_count: int
    retrieved: ScoredItems
    ranked: ScoredItems
    final_positions: np.ndarray

    @property
    def items(self) -> np.ndarray:
        return self.ranked.items[self.final_positions]

    @property
    def pool_shortfall(self) -> int:
        return max(0, self.budget - self.pool_size)

    @property
    def method_shortfall(self) -> int:
        return min(self.budget, self.pool_size) - len(self.retrieved.items)
