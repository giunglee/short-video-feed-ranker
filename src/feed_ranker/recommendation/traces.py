"""Label-free execution traces, including queries that return no candidates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feed_ranker.recommendation.contracts import Recommendation


@dataclass(frozen=True)
class RecommendationTrace:
    """One row per request plus one row per scored candidate."""

    requests: pd.DataFrame
    candidates: pd.DataFrame

    @classmethod
    def from_results(cls, results: list[Recommendation]) -> "RecommendationTrace":
        requests, candidates = [], []
        for result in results:
            requests.append(
                {
                    "query_id": result.request.query_id,
                    "user_id": result.request.user_id,
                    "channel": result.channel,
                    "retrieval_artifact": result.retrieval_artifact,
                    "ranking_artifact": result.ranking_artifact,
                    "pool_size": result.pool_size,
                    "budget": result.budget,
                    "supported_count": result.supported_count,
                    "returned": len(result.retrieved.items),
                    "final_slots": len(result.items),
                    "pool_shortfall": result.pool_shortfall,
                    "method_shortfall": result.method_shortfall,
                }
            )
            final_rank = np.zeros(len(result.retrieved.items), dtype=np.int64)
            final_rank[result.final_positions] = np.arange(1, len(result.items) + 1)
            candidates.append(
                pd.DataFrame(
                    {
                        "query_id": result.request.query_id,
                        "video_id": result.retrieved.items,
                        "source_rank": np.arange(1, len(result.retrieved.items) + 1),
                        "retrieval_score": result.retrieved.scores,
                        "ranking_score": result.ranked.scores,
                        "final_rank": final_rank,
                    }
                )
            )
        if len({r.request.query_id for r in results}) != len(results):
            raise ValueError("Duplicate query IDs in trace")
        if not results:
            raise ValueError("Require at least one request to export a trace")
        return cls(pd.DataFrame(requests), pd.concat(candidates, ignore_index=True))

    def write(self, prefix: Path) -> None:
        """Keep request-level shortfalls/artifact identities even when candidate rows are empty."""
        budgets = self.requests.budget.unique()
        if len(budgets) != 1:
            raise ValueError("Padded candidate export requires a common budget")
        self.requests.to_parquet(f"{prefix}_requests.parquet", index=False)
        self.candidates.to_parquet(f"{prefix}_trace.parquet", index=False)
        packed = np.full((len(self.requests), int(budgets[0])), -1, dtype=np.int64)
        grouped = self.candidates.groupby("query_id", sort=False)
        for u, query_id in enumerate(self.requests.query_id):
            if query_id in grouped.groups:
                items = grouped.get_group(query_id).video_id.to_numpy()
                packed[u, : len(items)] = items
        np.savez_compressed(
            f"{prefix}_candidates.npz", users=self.requests.user_id.to_numpy(), items=packed
        )
