"""KuaiRand observed-window retrieval/ranking evaluation; labels stay outside inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.data.prepared import PreparedDataset
from feed_ranker.evaluation.metrics import candidate_metrics
from feed_ranker.persistence import write_json
from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Ranker,
    Recommendation,
    Request,
    Retriever,
)
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.replay import ReplayRetriever, ScoreTable
from feed_ranker.recommendation.traces import RecommendationTrace


class WindowBenchmark:
    """KuaiRand warm observed-window candidates and positive-ever pair relevance."""

    def __init__(
        self, val: pd.DataFrame, items: np.ndarray, label: str, start: int, end: int
    ) -> None:
        self.queries: list[tuple[Request, CandidateScope]] = []
        self.positive: dict[str, np.ndarray] = {}
        self.all_positive: dict[str, int] = {}
        self.coverage = {
            "all_pairs": 0,
            "warm_pairs": 0,
            "all_positive_pairs": 0,
            "warm_positive_pairs": 0,
        }
        catalog = pd.Index(items)
        pairs = val.groupby(["user_id", "video_id"], as_index=False)[label].max()
        for user, frame in pairs.groupby("user_id", sort=True):
            request = Request(f"{int(user)}:{start}:{end}", int(user))
            warm = catalog.get_indexer(frame.video_id) >= 0
            scope = CandidateScope(frame.loc[warm, "video_id"].to_numpy(dtype=np.int64))
            relevant = frame.loc[warm & frame[label].eq(1), "video_id"].to_numpy(dtype=np.int64)
            self.queries.append((request, scope))
            self.positive[request.query_id] = relevant
            self.all_positive[request.query_id] = int(frame[label].sum())
            for key, count in [
                ("all_pairs", len(frame)),
                ("warm_pairs", len(scope.items)),
                ("all_positive_pairs", int(frame[label].sum())),
                ("warm_positive_pairs", len(relevant)),
            ]:
                self.coverage[key] += count

    def cache(self, retriever: Retriever, path: Path) -> ReplayRetriever:
        """Freeze full supported channel rankings once, before any ranker comparison."""
        frames = []
        for request, scope in self.queries:
            retrieved = retriever.retrieve(request, scope, max(1, len(scope.items)))
            candidates = retrieved.candidates
            if (
                type(retrieved.supported_count) is not int
                or retrieved.supported_count != len(candidates.items)
                or not np.isin(candidates.items, scope.items).all()
            ):
                raise ValueError("Retriever must return its complete supported benchmark scope")
            scores = np.full(len(scope.items), -np.inf)
            scores[pd.Index(scope.items).get_indexer(candidates.items)] = candidates.scores
            frames.append(
                pd.DataFrame(
                    {"query_id": request.query_id, "video_id": scope.items, "score": scores}
                )
            )
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(path, index=False)
        return ReplayRetriever(
            retriever.name, ScoreTable(frame, retriever.artifact_id, allow_unsupported=True)
        )

    def evaluate(self, results: list[Recommendation]) -> tuple[dict[str, Any], pd.DataFrame]:
        if len(results) != len(self.queries):
            raise ValueError("Query count mismatch")
        rows = []
        for (request, scope), result in zip(self.queries, results, strict=True):
            if result.request != request or result.pool_size != len(scope.items):
                raise ValueError("Query identity/scope mismatch")
            if not np.isin(result.retrieved.items, scope.items).all():
                raise ValueError("Out-of-pool candidates")
            rest = result.retrieved.items[~np.isin(result.retrieved.items, result.items)]
            ordered = np.r_[result.items, rest]
            relevant = self.positive[request.query_id]
            metrics = candidate_metrics(ordered, relevant)
            all_positive = self.all_positive[request.query_id]
            hits = int(np.isin(result.retrieved.items, relevant).sum())
            rows.append(
                {
                    "user_id": request.user_id,
                    **metrics,
                    "all_target_recall": hits / all_positive if all_positive else None,
                    "all_positive_items": all_positive,
                    "warm_positive_items": len(relevant),
                    "pool_items": len(scope.items),
                    "returned": len(ordered),
                    "pool_shortfall": result.pool_shortfall,
                    "method_shortfall": result.method_shortfall,
                }
            )
        per_user = pd.DataFrame(rows)
        summary = {
            name: float(per_user[name].mean()) if per_user[name].notna().any() else None
            for name in ["recall", "ndcg", "oracle_ndcg", "all_target_recall", "returned"]
        }
        summary.update(
            queries=len(rows),
            positive_warm_queries=int(per_user.recall.notna().sum()),
            positive_all_queries=int(per_user.all_target_recall.notna().sum()),
            pool_shortfall_queries=int(per_user.pool_shortfall.gt(0).sum()),
            method_shortfall_queries=int(per_user.method_shortfall.gt(0).sum()),
        )
        return summary, per_user


def open_benchmark(data_path: Path, target: str = "long_view") -> WindowBenchmark:
    data = PreparedDataset(data_path)
    if target not in data.manifest["targets"]:
        raise ValueError("Benchmark target must be a prepared target")
    state = data.load_feature_state()
    warm_items = state.items[state.columns["item_code"][1:] > 0]
    return WindowBenchmark(
        data.read_split("validation", ["user_id", "video_id", target]),
        warm_items,
        target,
        data.manifest["train_end"],
        data.manifest["val_end"],
    )


def evaluate_pipeline(
    benchmark: WindowBenchmark,
    ranker: Ranker,
    retriever: Retriever,
    directory: Path,
    *,
    budget: int = 50,
) -> dict[str, Any]:
    """Current metric protocol is top10, full warm-observed-pool IDCG."""
    directory.mkdir(parents=True, exist_ok=False)
    pipeline = RecommendationPipeline(retriever, ranker, budget, 10)
    results = [pipeline.recommend(request, scope) for request, scope in benchmark.queries]
    metrics, per_user = benchmark.evaluate(results)
    report = {
        "metrics": metrics,
        "coverage": benchmark.coverage,
        "budget": budget,
        "k": 10,
        "retrieval_artifact": retriever.artifact_id,
        "ranking_artifact": ranker.artifact_id,
        "scope": "reused validation; warm observed-window candidates; no final-window scoring",
    }
    write_json(directory / "metrics.json", report)
    per_user.to_parquet(directory / "per_user.parquet", index=False)
    RecommendationTrace.from_results(results).write(directory / "recommendation")
    return report
