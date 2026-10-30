"""Behavioral boundaries of composed inference, independent of model training."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.evaluation.funnel import WindowBenchmark
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Request,
    RetrievalResult,
    ScoredItems,
)
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.replay import ReplayRanker, ReplayRetriever, ScoreTable
from feed_ranker.recommendation.retrievers import (
    MFRetriever,
    PopularityRetriever,
    SparseCooccurrenceRetriever,
)
from feed_ranker.recommendation.traces import RecommendationTrace


class RecordingRanker:
    artifact_id = "synthetic"

    def score(self, request: Request, items: np.ndarray) -> ScoredItems:
        self.received = items.copy()
        return ScoredItems(items, -items.astype(float))


class RecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.request = Request("first-request", 7)
        self.scope = CandidateScope(np.array([10, 20, 30, 40]))
        self.pop = PopularityRetriever(self.scope.items, np.array([1, 3, 2, 0]), "counts-v1")

    def test_rank_only_candidates_and_swappable_ranker(self) -> None:
        ranker = RecordingRanker()
        result = RecommendationPipeline(self.pop, ranker, 2, 1).recommend(self.request, self.scope)
        np.testing.assert_array_equal(ranker.received, [20, 30])
        np.testing.assert_array_equal(result.items, [20])
        self.assertEqual(result.supported_count, 3)
        self.assertEqual(result.retrieval_artifact, "counts-v1")
        # Replay needs scores only for retrieved items, not every pool member.
        table = ScoreTable(
            pd.DataFrame(
                {"query_id": ["first-request"] * 2, "video_id": [30, 20], "score": [5.0, 1.0]}
            ),
            "saved",
        )
        other = RecommendationPipeline(self.pop, ReplayRanker(table), 2, 1).recommend(
            self.request, self.scope
        )
        np.testing.assert_array_equal(other.items, [30])

    def test_pool_and_method_shortfall_empty_scope(self) -> None:
        pipeline = RecommendationPipeline(self.pop, RecordingRanker(), 5, 10)
        result = pipeline.recommend(self.request, self.scope)
        self.assertEqual((result.pool_shortfall, result.method_shortfall), (1, 1))
        self.assertEqual(len(result.items), 3)
        empty = pipeline.recommend(self.request, CandidateScope(np.empty(0, dtype=int)))
        self.assertEqual(
            (len(empty.items), empty.pool_shortfall, empty.method_shortfall), (0, 5, 0)
        )

    def test_ties_are_independent_of_input_order(self) -> None:
        pop = PopularityRetriever(self.scope.items, np.ones(4), "ties")
        pipe = RecommendationPipeline(pop, RecordingRanker(), 2, 2)
        a = pipe.recommend(self.request, self.scope)
        b = pipe.recommend(self.request, CandidateScope(self.scope.items[::-1]))
        np.testing.assert_array_equal(a.retrieved.items, b.retrieved.items)

    def test_query_keys_allow_repeated_user_and_missing_scores_fail(self) -> None:
        frame = pd.DataFrame(
            {
                "query_id": ["first-request", "second-request"],
                "video_id": [10, 10],
                "score": [1.0, 2.0],
            }
        )
        table = ScoreTable(frame, "saved")
        self.assertEqual(table.lookup(self.request, np.array([10]))[0], 1)
        self.assertEqual(table.lookup(Request("second-request", 7), np.array([10]))[0], 2)
        with self.assertRaisesRegex(ValueError, "cover"):
            table.lookup(self.request, np.array([20]))
        with self.assertRaises(ValueError):
            ScoreTable(pd.concat([frame, frame]), "bad")
        with self.assertRaises(ValueError):
            ScoreTable(frame.assign(score=np.nan), "bad")

    def test_invalid_stage_outputs_fail(self) -> None:
        for items, scores in [
            (np.array([10, 10]), np.ones(2)),
            (np.array([10]), np.array([np.inf])),
        ]:
            with self.assertRaises(ValueError):
                ScoredItems(items, scores)

        class BadRetriever:
            name, artifact_id = "bad", "bad"

            def retrieve(self, request, scope, limit):
                return RetrievalResult(ScoredItems(np.array([999]), np.array([1.0])), 1)

        with self.assertRaisesRegex(ValueError, "ineligible"):
            RecommendationPipeline(BadRetriever(), RecordingRanker()).recommend(
                self.request, self.scope
            )

        class BadRanker:
            artifact_id = "bad"

            def score(self, request, items):
                return ScoredItems(items[::-1], np.ones(len(items)))

        with self.assertRaisesRegex(ValueError, "alignment"):
            RecommendationPipeline(self.pop, BadRanker()).recommend(self.request, self.scope)

    def test_replay_unsupported_is_not_a_silent_ranker_fallback(self) -> None:
        frame = pd.DataFrame(
            {
                "query_id": ["first-request"] * 4,
                "video_id": self.scope.items,
                "score": [1.0, -np.inf, 3.0, -np.inf],
            }
        )
        table = ScoreTable(frame, "legacy", allow_unsupported=True)
        retriever = ReplayRetriever("mf", table)
        result = RecommendationPipeline(retriever, RecordingRanker()).recommend(
            self.request, self.scope
        )
        self.assertEqual(result.supported_count, 2)
        self.assertEqual(result.method_shortfall, 2)
        with self.assertRaises(ValueError):
            ReplayRanker(table).score(self.request, self.scope.items)

    def test_model_retrievers_match_reference_and_handle_unknown_ids(self) -> None:
        users, items = np.array([7, 8, 9]), np.array([10, 20, 30, 40])
        h = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        positives = [np.flatnonzero(row.numpy()) for row in h]
        cooc = SparseCooccurrenceRetriever(
            users, items, SparseCosineCooccurrence(positives, 4), "fit"
        )
        expected = np.array([0.5, 0.5, 1.0, 0.0])
        expected[expected <= 0] = -np.inf
        got = cooc.retrieve(self.request, self.scope, 10).candidates
        index = pd.Index(items).get_indexer(got.items)
        np.testing.assert_allclose(got.scores, expected[index])
        np.testing.assert_allclose(got.scores, np.array([0.5, 0.5, 1.0, 0.0])[index])
        self.assertNotIn(40, got.items)
        self.assertEqual(cooc.retrieve(Request("cold", 99), self.scope, 10).supported_count, 0)
        model = BPRMF(3, 4, 2)
        model.item_updated[:] = torch.tensor([True, True, False, True])
        mf = MFRetriever(model, users, items, np.array([True, False, True]), "checkpoint")
        unknown_scope = CandidateScope(np.array([10, 20, 30, 40, 999]))
        result = mf.retrieve(self.request, unknown_scope, 10)
        self.assertEqual(set(result.candidates.items), {10, 20, 40})
        self.assertEqual(mf.retrieve(Request("unsupported", 8), self.scope, 10).supported_count, 0)
        self.assertEqual(mf.retrieve(Request("cold", 99), self.scope, 10).supported_count, 0)

    def test_trace_preserves_empty_queries_artifacts_and_selected_order(self) -> None:
        pipe = RecommendationPipeline(self.pop, RecordingRanker(), 3, 2)
        first = pipe.recommend(self.request, self.scope)
        empty = pipe.recommend(Request("empty", 8), CandidateScope(np.array([40])))
        trace = RecommendationTrace.from_results([first, empty])
        self.assertEqual(len(trace.requests), 2)
        self.assertEqual(trace.requests.method_shortfall.tolist(), [0, 1])
        self.assertEqual(trace.requests.retrieval_artifact.tolist(), ["counts-v1"] * 2)
        np.testing.assert_array_equal(
            trace.candidates.sort_values("final_rank").query("final_rank > 0").video_id, first.items
        )
        self.assertFalse(set(trace.requests) & {"watch_ratio", "label", "outcome"})
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "result"
            trace.write(prefix)
            packed = np.load(f"{prefix}_candidates.npz")
            np.testing.assert_array_equal(packed["items"][1], [-1, -1, -1])
            self.assertEqual(len(pd.read_parquet(f"{prefix}_requests.parquet")), 2)
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            RecommendationTrace.from_results([first, first])

    def test_full_pool_replay_preserves_float64_ordering(self) -> None:
        items = np.array([10, 20, 30])
        values = np.array([1.0, 1.0 + 1e-10, -np.inf])
        frame = pd.DataFrame({"query_id": ["q"] * 3, "video_id": items, "score": values})
        retriever = ReplayRetriever("fixture", ScoreTable(frame, "saved", allow_unsupported=True))
        candidates = retriever.retrieve(Request("q", 7), CandidateScope(items), 3).candidates
        got = (
            pd.Series(candidates.scores, index=candidates.items)
            .reindex(items, fill_value=-np.inf)
            .to_numpy()
        )
        np.testing.assert_array_equal(got, values)

    def test_full_scope_cache_rejects_incomplete_or_invalid_support(self) -> None:
        benchmark = WindowBenchmark(
            pd.DataFrame({"user_id": [7, 7, 7], "video_id": [10, 20, 30], "long_view": [1, 0, 1]}),
            np.array([10, 20, 30]),
            "long_view",
            0,
            1,
        )
        retriever = PopularityRetriever(np.array([10, 20, 30]), np.array([3, 2, 0]), "fixture")
        good = retriever.retrieve
        request, scope = benchmark.queries[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for support in (0, 3, -1, 2.0, True):
                retriever.retrieve = lambda *args: replace(good(*args), supported_count=support)
                path = root / "bad.parquet"
                with self.subTest(support=support):
                    with self.assertRaisesRegex(ValueError, "complete supported"):
                        benchmark.cache(retriever, path)
                    self.assertFalse(path.exists())
            retriever.retrieve = good
            cached = benchmark.cache(retriever, root / "good.parquet")
            actual, expected = cached.retrieve(request, scope, 3), good(request, scope, 3)
            np.testing.assert_array_equal(actual.candidates.items, expected.candidates.items)
            np.testing.assert_array_equal(actual.candidates.scores, expected.candidates.scores)
            self.assertEqual(actual.supported_count, 2)


if __name__ == "__main__":
    unittest.main()
