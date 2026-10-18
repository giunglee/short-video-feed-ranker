"""Behavioral boundaries of composed inference, independent of model training."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from feed_ranker.models.recall import BPRMF, baseline_scores
from feed_ranker.recommendation.contracts import (
    CandidateScope,
    Request,
    RetrievalResult,
    ScoredItems,
)
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.replay import ReplayRanker, ReplayRetriever, ScoreTable
from feed_ranker.recommendation.retrieval import (
    CooccurrenceRetriever,
    MFRetriever,
    PopularityRetriever,
)
from feed_ranker.training.funnel_evaluation import GainPool


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
        data = SimpleNamespace(
            users=users,
            items=items,
            positive=positives,
            pools=SimpleNamespace(
                items=np.tile(np.arange(4), 3),
                users=np.repeat(np.arange(3), 4),
                offsets=np.arange(0, 13, 4),
            ),
        )
        cooc = CooccurrenceRetriever(users, items, h, "fit")
        expected, _ = baseline_scores(data, "cooccurrence")
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

    def test_evaluator_consumes_final_order_and_full_pool_labels(self) -> None:
        pool = GainPool(
            np.array([7]),
            np.array([10, 20, 30]),
            np.array([0, 3]),
            np.array([2.0, 1.0, 0.0]),
            np.ones(3),
            k=2,
        )
        # Candidate order and final order need not match; evaluator must not reorder.
        metrics, _ = pool.evaluate_ordered([np.array([1, 0])], [np.array([1, 0])])
        expected = (1 + 2 / np.log2(3)) / (2 + 1 / np.log2(3))
        self.assertAlmostEqual(metrics["cap2_ndcg10"], expected)
        short, _ = pool.evaluate_ordered([np.array([1])], [np.array([1])])
        self.assertLess(short["candidate_oracle_ndcg10"], 1)
        # Labels affect evaluation only; runtime request has no label field.
        pipeline = RecommendationPipeline(self.pop, RecordingRanker(), 2, 1)
        before = pipeline.recommend(self.request, self.scope)
        pool.ratios[:] = 100
        after = pipeline.recommend(self.request, self.scope)
        np.testing.assert_array_equal(before.items, after.items)


if __name__ == "__main__":
    unittest.main()
