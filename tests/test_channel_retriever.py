"""Synthetic allocation and integration checks with no dataset dependency."""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from feed_ranker.evaluation.channels import ChannelDiagnostics, candidate_recall
from feed_ranker.evaluation.funnel import FunnelEvaluator, GainPool
from feed_ranker.recommendation.channels import ChannelRetriever
from feed_ranker.recommendation.contracts import CandidateScope, Request, ScoredItems
from feed_ranker.recommendation.merging import (
    anchored_candidates,
    merge_channels,
    reciprocal_rank_fusion,
)
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.replay import ReplayRanker, ScoreTable
from feed_ranker.recommendation.retrievers import PopularityRetriever


class IdentityRanker:
    artifact_id = "synthetic"

    def score(self, request: Request, items: np.ndarray) -> ScoredItems:
        return ScoredItems(items, items.astype(float))


class ChannelTests(unittest.TestCase):
    def test_anchor_protection_duplicates_and_exhaustion(self) -> None:
        mf = np.arange(1, 11)
        lists = {"mf": mf, "popularity": np.array([1, 2, 11, 12])}
        np.testing.assert_array_equal(anchored_candidates(lists, 5, 7), [1, 2, 3, 4, 11])
        # Duplicate-only supplements yield their unused capacity to the anchor tail.
        lists["popularity"] = np.array([1, 2])
        np.testing.assert_array_equal(anchored_candidates(lists, 5, 7), [1, 2, 3, 4, 5])
        # A short or unsupported anchor yields capacity to supplements.
        lists["mf"], lists["popularity"] = np.array([1]), np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(anchored_candidates(lists, 5, 7), [1, 2, 3, 4, 5])
        lists["mf"] = np.array([], dtype=int)
        np.testing.assert_array_equal(anchored_candidates(lists, 9, 7), [1, 2, 3, 4, 5])

    def test_anchor_multi_supplement_invariants(self) -> None:
        lists = {
            "mf": np.arange(80),
            "popularity": np.arange(30, 90),
            "cooccurrence": np.arange(95),
        }
        for budget in (1, 20, 50, 100):
            result = anchored_candidates(lists, budget, 7)
            protected = lists["mf"][: int(0.8 * budget)]
            np.testing.assert_array_equal(result[: len(protected)], protected)
            self.assertEqual(len(result), min(budget, 95))
            self.assertEqual(len(set(result)), len(result))
            np.testing.assert_array_equal(
                result, anchored_candidates(dict(reversed(list(lists.items()))), budget, 7)
            )
        for fraction in (0, 1, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                anchored_candidates(lists, 50, 7, fraction=fraction)
        # Caching one merged full-scope ordering and slicing it changes the quota.
        disjoint = {"mf": np.arange(100), "popularity": np.arange(100, 200)}
        small = anchored_candidates(disjoint, 20, 7)
        full = anchored_candidates(disjoint, 200, 7)
        self.assertFalse(np.array_equal(small, full[:20]))
        np.testing.assert_array_equal(small, np.r_[np.arange(16), np.arange(100, 104)])
        with self.assertRaises(ValueError):
            anchored_candidates({"mf": lists["mf"]}, 50, 7)

    def test_round_robin_duplicates_exhaustion_and_prefix(self) -> None:
        lists = {"a": np.array([1, 2, 3]), "b": np.array([1, 4])}
        full = merge_channels(lists, 9, 7)
        self.assertEqual(set(full.items), {1, 2, 3, 4})
        self.assertEqual(full.duplicate_skips.sum(), 1)
        for size in (1, 2, 3):
            np.testing.assert_array_equal(merge_channels(lists, size, 7).items, full.items[:size])
        np.testing.assert_array_equal(
            merge_channels(dict(reversed(list(lists.items()))), 9, 7).items, full.items
        )

    def test_rrf_exact_contributions_and_ties(self) -> None:
        lists = {"a": np.array([1, 2]), "b": np.array([2, 1, 3])}
        result = reciprocal_rank_fusion(lists, 3, 7)
        self.assertEqual(set(result.items[:2]), {1, 2})
        np.testing.assert_allclose(result.scores, [1 / 61 + 1 / 62] * 2 + [1 / 63])
        np.testing.assert_array_equal(
            reciprocal_rank_fusion(dict(reversed(list(lists.items()))), 2, 7).items,
            result.items[:2],
        )

    def test_adapter_runs_unmodified_pipeline(self) -> None:
        a = PopularityRetriever(np.array([10, 20, 30]), np.array([3, 2, 0]), "a")
        b = PopularityRetriever(np.array([10, 20, 30]), np.array([0, 3, 1]), "b")
        a.name, b.name = "a", "b"
        request, scope = Request("7", 7), CandidateScope(np.array([10, 20, 30, 40]))
        for policy in ("round_robin", "rrf", "anchored"):
            merger = ChannelRetriever([a, b], policy, anchor="a")
            result = RecommendationPipeline(merger, IdentityRanker(), 5, 2).recommend(
                request, scope
            )
            np.testing.assert_array_equal(result.items, [30, 20])
            self.assertEqual((result.pool_shortfall, result.method_shortfall), (1, 1))
            self.assertEqual(result.supported_count, 3)
            self.assertEqual(
                merger.artifact_id, ChannelRetriever([b, a], policy, anchor="a").artifact_id
            )
            if policy == "anchored":
                self.assertNotEqual(
                    merger.artifact_id,
                    ChannelRetriever([a, b], policy, anchor="a", anchor_fraction=0.6).artifact_id,
                )
            empty = merger.retrieve(request, CandidateScope(np.array([], dtype=int)), 1)
            self.assertEqual(empty.supported_count, 0)
            self.assertEqual(len(empty.candidates.items), 0)

    def test_singleton_preserves_candidate_order(self) -> None:
        a = PopularityRetriever(np.array([10, 20, 30]), np.array([3, 2, 1]), "a")
        request, scope = Request("7", 7), CandidateScope(np.array([30, 10, 20]))
        for policy in ("round_robin", "rrf"):
            merged = ChannelRetriever([a], policy).retrieve(request, scope, 2)
            np.testing.assert_array_equal(merged.candidates.items, [10, 20])

    def test_invalid_configuration_and_child_support(self) -> None:
        a = PopularityRetriever(np.array([10]), np.array([1]), "a")
        for channels, policy in (([], "rrf"), ([a, a], "rrf"), ([a], "unknown")):
            with self.assertRaises(ValueError):
                ChannelRetriever(channels, policy)
        for constant in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                ChannelRetriever([a], "rrf", rank_constant=constant)
        good = a.retrieve
        a.retrieve = lambda *args: replace(good(*args), supported_count=2)
        with self.assertRaisesRegex(ValueError, "complete supported"):
            ChannelRetriever([a]).retrieve(Request("7", 7), CandidateScope(np.array([10])), 1)

    def test_invalid_lists(self) -> None:
        for values in (np.array([1, 1]), np.array([-1]), np.array([1.5])):
            for fn in (merge_channels, reciprocal_rank_fusion):
                with self.assertRaises(ValueError):
                    fn({"a": values}, 2, 7)

    def test_displacement_diagnosis_matches_executed_delta(self) -> None:
        ids = np.array([10, 20, 30])
        pool = GainPool(
            np.array([7]),
            ids,
            np.array([0, 3]),
            np.array([2.0, 1.0, 0.1]),
            np.array([5000.0, 10000.0, 60000.0]),
            k=2,
        )
        predictions = np.array([0.9, 0.8, 1.2])
        ranker = ReplayRanker(
            ScoreTable(
                pd.DataFrame(
                    {
                        "query_id": ["7"] * 3,
                        "video_id": ids,
                        "score": predictions,
                    }
                ),
                "synthetic",
            )
        )
        request, scope = Request("7", 7), CandidateScope(ids)
        base = RecommendationPipeline(
            PopularityRetriever(ids, np.array([2, 1, 0]), "base"), ranker, 2, 2
        ).recommend(request, scope)
        merged = RecommendationPipeline(
            PopularityRetriever(ids, np.array([0, 2, 1]), "merged"), ranker, 2, 2
        ).recommend(request, scope)
        evaluator = FunnelEvaluator(pool)
        delta = (
            evaluator.evaluate([merged]).metrics["cap2_ndcg10"]
            - evaluator.evaluate([base]).metrics["cap2_ndcg10"]
        )
        rows, changed = ChannelDiagnostics(
            pool,
            np.array([True, True, False]),
            predictions,
            {"mf": np.array([1, 2, 3])},
        ).compare([base], [merged])
        row = rows.iloc[0]
        self.assertAlmostEqual(row.removal_contribution + row.addition_contribution, delta)
        self.assertEqual(row.mf_final_items_removed, 1)
        self.assertEqual(row.added_items_in_final, 1)
        self.assertEqual(row.positive_added, 0)
        self.assertEqual(row.positive_lost, 1)
        self.assertEqual(dict(zip(changed.video_id, changed.action)), {30: "added", 10: "lost"})
        recalled = candidate_recall(pool, np.array([True, True, False]), [merged])
        self.assertEqual(recalled.iloc[0].binary_recall, 0.5)
        empty = candidate_recall(pool, np.zeros(3, dtype=bool), [merged])
        self.assertTrue(np.isnan(empty.iloc[0].binary_recall))
        with self.assertRaises(ValueError):
            candidate_recall(pool, np.ones(2, dtype=bool), [merged])


if __name__ == "__main__":
    unittest.main()
