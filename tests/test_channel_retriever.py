"""Synthetic allocation and integration checks with no dataset dependency."""

import unittest
from dataclasses import replace

import numpy as np

from feed_ranker.recommendation.contracts import CandidateScope, Request, ScoredItems
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.merging import (
    anchored_candidates,
    merge_channels,
    reciprocal_rank_fusion,
)
from feed_ranker.recommendation.pipeline import RecommendationPipeline
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
            merger = MergedRetriever([a, b], policy, source_depth=None, anchor="a")
            result = RecommendationPipeline(merger, IdentityRanker(), 5, 2).recommend(
                request, scope
            )
            np.testing.assert_array_equal(result.items, [30, 20])
            self.assertEqual((result.pool_shortfall, result.method_shortfall), (1, 1))
            self.assertEqual(result.supported_count, 3)
            self.assertEqual(
                merger.artifact_id,
                MergedRetriever([b, a], policy, source_depth=None, anchor="a").artifact_id,
            )
            if policy == "anchored":
                self.assertNotEqual(
                    merger.artifact_id,
                    MergedRetriever(
                        [a, b], policy, source_depth=None, anchor="a", anchor_fraction=0.6
                    ).artifact_id,
                )
            empty = merger.retrieve(request, CandidateScope(np.array([], dtype=int)), 1)
            self.assertEqual(empty.supported_count, 0)
            self.assertEqual(len(empty.candidates.items), 0)

    def test_singleton_preserves_candidate_order(self) -> None:
        a = PopularityRetriever(np.array([10, 20, 30]), np.array([3, 2, 1]), "a")
        request, scope = Request("7", 7), CandidateScope(np.array([30, 10, 20]))
        for policy in ("round_robin", "rrf"):
            merged = MergedRetriever([a], policy, source_depth=None).retrieve(request, scope, 2)
            np.testing.assert_array_equal(merged.candidates.items, [10, 20])

    def test_invalid_configuration_and_child_support(self) -> None:
        a = PopularityRetriever(np.array([10]), np.array([1]), "a")
        for channels, policy in (([], "rrf"), ([a, a], "rrf"), ([a], "unknown")):
            with self.assertRaises(ValueError):
                MergedRetriever(channels, policy, source_depth=None)
        for constant in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                MergedRetriever([a], "rrf", source_depth=None, rank_constant=constant)
        good = a.retrieve
        a.retrieve = lambda *args: replace(good(*args), supported_count=2)
        with self.assertRaisesRegex(ValueError, "complete supported"):
            MergedRetriever([a], source_depth=None).retrieve(
                Request("7", 7), CandidateScope(np.array([10])), 1
            )

    def test_source_depth_changes_available_support_before_merging(self) -> None:
        a = PopularityRetriever(np.array([10, 20, 30]), np.array([3, 2, 1]), "a")
        b = PopularityRetriever(np.array([10, 20, 30]), np.array([3, 1, 2]), "b")
        a.name, b.name = "a", "b"
        request, scope = Request("q", 7), CandidateScope(np.array([10, 20, 30]))
        for policy in ("round_robin", "rrf", "anchored"):
            full = MergedRetriever([a, b], policy, source_depth=None, anchor="a")
            capped = MergedRetriever([a, b], policy, source_depth=1, anchor="a")
            result = capped.retrieve(request, scope, 3)
            np.testing.assert_array_equal(result.candidates.items, [10])
            self.assertEqual(result.supported_count, 1)
            result = full.retrieve(request, scope, 3)
            self.assertEqual(set(result.candidates.items), {10, 20, 30})
            self.assertEqual(result.supported_count, 3)
            self.assertNotEqual(full.artifact_id, capped.artifact_id)
        for depth in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                MergedRetriever([a], source_depth=depth)
        good = a.retrieve
        a.retrieve = lambda *args: replace(
            good(*args), candidates=ScoredItems(np.array([], dtype=int), np.array([]))
        )
        with self.assertRaisesRegex(ValueError, "requested depth"):
            MergedRetriever([a], source_depth=1).retrieve(request, scope, 3)

    def test_invalid_lists(self) -> None:
        for values in (np.array([1, 1]), np.array([-1]), np.array([1.5])):
            for fn in (merge_channels, reciprocal_rank_fusion):
                with self.assertRaises(ValueError):
                    fn({"a": values}, 2, 7)


if __name__ == "__main__":
    unittest.main()
