"""Public fixtures for snapshot vocabularies, cold items and frozen merge policies."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from feed_ranker.preprocessing.features import ANONYMOUS, fit_entity_features
from feed_ranker.recommendation.contracts import CandidateScope, Request
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.merging import merge_channels, reciprocal_rank_fusion
from feed_ranker.recommendation.replay import ReplayRetriever, ScoreTable


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "video_id": [10, 20, 20, 30, 10, 40, 30, 50],
            "time_ms": [5, 6, 5, 6, 10, 11, 10, 11],
            "duration_ms": [100, 200, 200, 300, 999, 999, 999, 999],
            "long_view": [1, 0, 1, 0, 1, 0, 0, 1],
            "is_rand": [0] * 8,
            "tab": [1] * 8,
        }
    )
    basic = pd.DataFrame(
        {
            "video_id": [10, 20, 30, 40, 50],
            "author_id": [7, 8, 9, 7, 999],
            "video_duration": [110.0, 220.0, np.nan, 440.0, 550.0],
        }
    )
    profile = pd.DataFrame(
        {"user_id": [1, 2, 99], **{name: [0.0, -124.0, 999.0] for name in ANONYMOUS}}
    )
    return events, basic, profile


class MetadataTests(unittest.TestCase):
    def test_merge_adapter_matches_fixed_policies_and_empty_support(self) -> None:
        sources = []
        ranked = {}
        for name, items in {"a": [1, 2, 3], "b": [2, 4, 5], "c": [5, 3, 6]}.items():
            ranked[name] = np.array(items)
            sources.append(
                ReplayRetriever(
                    name,
                    ScoreTable(
                        pd.DataFrame(
                            {
                                "query_id": ["q"] * 3,
                                "video_id": items,
                                "score": [3.0, 2.0, 1.0],
                            }
                        ),
                        name,
                    ),
                )
            )
        query, scope = Request("q", 1), CandidateScope(np.arange(1, 7))
        for source in sources:
            source.table.rows["q"] = source.table.rows["q"].reindex(scope.items, fill_value=-np.inf)
        for name in ("round_robin", "rrf"):
            retriever = MergedRetriever(sources, name, source_depth=3)
            actual = retriever.retrieve(query, scope, 4)
            expected = (
                merge_channels(ranked, 4, 1).items
                if name == "round_robin"
                else reciprocal_rank_fusion(ranked, 4, 1).items
            )
            np.testing.assert_array_equal(actual.candidates.items, expected)
            self.assertEqual(actual.supported_count, 6)
            empty = retriever.retrieve(query, CandidateScope(np.array([], dtype=np.int64)), 4)
            self.assertEqual(empty.supported_count, 0)
            self.assertEqual(len(empty.candidates.items), 0)

    def test_training_vocabularies_and_unknown_metadata_remain_distinct(self) -> None:
        events, basic, profiles = fixture()
        train = events.loc[events.time_ms.lt(10)]
        state = fit_entity_features(train, basic, profiles)
        values = state.transform(np.array([1, 2, 99]), np.array([40, 50, 999]))
        self.assertEqual(values["item_code"].tolist(), [0, 0, 0])
        self.assertEqual(values["author_code"].tolist(), [1, 0, 0])
        self.assertEqual(values["onehot_feat0"].tolist(), [1, 0, 0])
        np.testing.assert_allclose(
            values["metadata_duration_log1p"], np.log1p([440, 550, 0]), rtol=1e-7
        )
        self.assertEqual(state.vocab_sizes["author_code"], 4)
        self.assertEqual(state.vocab_sizes["onehot_feat0"], 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_entity_features(train, pd.concat([basic, basic.iloc[[0]]]), profiles)
        changed = basic.copy()
        changed.loc[changed.video_id.eq(50), "author_id"] = 12345
        after = fit_entity_features(train, changed, profiles)
        self.assertEqual(after.vocab_sizes, state.vocab_sizes)
        np.testing.assert_array_equal(
            after.item_features["author_code"], state.item_features["author_code"]
        )

    def test_lookup_rows_are_independent_of_embedding_codes(self) -> None:
        train = pd.DataFrame(
            {"user_id": [10, 10], "video_id": [100, 300], "duration_ms": [1000, 3000]}
        )
        basic = pd.DataFrame(
            {
                "video_id": [100, 200, 300],
                "author_id": [42, 42, 900],
                "video_duration": [1000, 2000, 3000],
            }
        )
        profiles = pd.DataFrame({"user_id": [10], **{n: [0] for n in ANONYMOUS}})
        lookup = fit_entity_features(train, basic, profiles)
        self.assertEqual(lookup.item_ids.tolist(), [100, 200, 300])
        values = lookup.transform(
            user_ids=np.array([10, 10, 10]), item_ids=np.array([200, 300, 999])
        )
        self.assertEqual(values["item_code"].tolist(), [0, 2, 0])
        self.assertEqual(values["author_code"].tolist(), [1, 2, 0])
        self.assertEqual(values["metadata_duration_missing"].tolist(), [0, 0, 1])
        self.assertEqual(values["train_duration_missing"].tolist(), [1, 0, 1])
