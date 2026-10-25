"""Public fixtures for snapshot vocabularies, cold items and frozen merge policies."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import sha256
from feed_ranker.data.kuairand import EventDataset, load_prepared, prepare
from feed_ranker.experiments.kuairand_study import run
from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures
from feed_ranker.features.kuairand_metadata import ANONYMOUS, MetadataEncoding
from feed_ranker.inference.kuairand import load_ranker
from feed_ranker.inference.merged import MergedRetriever
from feed_ranker.inference.replay import ReplayRetriever, ScoreTable
from feed_ranker.recommendation.contracts import CandidateScope, Request
from feed_ranker.recommendation.merging import merge_channels, reciprocal_rank_fusion


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
    def test_author_duration_is_combined_without_anonymous_inputs(self) -> None:
        events, basic, profiles = fixture()
        encoding = Encoding.fit(events.loc[events.time_ms.lt(10)])
        metadata = MetadataEncoding.fit(encoding.users, encoding.items, basic, profiles)
        combined = KuaiRandFeatures(encoding, "band_combined", metadata)
        ablated = KuaiRandFeatures(encoding, "band_author_duration", metadata)
        users, items = np.array([1, 2, 99]), np.array([10, 40, 50])
        a, b = combined.build(users, items), ablated.build(users, items)
        self.assertEqual(set(b), set(a) - set(ANONYMOUS))
        for key in b:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertIn("author_id", b)
        self.assertEqual(b["duration_ms"].tolist(), [110.0, 440.0, 550.0])

    def test_train_only_vocabularies_and_cold_feature_identity(self) -> None:
        events, basic, profiles = fixture()
        encoding = Encoding.fit(events.loc[events.time_ms.lt(10)])
        metadata = MetadataEncoding.fit(encoding.users, encoding.items, basic, profiles)
        builder = KuaiRandFeatures(encoding, "band_combined", metadata)
        values = builder.build(np.array([1, 2, 99]), np.array([40, 50, 999]))
        self.assertEqual(values["video_id"].tolist(), [0, 0, 0])
        self.assertEqual(values["author_id"].tolist(), [1, 0, 0])
        self.assertEqual(values["duration_ms"].tolist(), [440.0, 550.0, 0.0])
        self.assertEqual(values["onehot_feat0"].tolist(), [1, 0, 0])
        self.assertEqual(metadata.author_vocabulary.tolist(), [7, 8, 9])
        self.assertEqual(metadata.anonymous_vocabularies[0].tolist(), [0])
        changed = basic.copy()
        changed.loc[changed.video_id.eq(50), "author_id"] = 777
        changed_profiles = profiles.copy()
        changed_profiles.loc[changed_profiles.user_id.eq(99), list(ANONYMOUS)] = 777
        again = MetadataEncoding.fit(encoding.users, encoding.items, changed, changed_profiles)
        np.testing.assert_array_equal(again.author_vocabulary, metadata.author_vocabulary)
        for a, b in zip(again.anonymous_vocabularies, metadata.anonymous_vocabularies, strict=True):
            np.testing.assert_array_equal(a, b)
        original = EventDataset(events, encoding, "long_view", "band_combined", metadata)
        poisoned = EventDataset(
            events.assign(duration_ms=999999, long_view=1 - events.long_view),
            encoding,
            "long_view",
            "band_combined",
            metadata,
        )
        for key in original.features:
            torch.testing.assert_close(original.features[key], poisoned.features[key])
        with self.assertRaisesRegex(ValueError, "unique"):
            MetadataEncoding.fit(
                encoding.users, encoding.items, pd.concat([basic, basic.iloc[[0]]]), profiles
            )
        with self.assertRaisesRegex(ValueError, "requires fitted metadata"):
            KuaiRandFeatures(encoding, "band_author")

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
        for name in ("rr_all", "rrf_all"):
            retriever = MergedRetriever(sources, name, 3)
            actual = retriever.retrieve(query, scope, 4)
            expected = (
                merge_channels(ranked, 4, 1).items
                if name == "rr_all"
                else reciprocal_rank_fusion(ranked, 4, 1).items
            )
            np.testing.assert_array_equal(actual.candidates.items, expected)
            self.assertEqual(actual.supported_count, 6)
            empty = retriever.retrieve(query, CandidateScope(np.array([], dtype=np.int64)), 4)
            self.assertEqual(empty.supported_count, 0)
            self.assertEqual(len(empty.candidates.items), 0)

    def test_all_side_recipes_reload_and_share_candidate_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            events, basic, profiles = fixture()
            events.to_csv(raw / "log_standard_fixture.csv", index=False)
            prepare(raw, root / "data", users=0, train_end=10, val_end=15)
            _, val, encoding, _ = load_prepared(root / "data")
            metadata = MetadataEncoding.fit(encoding.users, encoding.items, basic, profiles)
            torch.save(
                {
                    "encoding": metadata.to_state(),
                    "data_manifest_sha256": sha256(root / "data/manifest.json"),
                    "manifest": {"assumptions": "fixture"},
                },
                root / "metadata.pt",
            )
            config = json.loads(
                (
                    Path(__file__).resolve().parents[1] / "configs/kuairand_side_study.json"
                ).read_text()
            )
            config["recipes"].append(
                {"name": "author_duration", "hidden": [64, 32], "features": "band_author_duration"}
            )
            (root / "recipes.json").write_text(json.dumps(config))
            result = run(
                argparse.Namespace(
                    data=root / "data",
                    out=root / "run",
                    seeds=[42],
                    epochs=2,
                    mf_epochs=1,
                    batch_size=4,
                    threads=1,
                    budget=2,
                    recipes=None,
                    selection_metric="roc_auc",
                    side_features=root / "metadata.pt",
                    include_merges=True,
                    config=root / "recipes.json",
                )
            )
            reference = {}
            for row in result["runs"]:
                directory = root / "run/seed42" / row["recipe"]
                loaded = load_ranker(directory / "inference.pt")
                predictions = pd.read_parquet(directory / "predictions.parquet")
                actual = loaded.predict(
                    loaded.features.build(val.user_id.to_numpy(), val.video_id.to_numpy())
                )
                np.testing.assert_array_equal(actual, predictions.logit.to_numpy())
                self.assertEqual(
                    set(row["funnel"]), {"mf", "popularity", "cooccurrence", "rr_all", "rrf_all"}
                )
                for channel in row["funnel"]:
                    ids = np.load(directory / f"{channel}_candidates.npz")["items"]
                    if channel not in reference:
                        reference[channel] = ids
                    else:
                        np.testing.assert_array_equal(reference[channel], ids)
                training = json.loads((directory / "config.json").read_text())
                self.assertEqual(
                    training["side_features_sha256"] is None, row["recipe"] == "reference"
                )
