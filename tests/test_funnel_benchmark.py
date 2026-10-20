"""Frozen Deep feature boundaries and a complete validation-only funnel fixture."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.recall import VERSION, RecallData
from feed_ranker.evaluation.kuairec import pair_codes
from feed_ranker.experiments.convert_inference import (
    FrozenDeepReference,
    convert_checkpoint,
    convert_retriever,
)
from feed_ranker.experiments.evaluate_funnel import run
from feed_ranker.experiments.ranking_recipe import Recipe, build_architecture, feature_config
from feed_ranker.features.kuairec import KuaiRecFeatureBuilder
from feed_ranker.inference.artifacts import load_ranker
from feed_ranker.inference.rankers import ModelRanker
from feed_ranker.models.recall import BPRMF
from feed_ranker.recommendation.contracts import Request


class DeepFunnelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.meta = {
            "protocol": "big-to-small-v1",
            "spec": {"crosses": {}},
            "sources": {"big_matrix.csv": "fixture"},
            "vocab": {
                "user_id": ["10", "20"],
                "video_id": ["100", "200", "300", "400"],
                "age_range": ["a", "b"],
                "follow_user_num_range": ["a", "b"],
                "duration_band": ["0–5s", "5–10s", "10–20s"],
            },
        }
        self.profiles = pd.DataFrame(
            {"raw_user_id": [10, 20], "age_range": [1, 2], "follow_user_num_range": [2, 1]}
        ).set_index("raw_user_id")
        self.features = KuaiRecFeatureBuilder(
            feature_config(self.meta, Recipe("deep", hidden=(4,))),
            self.meta["vocab"],
            self.profiles,
            pd.Series([5000.0, 10000.0, 15000.0, 20000.0], index=[100, 200, 300, 400]),
        )

    def test_provenance_identifies_source_checkout_from_another_working_directory(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                provenance = collect_run_provenance()
                self.assertEqual(provenance["commit"], expected)
        finally:
            os.chdir(previous)

    def test_features_ignore_labels_and_rederive_duration_band(self) -> None:
        raw = pd.DataFrame(
            {
                "user_id": [10, 20, 99],
                "video_id": [100, 300, 999],
                "video_duration": [5000.0, 5001.0, 20000.0],
                "watch_ratio": [999, 999, 999],
                "age_range": ["wrong"] * 3,
                "duration_band": ["wrong"] * 3,
            }
        )
        got = self.features.encode(raw)
        self.assertEqual(
            set(got),
            {
                "user_id",
                "video_id",
                "video_duration",
                "duration_band",
                "age_range",
                "follow_user_num_range",
            },
        )
        np.testing.assert_array_equal(got["duration_band"], [1, 2, 3])
        np.testing.assert_array_equal(got["age_range"], [1, 2, 0])
        np.testing.assert_array_equal(got["video_id"], [1, 3, 0])
        other = self.features.encode(raw.assign(watch_ratio=-100))
        for key in got:
            np.testing.assert_array_equal(got[key], other[key])
        with self.assertRaises(ValueError):
            self.features.encode(raw.assign(video_duration=0))

    def test_actual_ranker_scores_selected_items_with_frozen_features(self) -> None:
        features = KuaiRecFeatureBuilder(
            self.features.config,
            self.meta["vocab"],
            self.profiles,
            pd.Series([5000.0, 10000.0, 15000.0, 20000.0], index=[100, 200, 300, 400]),
        )
        recipe = Recipe("deep", hidden=(4,))
        model = build_architecture(feature_config(self.meta, recipe), recipe, 42)
        ranker = ModelRanker(model, features, "synthetic-checkpoint")
        all_items = np.array([100, 200, 300, 400])
        full = ranker.predict(features.build(np.full(4, 10), all_items))
        selected = np.array([400, 100])
        result = ranker.score(Request("q", 10), selected)
        np.testing.assert_array_equal(result.items, selected)
        np.testing.assert_allclose(result.scores, full[[3, 0]], rtol=1e-6, atol=1e-6)
        empty = ranker.score(Request("empty", 10), np.empty(0, dtype=int))
        self.assertEqual(len(empty.scores), 0)

    def test_conflicting_profiles_and_invalid_ids_fail(self) -> None:
        frame = self.profiles.reset_index()
        with self.assertRaises(ValueError):
            KuaiRecFeatureBuilder(
                self.features.config,
                self.meta["vocab"],
                pd.concat([frame, frame.assign(age_range=0)]).set_index("raw_user_id"),
                self.features.durations,
            )
        pairs = pd.DataFrame({"user_id": [0, 1, 1], "video_id": [1, 0, 1]})
        self.assertEqual(len(np.unique(pair_codes(pairs))), 3)
        with self.assertRaises(ValueError):
            pair_codes(pairs.assign(user_id=-1))

    def test_complete_frozen_funnel_and_checkpoint_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking, recall = root / "ranking", root / "recall"
            ranking.mkdir()
            recall.mkdir()
            train = pd.DataFrame(
                {
                    "user_id": [10, 10, 20, 20],
                    "video_id": [100, 200, 300, 400],
                    "outcome": [1, -1, 1, -1],
                    "watch_ratio": [1.0, 0.1, 1.0, 0.1],
                    "video_duration": [10000.0] * 4,
                }
            )
            val = train.assign(video_id=[300, 400, 100, 200])
            for name, raw in [("train", train), ("val", val)]:
                raw.to_parquet(recall / f"{name}.parquet", index=False)
                features = self.features.encode(raw)
                encoded = pd.DataFrame(features).assign(
                    raw_user_id=raw.user_id, raw_video_id=raw.video_id, watch_ratio=raw.watch_ratio
                )
                encoded.to_parquet(ranking / f"{name}.parquet", index=False)
            pd.DataFrame({"video_id": [100, 200, 300, 400], "median": [10000.0] * 4}).to_parquet(
                recall / "item_duration.parquet", index=False
            )
            self.meta["splits"] = {
                name: {"sha256": sha256(ranking / f"{name}.parquet")} for name in ("train", "val")
            }
            write_json(ranking / "meta.json", self.meta)
            write_json(
                recall / "meta.json",
                {
                    "version": VERSION,
                    "raw_big_sha256": "fixture",
                    "files": {
                        name: sha256(recall / name)
                        for name in ("train.parquet", "val.parquet", "item_duration.parquet")
                    },
                },
            )
            recipe = Recipe("deep", hidden=(4,))
            model = build_architecture(feature_config(self.meta, recipe), recipe, 42)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": 0,
                    "training": {"seed": 42, "selection_metric": "user_xauc"},
                    "metadata": {
                        "recipe": asdict(recipe),
                        "target_cap": 2,
                        "data_manifest_sha256": sha256(ranking / "meta.json"),
                    },
                },
                checkpoint,
            )
            trace = root / "trace.parquet"
            val[["user_id", "video_id"]].rename(
                columns={"user_id": "raw_user_id", "video_id": "raw_video_id"}
            ).assign(
                prediction=ModelRanker(model, self.features, "fixture").predict(
                    self.features.encode(val)
                )
            ).to_parquet(trace, index=False)
            runs = []
            for method in ("popularity", "cooccurrence", "mf"):
                directory = root / method
                directory.mkdir()
                np.save(directory / "scores.npy", [2.0, 1.0, 2.0, 1.0])
                write_json(
                    directory / "config.json",
                    {
                        "method": method,
                        "seed": 42,
                        "data_manifest_sha256": sha256(recall / "meta.json"),
                    },
                )
                runs.append(directory)
            mf = BPRMF(2, 4, 2)
            with torch.no_grad():
                mf.user.weight.fill_(1)
                mf.item.weight.copy_(torch.tensor([[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [0.5, 0.5]]))
                mf.item_updated.fill_(True)
            import json

            torch.save(
                {
                    "config": {**json.loads((runs[2] / "config.json").read_text()), "dimension": 2},
                    "model": mf.state_dict(),
                    "epoch": 0,
                },
                runs[2] / "best.pt",
            )
            config = torch.load(runs[2] / "best.pt", weights_only=True)["config"]
            write_json(runs[2] / "config.json", config)
            artifact = root / "inference/ranker"
            convert_checkpoint(checkpoint, ranking, recall / "item_duration.parquet", artifact)
            reference = FrozenDeepReference.read(checkpoint, sha256(checkpoint))
            parity = reference.verify_predictions(load_ranker(artifact), ranking, trace)
            self.assertEqual(parity["prediction_parity_max_abs"], 0)
            retrievers = []
            for directory in runs:
                target = root / "inference" / directory.name
                convert_retriever(RecallData(recall), directory, target)
                retrievers.append(target)
            # Neither evaluation mode opens old runs/checkpoints/reference traces.
            checkpoint.unlink()
            trace.unlink()
            for directory in runs:
                directory.rename(directory.with_name(directory.name + "-unavailable"))
            result = run(recall, ranking, artifact, retrievers, root / "out")
            actual = run(recall, ranking, artifact, retrievers, root / "model", execution="model")
            self.assertEqual(result["methods"], actual["methods"])
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(result["split_pair_membership"], "exact train and validation")
            packed = np.load(root / "out/mf_m20_candidates.npz")
            np.testing.assert_array_equal(packed["items"][:, :2], [[300, 400], [100, 200]])
            self.assertTrue((packed["items"][:, 2:] == -1).all())
            for policy in ("round_robin", "rrf", "anchored"):
                replay = run(
                    recall, ranking, artifact, retrievers, root / policy, merge_policy=policy
                )
                model = run(
                    recall,
                    ranking,
                    artifact,
                    retrievers,
                    root / f"{policy}-model",
                    execution="model",
                    merge_policy=policy,
                )
                self.assertEqual(replay["methods"], model["methods"])
                name = f"{policy}:cooccurrence+mf+popularity"
                self.assertEqual(replay["methods"]["mf"], result["methods"]["mf"])
                self.assertEqual(replay["methods"][name]["budgets"]["50"]["binary_recall"], 1)
                self.assertEqual(
                    replay["methods"][name]["composition"]["source_depth"],
                    "full_supported_scope",
                )
                cached = np.load(root / policy / f"{name}_m50_candidates.npz")
                executed = np.load(root / f"{policy}-model" / f"{name}_m50_candidates.npz")
                np.testing.assert_array_equal(cached["items"], executed["items"])
            with self.assertRaisesRegex(ValueError, "frozen finalist"):
                FrozenDeepReference.read(artifact / "state.pt", "wrong")
            # No test dataset exists: validation-only operation is an enforced fixture property.
            self.assertFalse((ranking / "test.parquet").exists())


if __name__ == "__main__":
    unittest.main()
