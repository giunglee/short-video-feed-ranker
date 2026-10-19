"""Feature-study preparation, bounded selection, and capped-training contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import sha256
from feed_ranker.data.feature_ablation import (
    BASE_COLUMNS,
    BASE_FEATURES,
    PROFILE_ADDITIONS,
    prepare_feature_data,
)
from feed_ranker.data.kuairec_preprocessor import KuaiRecPreprocessor
from feed_ranker.experiments.deep_features import (
    ARMS,
    RECIPE,
    fit_arm,
    make_dataset,
    reuse_baselines,
    select_candidates,
)
from feed_ranker.experiments.ranking_recipe import build_architecture
from feed_ranker.training.losses import TargetMAE
from feed_ranker.training.trainer import TrainConfig


class DeepFeaturesTest(unittest.TestCase):
    def test_selection_is_bounded_and_requires_positive_gain(self) -> None:
        rows = [{"arm": a, "seed": 42, "user_xauc": 0.7} for a in ARMS]
        self.assertEqual(select_candidates(rows), [])
        for row in rows[1:]:
            row["user_xauc"] = 0.71
        self.assertEqual(select_candidates(rows), ["no_age", "no_follow"])
        rows[-1]["user_xauc"] = 0.72
        self.assertEqual(select_candidates(rows), ["add_activity_profiles", "no_age"])
        with self.assertRaises(ValueError):
            select_candidates(rows[:-1])

    def test_reuse_rejects_different_objective_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "finalists.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {"seed": seed, "recipe": asdict(RECIPE), "run": str(root)}
                            for seed in range(42, 47)
                        ]
                    }
                )
            )
            saved = {
                "training": asdict(
                    TrainConfig(
                        epochs=120, seed=42, selection_metric="user_xauc", selection_mode="max"
                    )
                ),
                "metadata": {"target_cap": None, "recipe": asdict(RECIPE)},
            }
            with (
                patch("feed_ranker.experiments.deep_features.make_dataset") as make,
                patch("feed_ranker.experiments.deep_features.torch.load", return_value=saved),
            ):
                make.return_value.loader.return_value = [{"label": torch.zeros(1)}]
                with self.assertRaisesRegex(ValueError, "objective"):
                    reuse_baselines(root, root)
                (root / "meta.json").write_text("{}")
                saved["metadata"].update(
                    target_cap=2, data_manifest_sha256=sha256(root / "meta.json")
                )
                metrics = root / "evaluation/val/deep_42"
                metrics.mkdir(parents=True)
                (metrics / "metrics.json").write_text(json.dumps({"split": "test"}))
                with self.assertRaisesRegex(ValueError, "evaluation protocol"):
                    reuse_baselines(root, root)

    def test_preserved_rows_train_only_vocab_and_all_arm_backward(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, base, data = root / "raw", root / "base", root / "data"
            raw.mkdir()
            frame = pd.DataFrame(
                [
                    {
                        "user_id": u,
                        "video_id": v,
                        "watch_ratio": ((u + v) % 5) * 0.8,
                        "video_duration": 1000 + 2000 * v,
                        "timestamp": 100 * u + v,
                    }
                    for u in range(6)
                    for v in range(12)
                ]
            )
            frame["play_duration"] = frame.watch_ratio * frame.video_duration
            for name, selected in (
                ("big", frame[frame.video_id.lt(9)]),
                ("small", frame[frame.video_id.ge(9)]),
            ):
                selected.to_csv(raw / f"{name}_matrix.csv", index=False)
            profiles = pd.DataFrame(
                {
                    "user_id": [*range(6), 99],
                    "age_range": ["young", "older"] * 3 + ["unused"],
                    "follow_user_num_range": ["few", "many"] * 3 + ["unused"],
                }
            )
            for name in PROFILE_ADDITIONS:
                profiles[name] = ["a", "b"] * 3 + ["unseen"]
            profiles.to_csv(raw / "user_features_raw.csv", index=False)
            pd.DataFrame(
                {"video_id": [*range(12), 99], "feat": ["[1, 2]"] * 9 + ["[999]"] * 4}
            ).to_csv(raw / "item_categories.csv", index=False)
            KuaiRecPreprocessor(
                raw,
                base,
                {"categorical": list(BASE_FEATURES), "dense": ["video_duration"], "crosses": {}},
                split_seed=2026,
                val_ratio=0.3,
            ).run()
            # Runtime preparation and training must work with test physically absent.
            (base / "test.parquet").unlink()
            (raw / "small_matrix.csv").unlink()
            meta = prepare_feature_data(base, raw, data)
            self.assertEqual(set(meta["splits"]), {"train", "val"})
            self.assertFalse((data / "test.parquet").exists())
            self.assertEqual(meta["vocab"]["item_categories"], ["1", "2"])
            for name in PROFILE_ADDITIONS:
                self.assertEqual(meta["vocab"][name], ["a", "b"])
            for split in ("train", "val"):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(base / f"{split}.parquet", columns=BASE_COLUMNS),
                    pd.read_parquet(data / f"{split}.parquet", columns=BASE_COLUMNS),
                )
            for arm in ARMS:
                dataset = make_dataset(data, arm)
                batch = next(iter(dataset.loader("val", 100)))
                model = build_architecture(dataset.config.feature, RECIPE, 42)
                pred = model(batch["feature"]).flatten()
                TargetMAE(2)(pred, batch["label"]).backward()
                for emb in model.embedding_encoder.embeddings.values():
                    self.assertTrue(emb.weight[0].eq(0).all())
                    self.assertTrue(emb.weight.grad[0].eq(0).all())
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                )
            result = fit_arm(data, root / "run", "add_categories", 42, {}, epochs=2, steps=2)
            self.assertTrue(np.isfinite(result["val"]["user_xauc"]))
            config = json.loads((root / "run/config.json").read_text())
            self.assertEqual(config["target_cap"], 2)
            self.assertEqual(config["training"]["selection_metric"], "user_xauc")
            with self.assertRaises(FileExistsError):
                prepare_feature_data(base, raw, data)


if __name__ == "__main__":
    unittest.main()
