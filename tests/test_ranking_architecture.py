"""Observable architecture, target and study-isolation behavior on public fixtures."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import sha256
from feed_ranker.data.kuairec_preprocessor import KuaiRecPreprocessor
from feed_ranker.experiments.ranking_architecture import (
    Recipe,
    build_architecture,
    feature_config,
)
from feed_ranker.experiments.ranking_study import (
    RankingStudy,
    assert_finalists_unchanged,
    best_run,
    needs_longer_budget,
)
from feed_ranker.training.evaluation import RegressionEvaluator
from feed_ranker.training.losses import TargetMAE


class ArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.meta = {
            "vocab": {
                n: ["a", "b"]
                for n in (
                    "user_id",
                    "video_id",
                    "age_range",
                    "follow_user_num_range",
                    "duration_band",
                )
            },
            "spec": {
                "crosses": {
                    "main_age": {"age_range": "a"},
                    "age_duration": {"age_range": "a", "duration_band": "a"},
                    "follow_duration": {"follow_user_num_range": "a", "duration_band": "a"},
                }
            },
        }
        self.inputs = {n: torch.tensor([0, 1, 2]) for n in self.meta["vocab"]}
        self.inputs.update(
            video_duration=torch.tensor([1000.0, 2000.0, 3000.0]),
            age_duration=torch.tensor([0.0, 1.0, 0.0]),
            follow_duration=torch.tensor([0.0, 0.0, 1.0]),
        )

    def test_linear_is_additive_and_unknown_has_zero_coefficient(self) -> None:
        recipe = Recipe("linear")
        config = feature_config(self.meta, recipe)
        self.assertEqual(config.cross_feature_cols, [])
        model = build_architecture(config, recipe, 42)
        with torch.no_grad():
            model.linear.weight.zero_()
            model.linear.bias.fill_(0.5)
            model.main_effects.coefficients["user_id"].weight[1, 0] = 0.2
            model.main_effects.coefficients["video_id"].weight[2, 0] = 0.3
        torch.testing.assert_close(model(self.inputs).flatten(), torch.tensor([0.5, 0.7, 0.8]))

    def test_all_architectures_backward_and_cross_information(self) -> None:
        for architecture in ("linear", "wide", "deep", "wide_deep"):
            recipe = Recipe(architecture, hidden=(4,) if "deep" in architecture else ())
            config = feature_config(self.meta, recipe)
            self.assertEqual(
                len(config.cross_feature_cols), 2 if architecture in ("wide", "wide_deep") else 0
            )
            model = build_architecture(config, recipe, 42)
            predicted = model(self.inputs)
            self.assertEqual(predicted.shape, (3, 1))
            TargetMAE(2)(predicted.flatten(), torch.tensor([0.1, 1.0, 10.0])).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if architecture in ("deep", "wide_deep"):
                emb = model.embedding_encoder.embeddings["user_id"]
                self.assertTrue(emb.weight[0].eq(0).all())
                self.assertTrue(emb.weight.grad[0].eq(0).all())

    def test_cap_changes_targets_and_ties_without_clipping_predictions(self) -> None:
        labels = torch.tensor([1.0, 3.0, 10.0])
        predicted = torch.tensor([1.0, 4.0, 3.0], requires_grad=True)
        loss = TargetMAE(2)(predicted, labels)
        self.assertAlmostEqual(loss.item(), 1.0)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([0.0, 1 / 3, 1 / 3]))
        torch.testing.assert_close(labels, torch.tensor([1.0, 3.0, 10.0]))

        class Predictions(torch.nn.Module):
            def forward(self, features):
                return features["scores"]

        batch = {
            "feature": {"scores": predicted.detach()},
            "label": labels,
            "raw_user_id": torch.tensor([1, 1, 1]),
        }
        evaluation = RegressionEvaluator(
            {"val": [batch]}, ("mae", "xauc", "user_xauc"), "cpu", 0, 1000, 256, label_cap=2
        )
        result = evaluation.evaluate(Predictions(), "val")
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["xauc"], 1.0)
        self.assertEqual(result["user_xauc"], 1.0)

    def test_selection_and_integrity_guards(self) -> None:
        a = {"val": {"user_xauc": 0.7}, "run": "a"}
        b = {"val": {"user_xauc": 0.7}, "run": "b"}
        self.assertIs(best_run([a, b]), a)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "best.pt").write_bytes(b"checkpoint")
            frozen = {"runs": [{"run": str(root), "checkpoint_sha256": sha256(root / "best.pt")}]}
            assert_finalists_unchanged(frozen)
            (root / "best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                assert_finalists_unchanged(frozen)
            history = [{"val": {"user_xauc": 0.7 + i * 0.0001}} for i in range(120)]
            (root / "history.json").write_text(json.dumps(history))
            self.assertTrue(needs_longer_budget(root))
            history[-1]["val"]["user_xauc"] = 0.1
            (root / "history.json").write_text(json.dumps(history))
            self.assertFalse(needs_longer_budget(root))

    def test_synthetic_study_freezes_before_test_and_reports_all_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            big = pd.DataFrame(
                [
                    {
                        "user_id": u,
                        "video_id": v,
                        "watch_ratio": ((u + v) % 5) * 0.8,
                        "video_duration": 1000 + v * 2000,
                        "timestamp": u * 100 + v,
                    }
                    for u in range(6)
                    for v in range(12)
                ]
            )
            small = big[big.video_id.ge(9)].copy()
            big = big[big.video_id.lt(9)]
            for name, frame in (("big_matrix", big), ("small_matrix", small)):
                frame["play_duration"] = frame.watch_ratio * frame.video_duration
                frame.to_csv(raw / f"{name}.csv", index=False)
            pd.DataFrame(
                {
                    "user_id": range(6),
                    "age_range": ["a", "b"] * 3,
                    "follow_user_num_range": ["a", "b", "a"] * 2,
                }
            ).to_csv(raw / "user_features_raw.csv", index=False)
            spec = {
                "categorical": list(self.meta["vocab"]),
                "dense": ["video_duration"],
                "crosses": {
                    "age_duration": {"age_range": "a", "duration_band": "0–5s"},
                    "follow_duration": {"follow_user_num_range": "a", "duration_band": "0–5s"},
                },
            }
            data, output = root / "data", root / "study"
            KuaiRecPreprocessor(raw, data, spec, split_seed=2026, val_ratio=0.3).run()
            import feed_ranker.experiments.ranking_study as module

            original_score = module.score_checkpoint
            calls = []

            def guarded_score(data, run, destination, split, **kwargs):
                self.assertTrue((output / "finalists.json").exists())
                self.assertTrue((output / "recipe_freeze.json").exists())
                calls.append(split)
                return original_score(data, run, destination, split, **kwargs)

            with patch.object(module, "score_checkpoint", side_effect=guarded_score):
                RankingStudy(data, output, smoke=True).run()
            self.assertEqual(calls, ["val"] * 4 + ["test"] * 4)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "complete")
            self.assertEqual(status["completed_training_runs"], 21)
            result = pd.read_csv(output / "per_seed.csv")
            self.assertEqual(len(result), 8)
            self.assertTrue(np.isfinite(result.user_xauc).all())
            with self.assertRaises(FileExistsError):
                RankingStudy(data, output, smoke=True).run()


if __name__ == "__main__":
    unittest.main()
