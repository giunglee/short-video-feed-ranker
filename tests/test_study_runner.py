"""Study contracts exercised through current data, training and scoring artifacts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from test_kuairand_metadata import fixture
from test_kuairand_workflow import configuration
from test_prepared_data import raw_fixture

from experiments.kuairand import run_study, summarize
from feed_ranker.preprocessing.kuairand import prepare_kuairand
from feed_ranker.training.jobs import job_config


class StudyRunnerTests(unittest.TestCase):
    def test_reuse_resume_predictions_and_seed_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            config = configuration(root / "data")
            model = config.pop("model")
            config.update(
                seeds=[42, 43],
                primary_seeds=[43],
                mf_epochs=1,
                budget=2,
                recipes=[
                    {"name": "reference", "model": model},
                    {"name": "replica", "model": model},
                ],
            )
            initial = run_study(config, root / "first")
            with patch("experiments.kuairand.fit_recall_job", side_effect=AssertionError("refit")):
                reused = run_study(config, root / "reused", recall_from=root / "first")
            for a, b in zip(initial["runs"], reused["runs"], strict=True):
                for key in ("candidate_cache_sha256", "validation", "pipeline", "best_epoch"):
                    self.assertEqual(a[key], b[key])
            for row in initial["summary"]:
                self.assertEqual(row["seeds"], [43] if row["group"] == "primary" else [42])
                self.assertTrue(all(v == 0 for v in row["paired_differences"] if v is not None))
            predictions = pd.read_parquet(root / "first/seed42/reference/predictions.parquet")
            np.testing.assert_array_equal(
                predictions.event_id, data.read_split("validation").event_id
            )
            self.assertTrue((root / "first/source/experiments/kuairand.py").exists())
            self.assertTrue((root / "first/source/feed_ranker/training/jobs.py").exists())
            continued = copy.deepcopy(config)
            continued["training"]["epochs"] = 2
            with patch("experiments.kuairand.fit_recall_job", side_effect=AssertionError("refit")):
                resumed = run_study(continued, root / "resumed", resume_from=root / "first")
                continuous = run_study(continued, root / "continuous", recall_from=root / "first")
            for a, b in zip(resumed["runs"], continuous["runs"], strict=True):
                self.assertEqual(a["validation"], b["validation"])
                self.assertEqual(a["pipeline"], b["pipeline"])
            for checkpoint in ("best.pt", "last.pt"):
                a, b = [
                    torch.load(root / folder / "seed42/reference" / checkpoint, weights_only=True)
                    for folder in ("resumed", "continuous")
                ]
                for name in a["model"]:
                    torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            changed = copy.deepcopy(continued)
            changed["budget"] = 3
            with self.assertRaisesRegex(ValueError, "epoch cap"):
                run_study(changed, root / "bad-resume", resume_from=root / "first")
            changed = copy.deepcopy(config)
            changed["mf_epochs"] = 2
            with self.assertRaisesRegex(ValueError, "MF budget"):
                run_study(changed, root / "bad-budget", recall_from=root / "first")
            (root / "first/seed42/candidates/mf.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed or are incomplete"):
                run_study(config, root / "bad-cache", recall_from=root / "first")

    def test_six_arm_configuration_executes_and_isolates_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = raw_fixture(root)
            _, basic, profiles = fixture()
            basic.to_csv(root / "basic.csv", index=False)
            profiles.to_csv(root / "profiles.csv", index=False)
            prepare_kuairand(
                raw,
                root / "data",
                users=0,
                train_end=10,
                val_end=15,
                basic=root / "basic.csv",
                profiles=root / "profiles.csv",
            )
            config = json.loads(Path("configs/kuairand_side_study.json").read_text())
            config.update(data=str(root / "data"), seeds=[42], primary_seeds=[42], mf_epochs=1)
            config["training"]["epochs"] = 1
            config["loading"]["batch_size"] = 3
            features = {r["name"]: set(r["model"]["features"]) for r in config["recipes"]}
            anonymous = {f"onehot_feat{i}" for i in range(18)}
            self.assertEqual(features["combined"] - features["author_duration"], anonymous)
            self.assertEqual(
                features["author_duration"] - features["basic_duration"], {"author_code"}
            )
            self.assertFalse(anonymous & features["reference"])
            result = run_study(config, root / "study")
            self.assertEqual(len(result["runs"]), 6)
            self.assertEqual(len({r["candidate_cache_sha256"] for r in result["runs"]}), 1)
            for channel in ("popularity", "cooccurrence", "mf", "rr_all", "rrf_all"):
                for metric in ("recall", "oracle_ndcg", "all_target_recall"):
                    self.assertEqual(
                        len({r["pipeline"][channel][metric] for r in result["runs"]}), 1
                    )
            self.assertTrue(
                any(
                    r["recipe"] == "combined" and r["reference"] == "author_duration"
                    for r in result["summary"]
                )
            )
            # The strict screen's seven arms also resolve against the current model interface.
            screen = json.loads(Path("configs/kuairand_study.json").read_text())
            for recipe in screen["recipes"]:
                resolved = job_config(
                    {**screen, "data": str(root / "data"), "model": recipe["model"]}
                )
                self.assertEqual(
                    set(resolved.model.feature_names), set(recipe["model"]["features"])
                )

    def test_summary_pairs_by_seed_and_excludes_undefined_values(self) -> None:
        config = {
            "seeds": [42, 43, 44],
            "primary_seeds": [43, 44],
            "recipes": [{"name": "reference"}, {"name": "candidate"}],
        }
        rows = []
        for seed, score in ((44, 0.7), (42, 0.99), (43, 0.5)):
            for name, delta in (("candidate", 0.1), ("reference", 0.0)):
                rows.append(
                    {
                        "seed": seed,
                        "recipe": name,
                        "validation": {
                            "long_view/roc_auc": score + delta,
                            "long_view/cold_item/roc_auc": None,
                        },
                        "pipeline": {
                            n: {
                                m: None
                                for m in ("ndcg", "recall", "oracle_ndcg", "all_target_recall")
                            }
                            for n in ("popularity", "cooccurrence", "mf", "rr_all", "rrf_all")
                        },
                    }
                )
        summary = summarize(rows, config)
        row = next(
            r
            for r in summary
            if r["group"] == "primary"
            and r["recipe"] == "candidate"
            and r["metric"] == "long_view/roc_auc"
        )
        self.assertAlmostEqual(row["mean"], 0.7)
        self.assertAlmostEqual(row["paired_mean"], 0.1)
        self.assertAlmostEqual(row["sample_sd"], np.sqrt(0.02))
        self.assertEqual(row["seeds"], [43, 44])
        undefined = next(r for r in summary if r["metric"] == "long_view/cold_item/roc_auc")
        self.assertEqual(undefined["defined_seeds"], 0)
        self.assertIsNone(undefined["paired_mean"])

    def test_invalid_study_fails_before_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for overrides in (
                {"primary_seeds": [99]},
                {"budget": 0},
                {"comparisons": [["missing", "reference"]]},
                {"tasks": [{"output": "is_click", "target": "is_click"}]},
            ):
                config = {
                    "data": str(root / "absent"),
                    "seeds": [42],
                    "mf_epochs": 1,
                    "recipes": [{"name": "reference", "model": {}}],
                    "tasks": [{"output": "long_view", "target": "long_view"}],
                    **overrides,
                }
                with self.assertRaises(ValueError):
                    run_study(config, root / "output")
                self.assertFalse((root / "output").exists())

    def test_invalid_optimizer_fails_before_recall_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_kuairand(raw_fixture(root), root / "data", users=0, train_end=10, val_end=15)
            config = configuration(root / "data")
            model = config.pop("model")
            config.update(seeds=[42], mf_epochs=1, recipes=[{"name": "reference", "model": model}])
            config["training"]["optimizer"] = {"lr": -0.1}
            with patch("experiments.kuairand.fit_recall_job", side_effect=AssertionError("refit")):
                with self.assertRaises(ValueError):
                    run_study(config, root / "study")
            self.assertFalse((root / "study").exists())


if __name__ == "__main__":
    unittest.main()
