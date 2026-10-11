"""Protocol and training behavior checks on synthetic public fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.kuairec_preprocessor import KuaiRecPreprocessor, split_pairs
from feed_ranker.experiments.multi_seed import summarize
from feed_ranker.experiments.wide_deep_kuairec import (
    build_model,
    build_trainer,
    evaluate_checkpoint,
    run_experiment,
)
from feed_ranker.models.constant import ConstantRegressor
from feed_ranker.training import OptimizerConfig, TrainConfig
from feed_ranker.training.metrics import user_xauc, xauc
from feed_ranker.training.optimizer import build_optimizer


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        rows = [
            {
                "user_id": u,
                "video_id": v,
                "watch_ratio": float((u + v) % 5) / 2,
                "video_duration": 1000 + v * 100,
                "timestamp": float(u * 10 + v),
                "date": 20200705,
            }
            for u in range(8)
            for v in range(8)
        ]
        self.big = pd.DataFrame(rows + rows[:3])
        self.small = pd.DataFrame(
            [
                {
                    "user_id": u,
                    "video_id": 8,
                    "watch_ratio": u / 3,
                    "video_duration": 1800,
                    "timestamp": np.nan,
                    "date": np.nan,
                }
                for u in range(9)
            ]
        )
        self.big["play_duration"] = self.big.watch_ratio * self.big.video_duration
        self.small["play_duration"] = self.small.watch_ratio * self.small.video_duration
        self.big.to_csv(self.raw / "big_matrix.csv", index=False)
        self.small.to_csv(self.raw / "small_matrix.csv", index=False)
        self.spec = {
            "categorical": ["user_id", "video_id"],
            "dense": ["video_duration"],
            "crosses": {},
        }
        self.data = self.root / "prepared"
        self.meta = KuaiRecPreprocessor(
            self.raw, self.data, self.spec, split_seed=5, val_ratio=0.25
        ).run()

    def test_signature_policy_keeps_distinct_times_and_missing_timestamps(self) -> None:
        from feed_ranker.data.kuairec_preprocessor import (
            deduplicate_signatures,
        )

        extra = self.big.iloc[[0]].copy()
        extra["timestamp"] = np.nan
        source = pd.concat([self.big, extra, extra], ignore_index=True)
        cleaned, removed = deduplicate_signatures(source)
        self.assertEqual(removed, 3)
        self.assertEqual(cleaned.timestamp.isna().sum(), 2)
        self.assertEqual(len(source), 69)

    def test_pair_weights_sum_to_one_and_train_without_categories(self) -> None:
        from feed_ranker.experiments.wide_deep_kuairec import build_model, build_trainer

        extra = self.big.iloc[[0]].copy()
        extra["timestamp"] += 10000
        pd.concat([self.big, extra]).to_csv(self.raw / "big_matrix.csv", index=False)
        output = self.root / "pair_data"
        KuaiRecPreprocessor(self.raw, output, self.spec, split_seed=5, val_ratio=0.25).run()
        for split in ("train", "val"):
            frame = pd.read_parquet(output / f"{split}.parquet")
            np.testing.assert_allclose(
                frame.groupby(["raw_user_id", "raw_video_id"]).pair_weight.sum(), 1
            )
        model, dataset = build_model(output, "wide_deep_nn", [4], 0.0, True)
        trainer = build_trainer(
            model, dataset, TrainConfig(epochs=1, batch_size=16, weighting="pair")
        )
        result = trainer.fit(self.root / "pair_run", {})
        self.assertTrue(np.isfinite(result["val"]["mae"]))

    def test_fixed_training_diagnostic_and_allowlisted_trace(self) -> None:
        from feed_ranker.experiments.diagnostics import activity_groups, export_validation_trace
        from feed_ranker.experiments.wide_deep_kuairec import build_model, build_trainer

        model, dataset = build_model(self.data, "wide_deep_nn", [4], 0.0, True)

        def identities(loader):
            return [
                (int(u), int(v))
                for batch in loader
                for u, v in zip(batch["raw_user_id"], batch["raw_video_id"])
            ]

        first = identities(dataset.fixed_train_loader(9, 23, 4))
        self.assertEqual(first, identities(dataset.fixed_train_loader(9, 23, 7)))
        self.assertEqual(len(first), 9)
        trainer = build_trainer(
            model,
            dataset,
            TrainConfig(epochs=1, batch_size=16, train_eval_rows=9, train_eval_seed=23),
        )
        run = self.root / "diagnostic_run"
        trainer.fit(run, {})
        history = json.loads((run / "history.json").read_text())
        self.assertEqual(history[0]["train_eval"]["rows"], 9)
        self.assertGreater(history[0]["training_seconds"], 0)
        output = self.root / "trace"
        export_validation_trace(
            model, dataset, run / "best.pt", output, feature_allowlist=["video_duration"]
        )
        trace = pd.read_parquet(output / "validation_trace.parquet")
        self.assertIn("feature__video_duration", trace)
        self.assertNotIn("feature__user_id", trace)
        self.assertEqual(len(trace), self.meta["splits"]["val"]["rows"])
        cohorts = pd.read_csv(output / "cohort_metrics.csv")
        activity = cohorts[cohorts.dimension.eq("train_activity")]
        self.assertEqual(activity.rows.sum(), len(trace))
        self.assertEqual(
            activity_groups(pd.Series([0, 1, 5, 10]), (2.0, 6.0)).tolist(),
            ["unseen", "low", "medium", "high"],
        )

    def test_cpu_continuation_matches_uninterrupted_training(self) -> None:
        def run(output: str, epochs: int, source: Path | None = None):
            return run_experiment(
                self.data,
                self.root / output,
                "wide_deep_nn",
                TrainConfig(epochs=epochs, batch_size=16, seed=17),
                [8, 4],
                dropout=0.2,
                resume_from=source,
            )

        run("full", 4)
        run("first", 2)
        run("continued", 4, self.root / "first")
        a = torch.load(self.root / "full/last.pt", weights_only=True)
        b = torch.load(self.root / "continued/last.pt", weights_only=True)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        self.assertEqual(a["val"], b["val"])
        first = torch.load(self.root / "full/best.pt", weights_only=True)
        resumed = torch.load(self.root / "continued/best.pt", weights_only=True)
        self.assertEqual(first["epoch"], resumed["epoch"])
        with self.assertRaisesRegex(ValueError, "larger epoch cap"):
            run("invalid_cap", 2, self.root / "first")
        with self.assertRaisesRegex(ValueError, "only the epoch cap"):
            run_experiment(
                self.data,
                self.root / "invalid_weight",
                "wide_deep_nn",
                TrainConfig(epochs=4, batch_size=16, seed=17, weighting="pair"),
                [8, 4],
                dropout=0.2,
                resume_from=self.root / "first",
            )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_preprocessor_cli(self) -> None:
        feature_file = self.root / "features.json"
        feature_file.write_text(json.dumps(self.spec))
        output = self.root / "cli-data"
        command = [
            sys.executable,
            "-m",
            "feed_ranker.data.kuairec_preprocessor",
            "--raw-dir",
            str(self.raw),
            "--features",
            str(feature_file),
            "--output",
            str(output),
            "--split-seed",
            "5",
            "--val-ratio",
            "0.25",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        metadata = json.loads((output / "meta.json").read_text())
        self.assertEqual(metadata["splits"], self.meta["splits"])

    def test_split_encodings_and_undated_test(self) -> None:
        train, val, test = [
            pd.read_parquet(self.data / f"{s}.parquet") for s in ("train", "val", "test")
        ]

        def pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
            return set(zip(df.raw_user_id, df.raw_video_id))

        self.assertFalse(pairs(train) & pairs(val))
        self.assertFalse((pairs(train) | pairs(val)) & pairs(test))
        self.assertEqual(len(train) + len(val), len(self.big) - 3)
        self.assertEqual(self.meta["deduplication"]["removed_rows"]["big_matrix"], 3)
        self.assertEqual(len(test), len(self.small))
        self.assertTrue(test.video_id.eq(0).all())
        self.assertEqual(test.loc[test.raw_user_id.eq(8), "user_id"].item(), 0)
        for raw in set(train.raw_user_id) & set(val.raw_user_id):
            self.assertEqual(
                train.loc[train.raw_user_id.eq(raw), "user_id"].iloc[0],
                val.loc[val.raw_user_id.eq(raw), "user_id"].iloc[0],
            )
        mask = split_pairs(self.big, 5, 0.25)
        shuffled = self.big.sample(frac=1, random_state=99)
        self.assertEqual(
            set(map(tuple, self.big.loc[mask, ["user_id", "video_id"]].values)),
            set(
                map(
                    tuple,
                    shuffled.loc[split_pairs(shuffled, 5, 0.25), ["user_id", "video_id"]].values,
                )
            ),
        )

    def test_overlap_and_leaky_features_rejected(self) -> None:
        self.big.iloc[:1].to_csv(self.raw / "small_matrix.csv", index=False)
        with self.assertRaisesRegex(ValueError, "overlap"):
            KuaiRecPreprocessor(self.raw, self.root / "bad", self.spec).run()
        bad = {**self.spec, "dense": ["watch_ratio"]}
        with self.assertRaisesRegex(ValueError, "Label/time"):
            KuaiRecPreprocessor(self.raw, self.root / "bad", bad).run()

    def test_reserved_cross_names_rejected(self) -> None:
        for name in ("watch_ratio", "user_id", "raw_video_id"):
            spec = {**self.spec, "crosses": {name: {"video_duration": 1000}}}
            with self.assertRaises(ValueError):
                KuaiRecPreprocessor(self.raw, self.root / "bad", spec).run()

    def test_feature_spec_validation(self) -> None:
        invalid_specs = [
            {"categorical": "user_id", "dense": ["video_duration"]},
            {"categorical": ["user_id"], "dense": [""]},
            {**self.spec, "unknown_option": True},
            {**self.spec, "crosses": {"cross": {}}},
            {**self.spec, "crosses": {"cross": {"user_id": float("nan")}}},
        ]
        for spec in invalid_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                KuaiRecPreprocessor(self.raw, self.root / "invalid", spec)

    def test_failed_write_leaves_no_partial_dataset(self) -> None:
        output = self.root / "interrupted"
        preprocessor = KuaiRecPreprocessor(self.raw, output, self.spec)
        original_write = pd.DataFrame.to_parquet

        def fail_after_train(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
            if path.name == "val.parquet":
                raise OSError("Simulated disk failure")
            original_write(frame, path, index=index)

        with patch.object(pd.DataFrame, "to_parquet", new=fail_after_train):
            with self.assertRaisesRegex(OSError, "Simulated disk failure"):
                preprocessor.run()
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".interrupted-*")), [])
        preprocessor.run()
        self.assertTrue((output / "meta.json").exists())
        with self.assertRaises(FileExistsError):
            preprocessor.run()

    def test_float32_overflow_rejected_without_output(self) -> None:
        self.small["video_duration"] = self.small["video_duration"].astype(float)
        self.small.loc[0, "video_duration"] = 1e100
        self.small.to_csv(self.raw / "small_matrix.csv", index=False)
        output = self.root / "overflow"
        with self.assertRaisesRegex(ValueError, "float32"):
            KuaiRecPreprocessor(self.raw, output, self.spec).run()
        self.assertFalse(output.exists())

    def test_matrix_only_features_do_not_read_user_tables(self) -> None:
        for filename in ("user_features_raw.csv", "user_features.csv"):
            (self.raw / filename).write_text('user_id,onehot_feat0\n"unterminated')
        output = self.root / "matrix-only"
        metadata = KuaiRecPreprocessor(self.raw, output, self.spec).run()
        self.assertEqual(set(metadata["sources"]), {"big_matrix.csv", "small_matrix.csv"})
        self.assertTrue((output / "test.parquet").exists())

    def test_profile_precedence_crosses_and_pair_membership(self) -> None:
        pd.DataFrame({"user_id": [2, 0, 1], "onehot_feat0": ["B", "A", "A"]}).to_csv(
            self.raw / "user_features_raw.csv", index=False
        )
        pd.DataFrame(
            {
                "user_id": [1, 0, 2],
                "onehot_feat0": ["fallback"] * 3,
                "profile_count": [10, 20, 30],
            }
        ).to_csv(self.raw / "user_features.csv", index=False)
        spec = {
            "categorical": ["user_onehot_feat0"],
            "dense": ["video_duration", "profile_count"],
            "crosses": {"group_a": {"user_onehot_feat0": "A"}},
        }
        output = self.root / "profiles"
        metadata = KuaiRecPreprocessor(self.raw, output, spec, 5, 0.25).run()
        self.assertEqual(metadata["vocab"]["user_onehot_feat0"], ["A", "B"])
        self.assertIn("user_features_raw.csv", metadata["sources"])
        self.assertIn("user_features.csv", metadata["sources"])
        for split in ("train", "val", "test"):
            frame = pd.read_parquet(output / f"{split}.parquet")
            original = pd.read_parquet(self.data / f"{split}.parquet")
            pd.testing.assert_frame_equal(
                frame[["raw_user_id", "raw_video_id"]],
                original[["raw_user_id", "raw_video_id"]],
            )
            self.assertTrue(frame.group_a.eq(frame.raw_user_id.isin([0, 1])).all())
            missing = frame.raw_user_id.ge(3)
            self.assertTrue(frame.loc[missing, "user_onehot_feat0"].eq(0).all())
            self.assertTrue(frame.loc[missing, "profile_count"].eq(0).all())

    def test_loader_covers_remainder_and_fixed_order(self) -> None:
        _, ds = build_model(self.data, "wide_deep_nn", [8], 0)
        batches = list(ds.loader("train", 7, shuffle_seed=2))
        self.assertEqual(
            sum(len(b["label"]) for b in batches), self.meta["splits"]["train"]["rows"]
        )
        first = torch.cat([b["label"] for b in batches])
        torch.rand(100)
        again = torch.cat([b["label"] for b in ds.loader("train", 7, shuffle_seed=2)])
        self.assertTrue(torch.equal(first, again))

    def test_train_never_reads_test_and_restores_checkpoint(self) -> None:
        (self.data / "test.parquet").rename(self.data / "held-out.parquet")
        cfg = TrainConfig(epochs=2, batch_size=7, eval_batch_size=5, xauc_pairs=200, user_pairs=30)
        first = run_experiment(self.data, self.root / "run1", "wide_deep_nn", cfg, [8])
        second = run_experiment(self.data, self.root / "run2", "wide_deep_nn", cfg, [8])
        self.assertEqual(first["val"], second["val"])
        checkpoint = torch.load(self.root / "run1/best.pt", weights_only=True)
        self.assertEqual(checkpoint["val"], first["val"])
        self.assertEqual(checkpoint["epoch"], first["best_epoch"])
        self.assertTrue((self.root / "run1/last.pt").exists())
        history = json.loads((self.root / "run1/history.json").read_text())
        self.assertEqual(len(history), 2)
        self.assertEqual(first["val"]["mae"], min(r["val"]["mae"] for r in history))
        (self.data / "held-out.parquet").rename(self.data / "test.parquet")
        result = evaluate_checkpoint(self.data, self.root / "run1/best.pt", self.root / "test.json")
        self.assertEqual(result["test"]["rows"], 9)
        with self.assertRaises(FileExistsError):
            evaluate_checkpoint(self.data, self.root / "run1/best.pt", self.root / "test.json")
        self.small.to_parquet(self.data / "test.parquet")
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate_checkpoint(self.data, self.root / "run1/best.pt", self.root / "other.json")

    def test_metrics(self) -> None:
        y = torch.tensor([0.0, 1.0, 1.0, 3.0])
        self.assertEqual(xauc(y, y, 1000), 1)
        self.assertEqual(xauc(-y, y, 1000), 0)
        self.assertEqual(xauc(torch.ones(4), y, 1000), 0.5)
        self.assertIsNone(xauc(y, torch.ones(4)))
        self.assertEqual(xauc(y, y, 1000, 8), xauc(y, y, 1000, 8))
        result = user_xauc(y, y, torch.tensor([0, 0, 1, 2]), 100)
        self.assertEqual(result, {"user_xauc": 1.0, "eligible_users": 1, "excluded_users": 2})
        with self.assertRaises(ValueError):
            xauc(torch.tensor([float("nan")]), torch.ones(1))

    def test_optimizer_and_models(self) -> None:
        for name in ("deep_nn", "wide_deep_nn", "wide_nn", "linear_reg"):
            model, ds = build_model(self.data, name, [8], 0)
            batch = next(iter(ds.loader("train", 7)))
            pred = model(batch["feature"])
            self.assertEqual(pred.shape, (7, 1))
            pred.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model, _ = build_model(self.data, "wide_deep_nn", [8], 0)
        opt = build_optimizer(model, OptimizerConfig(weight_decay=0.1))
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        emb = {
            id(p)
            for m in model.modules()
            if isinstance(m, torch.nn.Embedding)
            for p in m.parameters()
        }
        self.assertTrue(
            all(
                g["weight_decay"] == 0
                for g in opt.param_groups
                if any(id(p) in emb for p in g["params"])
            )
        )
        with self.assertRaises(ValueError):
            TrainConfig(epochs=0)

    def test_evaluation_disables_dropout_and_preserves_weights(self) -> None:
        model, dataset = build_model(self.data, "deep_nn", [8], dropout=0.5)
        trainer = build_trainer(model, dataset, TrainConfig(eval_batch_size=7, xauc_pairs=100))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        model.train()
        first = trainer.evaluator.evaluate(trainer.model, "val")
        second = trainer.evaluator.evaluate(trainer.model, "val")
        self.assertEqual(first, second)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_prediction_contract_rejects_extra_outputs(self) -> None:
        model, dataset = build_model(self.data, "deep_nn", [8], dropout=0)
        trainer = build_trainer(model, dataset, TrainConfig(eval_batch_size=7))
        with patch.object(model, "forward", return_value=torch.zeros(7, 2)):
            with self.assertRaisesRegex(ValueError, "Expected predictions"):
                trainer.evaluator.evaluate(trainer.model, "val")

    def test_invalid_runtime_configuration_and_metric_alignment(self) -> None:
        for options in ({"epochs": 1.5}, {"seed": -1}, {"metric_seed": 2**32}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainConfig(**options)
        values = torch.tensor([0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "aligned"):
            user_xauc(values, values, torch.tensor([0, 1]))
        _, dataset = build_model(self.data, "deep_nn", [8], dropout=0)
        with self.assertRaisesRegex(ValueError, "Only training"):
            list(dataset.loader("val", 7, shuffle_seed=42))

    def test_constant_models_fit_and_restore(self) -> None:
        labels = torch.tensor([0.0, 1.0, 3.0, 8.0])
        features = {"video_id": torch.arange(3)}
        for statistic, expected in (("mean", 3.0), ("median", 2.0)):
            model = ConstantRegressor.fit(labels, statistic)
            self.assertEqual(list(model.parameters()), [])
            self.assertTrue(torch.equal(model(features), torch.full((3, 1), expected)))
            restored = ConstantRegressor(0)
            restored.load_state_dict(model.state_dict())
            self.assertTrue(torch.equal(restored(features), model(features)))
        with self.assertRaises(ValueError):
            ConstantRegressor.fit(torch.tensor([]), "mean")
        with self.assertRaises(ValueError):
            ConstantRegressor.fit(labels, "min")

    def test_smooth_l1_training_loss(self) -> None:
        config = TrainConfig(epochs=1, steps=1, batch_size=7, loss="smooth_l1", metrics=("mae",))
        model, dataset = build_model(self.data, "deep_nn", [8], 0)
        batch = next(iter(dataset.loader("train", 7, shuffle_seed=config.seed)))
        with torch.no_grad():
            error = (model(batch["feature"]).reshape(-1) - batch["label"]).abs()
            expected = torch.where(error < 1, 0.5 * error.square(), error - 0.5).mean().item()
        trainer = build_trainer(model, dataset, config)
        directory = self.root / "smooth"
        trainer.fit(directory, {})
        history = json.loads((directory / "history.json").read_text())
        self.assertAlmostEqual(history[0]["train_loss"], expected)

    def test_mse_configuration_round_trips_through_checkpoint(self) -> None:
        config = TrainConfig(
            epochs=2,
            batch_size=7,
            loss="mse",
            metrics=("mse",),
            selection_metric="mse",
            selection_mode="min",
        )
        output = self.root / "mse"
        result = run_experiment(self.data, output, "deep_nn", config, [8])
        self.assertEqual(set(result["val"]), {"mse", "rows"})
        history = json.loads((output / "history.json").read_text())
        self.assertIn("train_loss", history[0])
        self.assertNotIn("train_online_mae", history[0])
        self.assertEqual(result["val"]["mse"], min(row["val"]["mse"] for row in history))
        (self.data / "train.parquet").unlink()
        test = evaluate_checkpoint(self.data, output / "best.pt", self.root / "mse-test.json")
        self.assertEqual(set(test["test"]), {"mse", "rows"})
        self.assertIn("baselines_test", test)

    def test_selection_max_keeps_first_tie_and_rejects_undefined(self) -> None:
        config = TrainConfig(
            epochs=3,
            batch_size=7,
            metrics=("xauc",),
            selection_metric="xauc",
            selection_mode="max",
            train_eval_rows=0,
        )
        model, dataset = build_model(self.data, "deep_nn", [8], 0)
        trainer = build_trainer(model, dataset, config)
        values = [{"xauc": score, "rows": 10} for score in (0.4, 0.7, 0.7)]
        with patch.object(trainer.evaluator, "evaluate", side_effect=values):
            result = trainer.fit(self.root / "max", {})
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["val"]["xauc"], 0.7)
        with patch.object(trainer.evaluator, "evaluate", return_value={"xauc": None}):
            with self.assertRaisesRegex(ValueError, "undefined"):
                trainer.fit(self.root / "undefined", {})

    def test_metric_configuration_and_raw_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            TrainConfig(metrics=("mse",))
        with self.assertRaises(ValueError):
            TrainConfig(loss="unknown")
        model, dataset = build_model(self.data, "deep_nn", [8], 0)
        config = TrainConfig(metrics=("mae", "mse"))
        trainer = build_trainer(model, dataset, config)
        measured = trainer.evaluator.evaluate(ConstantRegressor(0), "val")
        labels = pd.read_parquet(self.data / "val.parquet").watch_ratio.to_numpy(dtype=float)
        self.assertAlmostEqual(measured["mae"], float(abs(labels).mean()))
        self.assertAlmostEqual(measured["mse"], float((labels**2).mean()))
        batch = next(iter(dataset.loader("test", 20)))
        self.assertIn("raw_user_id", batch)
        self.assertIn("raw_video_id", batch)
        self.assertNotIn("user_id", batch)
        self.assertTrue(batch["feature"]["video_id"].eq(0).all())
        self.assertTrue(batch["raw_video_id"].eq(8).all())
        runs = [{"model": "deep_nn", "seed": 42, "val": {"mse": 2.0}}]
        summary = summarize(runs, ["deep_nn"], [42], metrics=("mse",))
        self.assertEqual(set(summary["models"]["deep_nn"]), {"mse"})

    def test_schema_rejects_invalid_required_observations(self) -> None:
        from feed_ranker.data.schema import KuaiRecInstance

        observation = KuaiRecInstance(1, 2, 0.5, 1000.0)
        self.assertIsNone(observation.gender)
        with self.assertRaises(ValueError):
            KuaiRecInstance(-1, 2, 0.5, 1000.0)
        with self.assertRaises(ValueError):
            KuaiRecInstance(1, 2, float("nan"), 1000.0)
        with self.assertRaises(TypeError):
            KuaiRecInstance(1, 2)

    def test_loader_does_not_consume_model_rng(self) -> None:
        _, dataset = build_model(self.data, "deep_nn", [4], 0.0)
        before = torch.get_rng_state().clone()
        batches = list(dataset.loader("val", 7))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(
            sum(len(batch["label"]) for batch in batches), self.meta["splits"]["val"]["rows"]
        )

    def test_full_protocol_resumes_completed_runs_and_rejects_changed_recipe(self) -> None:
        from feed_ranker.experiments.run_protocol import run_protocol

        for name in ("big", "small"):
            path = self.raw / f"{name}_matrix.csv"
            frame = pd.read_csv(path)
            frame["play_duration"] = frame.watch_ratio * frame.video_duration
            frame.to_csv(path, index=False)
        output = self.root / "full"
        config = TrainConfig(epochs=1, steps=1, batch_size=16, xauc_pairs=20, user_pairs=10)
        with patch("feed_ranker.experiments.run_protocol.SEEDS", [42]):
            run_protocol(self.raw, output, config, threads=1)
            self.assertEqual(json.loads((output / "status.json").read_text())["stage"], "complete")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["per_seed"]), 2)
            self.assertTrue((output / "wide_deep_nn-42/curves.png").exists())
            with patch(
                "feed_ranker.experiments.run_protocol.run_experiment",
                side_effect=AssertionError("must reuse completed runs"),
            ):
                run_protocol(self.raw, output, config, threads=1)
            with self.assertRaisesRegex(ValueError, "Recipe/source changed"):
                run_protocol(self.raw, output, TrainConfig(epochs=2), threads=1)

    def test_paired_summary(self) -> None:
        runs = [
            {"model": m, "seed": s, "val": {"mae": v, "xauc": 0.5, "user_xauc": None}}
            for m, s, v in [
                ("deep_nn", 42, 1.0),
                ("deep_nn", 43, 2.0),
                ("wide_deep_nn", 42, 0.9),
                ("wide_deep_nn", 43, 1.8),
            ]
        ]
        result = summarize(runs, ["deep_nn", "wide_deep_nn"], [42, 43])
        self.assertAlmostEqual(result["paired_deltas"]["wide_deep_nn"]["mae"]["mean"], -0.15)
        self.assertIsNone(result["models"]["deep_nn"]["user_xauc"]["mean"])
        with self.assertRaises(ValueError):
            summarize(runs[:-1], ["deep_nn", "wide_deep_nn"], [42, 43])


if __name__ == "__main__":
    unittest.main()
