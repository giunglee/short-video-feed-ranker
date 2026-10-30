"""Independent module behavior, without the local experiment job."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_prepared_data import raw_fixture

from feed_ranker.artifacts import RankerArtifact, save_artifact
from feed_ranker.data.dataset import EventDataset, make_loader
from feed_ranker.evaluation.predictions import (
    PredictionEvaluator,
    Predictions,
    evaluate_predictions,
)
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.preprocessing.features import FeatureState
from feed_ranker.preprocessing.kuairand import prepare_kuairand
from feed_ranker.training.jobs import job_config
from feed_ranker.training.objectives import Objective, TaskLossConfig
from feed_ranker.training.trainer import TrainConfig, Trainer


class ModuleContractTests(unittest.TestCase):
    def test_trainer_consumes_instances_and_emits_queryable_metrics_without_etl(self) -> None:
        torch.manual_seed(42)
        model = DeepModel(DeepConfig(dense=("value",), hidden=(4,)))
        batch = {
            "features": {"value": torch.tensor([-2.0, -1.0, 1.0, 2.0])},
            "targets": {"long_view": torch.tensor([0.0, 0.0, 1.0, 1.0])},
            "metadata": {"user_id": torch.tensor([1, 2, 1, 2])},
        }
        evaluator = PredictionEvaluator({"long_view": "long_view"})
        records = []
        initial = {n: v.clone() for n, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model,
                TrainConfig(epochs=2),
                loss=Objective((TaskLossConfig("long_view", "long_view"),)),
            )
            result = trainer.fit(
                lambda epoch: [batch],
                lambda current: evaluator.evaluate(current, [batch]),
                Path(tmp) / "training",
                {"resume_contract": {"source": "in-memory-public-fixture"}},
                train_evaluate=lambda current: evaluator.evaluate(current, [batch]),
                record_metrics=lambda row: records.append(json.loads(json.dumps(row))),
            )
        self.assertEqual([r["epoch"] for r in records], [0, 1])
        self.assertTrue(any(not torch.equal(initial[n], v) for n, v in model.state_dict().items()))
        self.assertEqual(result["val"], evaluator.evaluate(model, [batch]))
        for record in records:
            self.assertEqual(record["train_rows"], 4)
            self.assertEqual(record["train_eval"], record["val"])
            self.assertEqual(record["tasks"]["long_view"]["denominator"], 4)

    def test_offline_prediction_records_need_no_model_and_validate_alignment(self) -> None:
        values = dict(
            labels=np.array([0.0, 1.0, np.nan]),
            logits=np.array([-1.0, 1.0, np.nan]),
            users=np.array([1, 1, 2]),
            warm=np.array([True, False, False]),
            valid=np.array([True, True, False]),
        )
        metrics = evaluate_predictions({"long_view": Predictions(**values)})
        self.assertEqual(metrics["long_view/roc_auc"], 1)
        self.assertEqual(metrics["long_view/rows"], 2)
        self.assertEqual(metrics["long_view/rows_total"], 3)
        with self.assertRaisesRegex(ValueError, "align"):
            Predictions(**{**values, "logits": np.zeros(2)})
        with self.assertRaisesRegex(ValueError, "finite"):
            Predictions(**{**values, "valid": np.ones(3, dtype=bool)})

    def test_repeated_validation_and_etl_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            tables = data.load_feature_state()
            dataset = EventDataset(
                data.read_split("validation"), ("user_code", "item_code"), ("long_view",)
            )
            loader = make_loader(dataset, 3)
            first, second = list(loader), list(loader)
            self.assertEqual([len(b["targets"]["long_view"]) for b in second], [3, 1])
            for a, b in zip(first, second, strict=True):
                torch.testing.assert_close(a["metadata"]["event_id"], b["metadata"]["event_id"])
            model = DeepModel(DeepConfig(sparse={"user_code": 999}))
            with self.assertRaisesRegex(ValueError, "disagree"):
                RankerArtifact(model, tables, "", {})
            with self.assertRaisesRegex(ValueError, "never targets"):
                job_config({"data": str(root / "data"), "model": {"features": ["long_view"]}})

    def test_model_input_configuration_cannot_drift_after_construction(self) -> None:
        sparse = {"item": 4}
        config = DeepConfig(sparse=sparse, hidden=(4,))
        model = DeepModel(config).eval()
        features = {"item": torch.tensor([1, 2])}
        expected = model(features)["long_view"]
        sparse["new_input"] = 3
        with self.assertRaises(TypeError):
            config.sparse["new_input"] = 3
        serialized = config.to_dict()
        serialized["sparse"]["new_input"] = 3
        self.assertEqual(config.feature_names, ("item",))
        torch.testing.assert_close(expected, model(features)["long_view"], rtol=0, atol=0)

    def test_model_inputs_resolve_from_manifest_without_feature_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_kuairand(raw_fixture(root), root / "data", users=0, train_end=10, val_end=15)
            with patch.object(FeatureState, "load", side_effect=AssertionError("state read")):
                config = job_config(
                    {
                        "data": str(root / "data"),
                        "model": {"features": ["item_code", "train_duration_log1p"]},
                        "tasks": [{"output": "long_view", "target": "long_view"}],
                        "training": {},
                    }
                )
            self.assertEqual(dict(config.model.sparse), {"item_code": 4})
            self.assertEqual(config.model.dense, ("train_duration_log1p",))

    def test_independent_loader_rejects_invalid_roles_ids_and_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            frame = data.read_split("train")
            features = ("user_code", "item_code")
            with self.assertRaisesRegex(ValueError, "separate"):
                EventDataset(frame, features, ("user_id",))
            for column in ("user_id", "video_id", "time_ms", "event_id"):
                for invalid in (1.9, float("nan"), -1):
                    broken = frame.copy()
                    broken[column] = broken[column].astype(float)
                    broken.loc[0, column] = invalid
                    with self.subTest(column=column, invalid=invalid):
                        with self.assertRaisesRegex(ValueError, "integer event metadata"):
                            EventDataset(broken, features, ("long_view",))
            for invalid in (2.0, float("nan")):
                broken = frame.copy()
                broken.loc[0, "long_view"] = invalid
                with self.assertRaisesRegex(ValueError, "binary targets"):
                    EventDataset(broken, features, ("long_view",))
            with self.assertRaisesRegex(ValueError, "unique"):
                EventDataset(frame.drop(columns="time_ms"), features, ("long_view",))
            with self.assertRaisesRegex(ValueError, "unique"):
                EventDataset(frame, features, ("long_view", "long_view"))

    def test_feature_order_survives_serialization_and_missing_inputs_fail(self) -> None:
        config = DeepConfig(dense=("duration",), sparse={"user": 3, "item": 4}, hidden=(4,))
        model = DeepModel(config).eval()
        features = {
            "user": torch.tensor([1]),
            "item": torch.tensor([2]),
            "duration": torch.tensor([0.5]),
        }
        reordered = DeepConfig(dense=("duration",), sparse={"item": 4, "user": 3}, hidden=(4,))
        restored = DeepModel(reordered).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            model(features)["long_view"], restored(features)["long_view"], rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "Missing model feature"):
            model({"user": features["user"]})

    def test_loading_and_inference_do_not_import_raw_preparation_or_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            tables = data.load_feature_state()
            model = DeepModel(DeepConfig(dense=("train_duration_log1p",), hidden=(4,)))
            save_artifact(root / "model", RankerArtifact(model, tables, "", {}))
            script = textwrap.dedent("""
                import importlib.abc
                import sys
                from pathlib import Path

                class BlockTrainingAndPreparation(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.startswith(("feed_ranker.training", "feed_ranker.preprocessing.kuairand")):
                            raise AssertionError(f"Unexpected inference dependency: {fullname}")
                        return None

                sys.meta_path.insert(0, BlockTrainingAndPreparation())
                from feed_ranker.data.parquet import ParquetReader
                from feed_ranker.data.dataset import EventDataset, make_loader
                from feed_ranker.recommendation.rankers import load_ranker
                from feed_ranker.recommendation.contracts import Request
                from feed_ranker.evaluation.predictions import PredictionEvaluator
                data = ParquetReader(Path("data"))
                ranker = load_ranker(Path("model"))
                events = data.read_split("validation")
                dataset = EventDataset(events, ranker.model.config.feature_names, ("long_view",))
                records = PredictionEvaluator({"long_view": "long_view"}).collect(ranker.model, make_loader(dataset, 2))
                result = ranker.score(Request("q", 1), events.video_id.unique())
                assert len(result.items) > 0
                assert len(records["long_view"].labels) == len(events)
            """)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
