"""A complete configured job, exact CPU continuation and changed-contract rejection."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from test_prepared_data import raw_fixture

from feed_ranker.data.prepared import prepare_kuairand
from feed_ranker.models.artifacts import load_artifact
from feed_ranker.models.deep import ModelConfig
from feed_ranker.training.job import TrainingJobConfig, train_job
from feed_ranker.training.objective import TaskLossConfig
from feed_ranker.training.trainer import TrainConfig


class TrainingJobTests(unittest.TestCase):
    def test_resume_matches_continuous_and_checks_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            prepare_kuairand(raw_fixture(root), data, users=0, train_end=10, val_end=15)

            def config(epochs: int, weight: float = 1) -> TrainingJobConfig:
                return TrainingJobConfig(
                    data,
                    ModelConfig(
                        ("user_code", "item_code", "train_duration_log1p"), hidden=(4,), dropout=0.2
                    ),
                    (TaskLossConfig("long_view", "long_view", task_weight=weight),),
                    TrainConfig(
                        epochs=epochs,
                        batch_size=3,
                        train_eval_rows=0,
                        seed=42,
                        loss="bce_logits",
                        metrics=("long_view/roc_auc", "long_view/logloss"),
                        selection_metric="long_view/roc_auc",
                        selection_mode="max",
                    ),
                )

            train_job(config(1), root / "first")
            train_job(config(3), root / "resumed", resume_from=root / "first")
            train_job(config(3), root / "continuous")
            for name in ("best.pt", "last.pt"):
                a, b = [
                    torch.load(root / run / name, weights_only=True)
                    for run in ("resumed", "continuous")
                ]
                self.assertEqual(a["epoch"], b["epoch"])
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            histories = [
                json.loads((root / run / "history.json").read_text())
                for run in ("resumed", "continuous")
            ]
            stripped = [
                [{k: v for k, v in row.items() if "seconds" not in k} for row in history]
                for history in histories
            ]
            self.assertEqual(*stripped)
            values = [r["val"]["long_view/roc_auc"] for r in histories[0]]
            result = json.loads((root / "resumed" / "result.json").read_text())
            self.assertEqual(result["best_epoch"], values.index(max(values)))
            load_artifact(root / "resumed" / "model")
            with self.assertRaisesRegex(ValueError, "resume_contract"):
                train_job(config(3, weight=2), root / "invalid", resume_from=root / "first")


if __name__ == "__main__":
    unittest.main()
