"""A complete configured job, exact CPU continuation and changed-contract rejection."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable

import torch
from test_prepared_data import raw_fixture

from feed_ranker.artifacts import load_artifact
from feed_ranker.data.dataset import Batch
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.preprocessing.kuairand import prepare_kuairand
from feed_ranker.training.jobs import TrainingJobConfig, train_job
from feed_ranker.training.objectives import Objective, TaskLossConfig
from feed_ranker.training.optimizer import OptimizerConfig, build_optimizer
from feed_ranker.training.trainer import TrainConfig, Trainer


class ConstantEvaluator:
    def evaluate(self, model: torch.nn.Module, loader: Iterable[Batch]) -> dict[str, Any]:
        return {"long_view/roc_auc": 0.5, "rows": 2}


class TrainingJobTests(unittest.TestCase):
    def test_resume_matches_continuous_and_checks_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            prepare_kuairand(raw_fixture(root), data, users=0, train_end=10, val_end=15)

            def config(epochs: int, weight: float = 1) -> TrainingJobConfig:
                return TrainingJobConfig(
                    data,
                    DeepConfig(
                        dense=("train_duration_log1p",),
                        sparse={"user_code": 3, "item_code": 4},
                        hidden=(4,),
                        dropout=0.2,
                    ),
                    (TaskLossConfig("long_view", "long_view", task_weight=weight),),
                    TrainConfig(
                        epochs=epochs,
                        seed=42,
                        selection_metric="long_view/roc_auc",
                        selection_mode="max",
                    ),
                    batch_size=3,
                )

            train_job(config(1), root / "first")
            train_job(config(3), root / "resumed", resume_from=root / "first")
            train_job(config(3), root / "continuous")
            for name in ("best.pt", "last.pt"):
                a, b = [
                    torch.load(root / run / "checkpoints" / name, weights_only=True)
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
            exported = load_artifact(root / "resumed" / "model")
            best = torch.load(root / "resumed/checkpoints/best.pt", weights_only=True)
            self.assertNotIn("optimizer", best)
            self.assertNotIn("torch_rng", best)
            for name, weight in exported.model.state_dict().items():
                torch.testing.assert_close(weight, best["model"][name], rtol=0, atol=0)
            self.assertEqual(
                {p.name for p in (root / "resumed").iterdir()},
                {
                    "config.json",
                    "provenance.json",
                    "history.json",
                    "result.json",
                    "checkpoints",
                    "model",
                },
            )
            with self.assertRaisesRegex(ValueError, "resume contract"):
                train_job(config(3, weight=2), root / "invalid", resume_from=root / "first")

    def test_resume_rejects_mixed_checkpoints_and_keeps_earlier_best_across_resumes(self) -> None:
        batch = {
            "features": {"value": torch.tensor([-1.0, 1.0])},
            "targets": {"long_view": torch.tensor([0.0, 1.0])},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fit(
                name: str,
                epochs: int,
                seed: int = 42,
                contract: str = "a",
                source: str | None = None,
            ) -> None:
                torch.manual_seed(seed)
                model = DeepModel(DeepConfig(dense=("value",), hidden=(4,)))
                trainer = Trainer(
                    model,
                    build_optimizer(model, OptimizerConfig()),
                    Objective((TaskLossConfig("long_view", "long_view"),)),
                    TrainConfig(epochs=epochs, seed=seed),
                )
                trainer.fit(
                    [batch],
                    [batch],
                    ConstantEvaluator(),
                    checkpoint_dir=root / name / "checkpoints",
                    run_identity={"source": contract},
                    resume_from=None if source is None else root / source / "checkpoints",
                )

            fit("original", 1)
            fit("other_seed", 1, seed=43)
            fit("other_contract", 1, contract="b")
            original = (root / "original/checkpoints/best.pt").read_bytes()
            for foreign in ("other_seed", "other_contract"):
                shutil.copy2(
                    root / foreign / "checkpoints/best.pt", root / "original/checkpoints/best.pt"
                )
                with self.subTest(foreign=foreign):
                    with self.assertRaisesRegex(ValueError, "Best checkpoint.*contract"):
                        fit(f"rejected_{foreign}", 2, source="original")
            (root / "original/checkpoints/best.pt").write_bytes(original)
            # An unchanged best can retain an older epoch cap across multiple resumes.
            fit("second", 2, source="original")
            fit("third", 3, source="second")
            self.assertEqual((root / "third/checkpoints/best.pt").read_bytes(), original)
            mixed = torch.load(root / "third/checkpoints/last.pt", weights_only=True)
            mixed["val"]["rows"] = 999
            torch.save(mixed, root / "third/checkpoints/last.pt")
            with self.assertRaisesRegex(ValueError, "Last checkpoint.*history"):
                fit("bad_history", 4, source="third")

    def test_resume_rejects_changed_supplied_optimizer_before_restoring_state(self) -> None:
        batch = {
            "features": {"value": torch.tensor([-1.0, 1.0])},
            "targets": {"long_view": torch.tensor([0.0, 1.0])},
        }
        objective = Objective((TaskLossConfig("long_view", "long_view"),))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = DeepModel(DeepConfig(dense=("value",), hidden=(4,)))
            trainer = Trainer(
                model, torch.optim.SGD(model.parameters(), lr=0.1), objective, TrainConfig(epochs=1)
            )
            trainer.fit(
                [batch],
                [batch],
                ConstantEvaluator(),
                checkpoint_dir=root / "first/checkpoints",
                run_identity={"source": "fixture"},
            )
            for name in ("lr", "class", "order"):
                model = DeepModel(DeepConfig(dense=("value",), hidden=(4,)))
                parameters = list(model.parameters())
                if name == "order":
                    parameters.reverse()
                optimizer = (
                    torch.optim.Adam(parameters, lr=0.1)
                    if name == "class"
                    else torch.optim.SGD(parameters, lr=0.2 if name == "lr" else 0.1)
                )
                trainer = Trainer(model, optimizer, objective, TrainConfig(epochs=2))
                initial = {key: value.clone() for key, value in model.state_dict().items()}
                with self.subTest(change=name):
                    with self.assertRaisesRegex(ValueError, "contract mismatch"):
                        trainer.fit(
                            [batch],
                            [batch],
                            ConstantEvaluator(),
                            checkpoint_dir=root / name / "checkpoints",
                            run_identity={"source": "fixture"},
                            resume_from=root / "first/checkpoints",
                        )
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    def test_deep_job_rejects_missing_multiple_or_mismatched_objectives(self) -> None:
        for tasks in (
            (),
            (TaskLossConfig("long_view", "long_view"), TaskLossConfig("click", "is_click")),
            (TaskLossConfig("click", "is_click"),),
        ):
            with self.subTest(tasks=tasks):
                with self.assertRaisesRegex(ValueError, "one objective"):
                    TrainingJobConfig(
                        Path("unused"), DeepConfig(dense=("value",)), tasks, TrainConfig()
                    )


if __name__ == "__main__":
    unittest.main()
