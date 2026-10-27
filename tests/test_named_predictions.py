"""Behavior at the named feature/prediction/supervision interfaces."""

import unittest

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.prepared import FieldSpec, fit_features
from feed_ranker.evaluation.predictions import evaluate_predictions
from feed_ranker.models.deep import ModelConfig, build_model
from feed_ranker.training.objective import Objective, TaskLossConfig


class NamedPredictionTests(unittest.TestCase):
    def test_model_ignores_targets_and_rejects_target_allowlist(self) -> None:
        state = fit_features(
            pd.DataFrame({"user_id": [1, 1], "video_id": [10, 20], "duration_ms": [100, 200]})
        )
        config = ModelConfig(("user_code", "item_code", "train_duration_log1p"), hidden=(4,))
        schema = {**state.schema, "long_view": FieldSpec("long_view", "target", "float32")}
        model = build_model(config, schema).eval()
        features = state.gather(np.array([1, 1]), np.array([10, 20]))
        expected = model({"features": features})["long_view"]
        actual = model(
            {"features": features, "targets": {"long_view": torch.full((2,), float("nan"))}}
        )["long_view"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (2,))
        with self.assertRaisesRegex(ValueError, "never targets"):
            build_model(ModelConfig(("long_view",)), schema)
        with self.assertRaisesRegex(ValueError, "aligned"):
            model({"features": {**features, "user_code": features["user_code"][:, None]}})

    def test_weighted_tasks_mask_before_loss_and_do_not_renormalize(self) -> None:
        a = torch.tensor([0.0, 2.0, float("nan")], requires_grad=True)
        b = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
        objective = Objective(
            (
                TaskLossConfig("a", "a", mask="valid", weight="weights", task_weight=2),
                TaskLossConfig("b", "b", mask="second"),
            )
        )
        batch = {
            "targets": {
                "a": torch.tensor([1.0, 0.0, float("nan")]),
                "b": torch.tensor([1.0, 0.0, 1.0]),
            },
            "target_masks": {
                "valid": torch.tensor([True, True, False]),
                "second": torch.tensor([False, False, False]),
            },
            "target_weights": {"weights": torch.tensor([1.0, 3.0, 0.0])},
        }
        loss, diagnostics = objective({"a": a, "b": b}, batch)
        expected = (
            (torch.nn.functional.softplus(-a[0]) + 3 * torch.nn.functional.softplus(a[1])) / 4 * 2
        )
        torch.testing.assert_close(loss, expected)
        self.assertEqual(diagnostics["a"]["denominator"], 4)
        self.assertEqual(diagnostics["a"]["valid_count"], 2)
        self.assertEqual(diagnostics["b"]["denominator"], 0)
        loss.backward()
        self.assertEqual(float(a.grad[2]), 0)
        self.assertTrue(torch.isfinite(a.grad).all())
        batch["target_masks"]["second"][0] = True
        loss, _ = objective({"a": a, "b": b}, batch)
        loss.backward()
        self.assertNotEqual(float(b.grad[0]), 0)
        batch["target_masks"]["valid"][:] = False
        batch["target_masks"]["second"][:] = False
        with self.assertRaisesRegex(ValueError, "inactive"):
            objective({"a": a, "b": b}, batch)
        batch["target_weights"]["weights"][0] = -1
        with self.assertRaisesRegex(ValueError, "weights"):
            objective({"a": a, "b": b}, batch)

    def test_metrics_use_complete_predictions_and_expose_exclusions(self) -> None:
        class Fixed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(()))

            def forward(self, batch):
                return {"score": batch["features"]["score"] + self.offset}

        def batch(labels, logits, users, warm, valid):
            return {
                "features": {"score": torch.tensor(logits)},
                "targets": {"label": torch.tensor(labels)},
                "target_masks": {"valid": torch.tensor(valid)},
                "metadata": {"user_id": torch.tensor(users), "warm_item": torch.tensor(warm)},
            }

        batches = [
            batch([0.0, 0.0], [-2.0, -1.0], [1, 2], [True, True], [True, True]),
            batch(
                [1.0, 1.0, float("nan")],
                [1.0, 2.0, 0.0],
                [1, 2, 3],
                [False, False, False],
                [True, True, False],
            ),
        ]
        result = evaluate_predictions(Fixed(), batches, {"score": "label"}, {"score": "valid"})
        self.assertEqual(result["score/roc_auc"], 1)
        self.assertEqual(result["score/macro_user_auc"], 1)
        self.assertEqual(result["score/rows_total"], 5)
        self.assertEqual(result["score/rows"], 4)
        self.assertIsNone(result["score/warm_item/roc_auc"])
        self.assertEqual(result["score/warm_item/users_with_both_classes"], 0)


if __name__ == "__main__":
    unittest.main()
