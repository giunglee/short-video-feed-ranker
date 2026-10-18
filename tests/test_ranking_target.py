"""Behavior checks for target capping and full-pool checkpoint selection."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feed_ranker.training.funnel_evaluation import GainPool, TargetMAE
from feed_ranker.training.trainer import TrainConfig, Trainer


class RankingTargetTest(unittest.TestCase):
    def pool(self, ratios, k=10):
        return GainPool(
            np.array([7]),
            np.arange(len(ratios)),
            np.array([0, len(ratios)]),
            np.array(ratios, dtype=float),
            np.full(len(ratios), 1000.0),
            k=k,
        )

    def test_cap_keeps_tail_rows_and_raw_labels(self):
        labels = torch.tensor([0.5, 2.0, 10.0])
        before = labels.clone()
        predicted = torch.tensor([0.0, 1.0, 3.0], requires_grad=True)
        loss = TargetMAE(2)(predicted, labels)
        self.assertAlmostEqual(loss.item(), 2.5 / 3, places=6)
        loss.backward()
        torch.testing.assert_close(labels, before)
        torch.testing.assert_close(predicted.grad, torch.tensor([-1.0, -1.0, 1.0]) / 3)
        self.assertAlmostEqual(TargetMAE()(predicted, labels).item(), 8.5 / 3, places=6)
        for bad in [float("nan"), -1.0]:
            with self.assertRaises(ValueError):
                TargetMAE(2)(torch.zeros(1), torch.tensor([bad]))

    def test_handoff_example_and_full_pool_denominator(self):
        pool = self.pool([3, 1.5, 1.2, 0.8], k=3)
        predictions = np.array([3.0, 0.0, 2.0, 4.0])  # D A C B
        report, _ = pool.evaluate(predictions)
        expected = (0.8 + 2 / np.log2(3) + 1.2 / 2) / (2 + 1.5 / np.log2(3) + 1.2 / 2)
        self.assertAlmostEqual(report["cap2_ndcg10"], expected)
        subset, _ = pool.evaluate(predictions, [np.array([3, 0, 2])])
        self.assertAlmostEqual(subset["cap2_ndcg10"], expected)
        self.assertLess(subset["candidate_oracle_ndcg10"], 1)
        np.testing.assert_array_equal(pool.gains, [2, 1.5, 1.2, 0.8])

    def test_zero_ideal_shortfalls_and_invalid_pools(self):
        zero, _ = self.pool([0.0, 0.0]).evaluate(np.array([1.0, 2.0]))
        self.assertIsNone(zero["cap2_ndcg10"])
        self.assertEqual(zero["users_zero_idcg"], 1)
        short, _ = self.pool([2.0, 1.0]).evaluate(np.array([2.0, 1.0]), [np.array([0])])
        self.assertAlmostEqual(short["utility_per_slot10_all_users"], 0.2)
        self.assertLess(short["cap2_ndcg10"], 1)
        for values in [[-1.0], [np.nan]]:
            with self.assertRaises(ValueError):
                self.pool(values)
        with self.assertRaises(ValueError):
            self.pool([1, 2]).evaluate(np.array([0.0, 1.0]), [np.array([0, 0])])

    def test_ties_and_candidates_do_not_depend_on_labels(self):
        a, b = self.pool([1, 2, 3]), self.pool([3, 1, 2])
        scores = np.array([0.0, 0.0, -np.inf])
        np.testing.assert_array_equal(a.candidates(scores, 2)[0], b.candidates(scores, 2)[0])
        self.assertEqual(len(a.candidates(scores, 10)[0]), 2)

    def test_custom_selection_keeps_earliest_tie_and_standard_checkpoints(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, features):
                return self.weight * features["x"]

        class Evaluator:
            def __init__(self):
                self.scores = iter([0.4, 0.7, 0.7])

            def evaluate(self, model, split):
                return {"cap2_ndcg10": next(self.scores)}

        batch = {"feature": {"x": torch.ones(2)}, "label": torch.tensor([1.0, 10.0])}
        config = TrainConfig(
            epochs=3,
            metrics=("cap2_ndcg10",),
            selection_metric="cap2_ndcg10",
            selection_mode="max",
            train_eval_rows=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            trainer = Trainer(
                Model(), lambda epoch: [batch], config, loss_fn=TargetMAE(2), evaluator=Evaluator()
            )
            result = trainer.fit(path, {"target_cap": 2})
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(len(json.loads((path / "history.json").read_text())), 3)
            saved = torch.load(path / "best.pt", weights_only=True)
            self.assertEqual(saved["metadata"]["target_cap"], 2)
            self.assertIn("optimizer", saved)


if __name__ == "__main__":
    unittest.main()
