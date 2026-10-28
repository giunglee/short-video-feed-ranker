"""Public synthetic behavior checks for strict temporal KuaiRand study setup."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from feed_ranker.data.instances import EventDataset
from feed_ranker.evaluation.metrics import binary_metrics, user_auc_metrics
from feed_ranker.models.recall import SparseCosineCooccurrence
from feed_ranker.preprocessing.features import fit_features
from feed_ranker.preprocessing.kuairand import temporal_events
from feed_ranker.training.bpr import ObservedSampler, histories


def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 3],
            "video_id": [10, 10, 20, 20, 30, 10, 40, 20, 20, 30, 50],
            "time_ms": [5, 6, 8, 5, 6, 10, 10, 11, 10, 11, 15],
            "duration_ms": [100, 200, 0, 0, 300, 99999, 88888, 2, 3, 4, 500],
            "long_view": [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "is_rand": [0] * 11,
            "tab": [1] * 11,
        }
    )


class KuaiRandStudyTests(unittest.TestCase):
    def test_user_auc_exposes_propensity_and_one_class_exclusions(self) -> None:
        users = np.array([1, 1, 1, 2, 2, 2, 3])
        labels = np.array([1, 1, 0, 1, 0, 0, 1])
        logits = np.array([2.0, 2.0, 2.0, -2.0, -2.0, -2.0, 0.0])
        result = user_auc_metrics(users, labels, logits)
        self.assertEqual(
            result,
            {
                "macro_user_auc": 0.5,
                "users_with_both_classes": 2,
                "users_all": 3,
            },
        )
        self.assertGreater(binary_metrics(labels, logits)["roc_auc"], 0.5)

    def test_sparse_exact_cosine_and_singleton_diagonal(self) -> None:
        h = np.array(
            [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=float
        )
        positive = [np.flatnonzero(row) for row in h]
        model = SparseCosineCooccurrence(positive, h.shape[1])
        norms = np.sqrt(np.maximum(h.sum(0), 1))
        sim = h.T @ h / norms[:, None] / norms[None, :]
        np.fill_diagonal(sim, 0)
        for u in range(len(h)):
            np.testing.assert_allclose(model.score(u, np.arange(5)), h[u] @ sim, atol=1e-12)
        single = SparseCosineCooccurrence([np.array([0]), np.array([0])], 2)
        np.testing.assert_array_equal(single.score(0, np.arange(2)), np.zeros(2))

    def test_large_sparse_vocabulary_and_invalid_histories(self) -> None:
        model = SparseCosineCooccurrence([np.array([0, 99999]), np.array([99999])], 100000)
        self.assertLess(model.storage_bytes, 4_000_000)
        np.testing.assert_allclose(
            model.score(0, np.array([0, 99999, 50])), [1 / np.sqrt(2), 1 / np.sqrt(2), 0]
        )
        with self.assertRaises(ValueError):
            SparseCosineCooccurrence([np.array([1, 1])], 3)

    def test_temporal_repeats_and_train_only_features_survive_retirement(self) -> None:
        source = events()
        train, val, duplicates = temporal_events(
            pd.concat([source, source.iloc[[0]]]), "long_view", 10, 15
        )
        self.assertEqual((duplicates, len(train), len(val)), (1, 5, 5))
        self.assertEqual(val.time_ms.eq(10).sum(), 3)
        state = fit_features(train)
        pos, neg = histories(train, state.users, state.items, "long_view")
        self.assertEqual(pos[0].tolist(), [0])
        self.assertEqual(neg[0].tolist(), [1])
        names = tuple(state.columns)
        a = EventDataset(val, state, names, ("long_view",))
        b = EventDataset(
            val.assign(duration_ms=999999, long_view=1 - val.long_view),
            state,
            names,
            ("long_view",),
        )
        for name in names:
            np.testing.assert_array_equal(
                a[list(range(len(a)))]["features"][name], b[list(range(len(b)))]["features"][name]
            )
        sampler = ObservedSampler(pos, neg)
        users, positive, negative = sampler.sample(np.random.default_rng(42))
        for u, p, n in zip(users.tolist(), positive.tolist(), negative.tolist(), strict=True):
            self.assertIn(p, pos[u])
            self.assertIn(n, neg[u])
            self.assertNotIn(n, pos[u])
        with self.assertRaisesRegex(ValueError, "No users"):
            ObservedSampler([np.array([0])], [np.array([], dtype=np.int64)])


if __name__ == "__main__":
    unittest.main()
