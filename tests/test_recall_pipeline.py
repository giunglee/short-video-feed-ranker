"""Public tiny fixtures for first-event recall, BPR, and shared-pool metrics."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.recall import Pools, RecallData, classify, first_events, prepare
from feed_ranker.evaluation.retrieval import evaluate, ordered_positions
from feed_ranker.experiments.recall_kuairec import run
from feed_ranker.models.recall import BPRMF
from feed_ranker.training.recall import ObservedSampler, fit_baseline


class RecallTests(unittest.TestCase):
    def test_neutral_policy_changes_only_training_comparison_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            train = pd.DataFrame(
                {
                    "user_id": [0, 0, 0, 1, 1, 1, 2, 2],
                    "video_id": [0, 1, 2, 2, 1, 3, 0, 2],
                    "outcome": [1, -1, 0, 1, -1, 0, 1, 0],
                }
            )
            val = pd.DataFrame({"user_id": [0, 1, 2], "video_id": [3, 0, 1], "outcome": [1, 1, 1]})
            for name, frame in [("train", train), ("val", val)]:
                frame.to_parquet(directory / f"{name}.parquet", index=False)
            write_json(
                directory / "meta.json",
                {
                    "version": "recall-first-watch-v1",
                    "files": {
                        f"{s}.parquet": sha256(directory / f"{s}.parquet") for s in ["train", "val"]
                    },
                },
            )
            data = RecallData(directory)
            clear = ObservedSampler(data.training)
            mixed = ObservedSampler(data.training, "negative_and_neutral")
            observed = {0: set(), 1: set()}
            for seed in range(50):
                u, p, n = mixed.sample(np.random.default_rng(seed))
                cu, cp, cn = clear.sample(np.random.default_rng(seed))
                torch.testing.assert_close(u, cu)
                torch.testing.assert_close(p, cp)
                self.assertEqual(set(cn.tolist()), {1})
                self.assertEqual(mixed.last_neutral_draws, int((n != 1).sum()))
                for user, item in zip(u.tolist(), n.tolist(), strict=True):
                    observed[user].add(item)
            self.assertEqual(observed, {0: {1, 2}, 1: {1, 3}})
            self.assertNotIn(2, mixed.users)
            with self.assertRaisesRegex(ValueError, "Unknown"):
                ObservedSampler(data.training, "unobserved")

    def test_mf_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            frames = {
                "train.parquet": pd.DataFrame(
                    {"user_id": [0, 0, 1, 1], "video_id": [0, 1, 2, 1], "outcome": [1, -1, 1, -1]}
                ),
                "val.parquet": pd.DataFrame(
                    {"user_id": [0, 1], "video_id": [2, 0], "outcome": [1, 1]}
                ),
            }
            for name, frame in frames.items():
                frame.to_parquet(data / name, index=False)
            write_json(
                data / "meta.json",
                {
                    "version": "recall-first-watch-v1",
                    "files": {name: sha256(data / name) for name in frames},
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run(data, root / "resumed", "mf", 1, False)
                run(data, root / "resumed", "mf", 2, True)
                run(data, root / "whole", "mf", 2, False)
            resumed = torch.load(root / "resumed" / "last.pt", weights_only=True)
            whole = torch.load(root / "whole" / "last.pt", weights_only=True)
            for name in whole["model"]:
                torch.testing.assert_close(
                    whole["model"][name], resumed["model"][name], rtol=0, atol=0
                )
            self.assertEqual(resumed["best_epoch"], 1)

    def test_no_supported_training_users_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "both positive and observed-negative"):
            ObservedSampler(SimpleNamespace(supported=np.array([False])))

    def test_class_boundaries(self) -> None:
        d = np.array([120000.0] * 5)
        p = np.sqrt(10 / 120)
        np.testing.assert_array_equal(
            classify(d, np.array([0.0, p / 2, p - 0.0001, p, 2.0])), [-1, 0, 0, 1, 1]
        )
        durations = np.geomspace(1, 1e8, 500)
        p = np.minimum(1, np.sqrt(10000 / durations))
        self.assertTrue((np.minimum(0.3, 0.5 * p) < p).all())
        with self.assertRaises(ValueError):
            classify(np.array([0]), np.array([0.1]))

    def test_first_event_and_missing_tie_policy(self) -> None:
        frame = pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 2, 2],
                "video_id": [3] * 6,
                "timestamp": [2.0, 1.0, np.nan, 0.0, 1.0, 1.0],
                "source_row": [0, 1, 2, 3, 4, 5],
                "video_duration": [10000] * 6,
                "watch_ratio": [0.0, 1.0, 0.5, 2.0, 0.1, 1.0],
            }
        )
        first, audit = first_events(frame.sample(frac=1, random_state=2))
        self.assertEqual(first.source_row.tolist(), [1, 2, 4])
        self.assertEqual(first.outcome.tolist(), [1, 0, -1])
        self.assertEqual(audit, {"missing_time_pairs": 1, "tied_first_time_pairs": 1})

    def test_shared_denominator_shortfall_and_empty_users(self) -> None:
        data = SimpleNamespace(
            users=np.array([1, 2]),
            items=np.array([10, 11, 12]),
            positive=[np.array([0]), np.array([])],
            negative=[np.array([1]), np.array([])],
            seen=[np.array([0]), np.array([])],
            supported=np.array([True, False]),
            pools=Pools(
                np.array([0, 0, 0, 1]),
                np.array([0, 1, 2, 0]),
                np.array([True, True, False, False]),
                np.array([0, 3, 4]),
            ),
        )
        data.training = SimpleNamespace(**vars(data))
        report, rows, _ = evaluate(data, np.array([1.0, -np.inf, -np.inf, -np.inf]))
        self.assertEqual(report["users_quality"], 1)
        self.assertEqual(report["metrics"]["recall@50"], 0.5)
        self.assertAlmostEqual(report["metrics"]["ndcg@10_m50"], 1 / (1 + 1 / np.log2(3)))
        self.assertTrue(rows["method_short@50"].all())
        full, _, _ = evaluate(data, np.array([2.0, 1.0, 0.0, 0.0]))
        self.assertEqual(full["metrics"]["recall@50"], 1.0)
        self.assertEqual(full["metrics"]["ndcg@10_m50"], 1.0)

    def test_ranker_cannot_change_candidate_oracle(self) -> None:
        data = SimpleNamespace(
            users=np.array([1]),
            items=np.array([10, 11]),
            positive=[np.array([0])],
            negative=[np.array([1])],
            seen=[np.array([0])],
            supported=np.array([True]),
            pools=Pools(
                np.array([0, 0]), np.array([0, 1]), np.array([True, False]), np.array([0, 2])
            ),
        )
        data.training = SimpleNamespace(**vars(data))
        result, _, _ = evaluate(data, np.array([2.0, 1.0]), np.array([0.0, 1.0]))
        self.assertEqual(result["metrics"]["oracle_ndcg@10_m50"], 1.0)
        self.assertAlmostEqual(result["metrics"]["ndcg@10_m50"], 1 / np.log2(3))

    def test_ties_stable_under_input_permutation(self) -> None:
        ids = np.array([11, 12, 13, 14])
        original = ids[ordered_positions(np.ones(4), 3, ids)]
        shuffled = ids[[2, 0, 3, 1]]
        np.testing.assert_array_equal(
            original, shuffled[ordered_positions(np.ones(4), 3, shuffled)]
        )
        with self.assertRaises(ValueError):
            ordered_positions(np.array([np.nan]), 1, np.array([1]))

    def test_sampler_and_bpr_learn_without_false_negatives(self) -> None:
        data = SimpleNamespace(
            users=np.array([0, 1]),
            items=np.array([0, 1, 2]),
            supported=np.array([True, False]),
            positive=[np.array([0]), np.array([2])],
            negative=[np.array([1]), np.array([], dtype=np.int64)],
        )
        sampler = ObservedSampler(data)
        u, p, n = sampler.sample(np.random.default_rng(42))
        self.assertEqual((u.tolist(), p.tolist(), n.tolist()), ([0], [0], [1]))
        torch.manual_seed(42)
        model = BPRMF(2, 3, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        start = float(model(u, p, n).detach())
        for _ in range(30):
            optimizer.zero_grad()
            loss = torch.nn.functional.softplus(-model(u, p, n)).mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(float(model(u, p, n).detach()), start + 1)

    def test_prepare_reuses_split_and_never_needs_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking = root / "ranking"
            ranking.mkdir()
            raw = pd.DataFrame(
                {
                    "user_id": [0, 0, 0, 0, 1, 1],
                    "video_id": [0, 0, 1, 2, 0, 1],
                    "timestamp": [1.0, 1.0, 2.0, 3.0, 1.0, 2.0],
                    "play_duration": [10000, 10000, 100, 5000, 10000, 100],
                    "video_duration": [10000] * 6,
                    "watch_ratio": [1.0, 1.0, 0.01, 0.5, 1.0, 0.01],
                }
            )
            raw.to_csv(root / "big.csv", index=False)
            train = pd.DataFrame({"raw_user_id": [0, 0, 1, 1], "raw_video_id": [0, 1, 0, 1]})
            val = pd.DataFrame({"raw_user_id": [0], "raw_video_id": [2]})
            for split, frame in [("train", train), ("val", val)]:
                frame.to_parquet(ranking / f"{split}.parquet", index=False)
            write_json(
                ranking / "meta.json",
                {
                    "preprocessing_version": "signature-dedup-v2",
                    "split_seed": 2026,
                    "sources": {"big_matrix.csv": sha256(root / "big.csv")},
                    "splits": {
                        s: {"sha256": sha256(ranking / f"{s}.parquet")} for s in ["train", "val"]
                    },
                },
            )
            prepare(root / "big.csv", ranking, root / "recall")
            data = RecallData(root / "recall")
            self.assertEqual(data.audit["excluded_validation_pairs"], 1)
            self.assertEqual(data.audit["mf_supported_users"], 2)
            self.assertEqual(data.meta["removed_duplicates"], 1)
            self.assertFalse((root / "recall" / "test.parquet").exists())

    def test_cosine_uses_full_history_without_popularity_backfill(self) -> None:
        data = SimpleNamespace(
            users=np.array([0, 1, 2]),
            items=np.array([0, 1, 2, 3]),
            positive=[np.array([0]), np.array([0, 1]), np.array([0, 2])],
            pools=Pools(
                np.array([0, 0, 0]),
                np.array([1, 2, 3]),
                np.zeros(3, dtype=bool),
                np.array([0, 3, 3, 3]),
            ),
        )
        data.training = SimpleNamespace(**vars(data))
        scores = fit_baseline(data.training, "cooccurrence", "fixture").model.score_users(
            0, len(data.users)
        )[data.pools.users, data.pools.items]
        scores[scores <= 0] = -np.inf
        np.testing.assert_allclose(scores[:2], np.array([1 / np.sqrt(3)] * 2), rtol=1e-6)
        self.assertTrue(np.isneginf(scores[2]))


if __name__ == "__main__":
    unittest.main()
