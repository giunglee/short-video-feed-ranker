"""Public synthetic behavior checks for strict temporal KuaiRand study setup."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.kuairand import (
    EventDataset,
    histories,
    load_prepared,
    prepare,
    temporal_events,
)
from feed_ranker.evaluation.binary import binary_metrics, user_auc_metrics
from feed_ranker.experiments.kuairand_study import run
from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures
from feed_ranker.inference.kuairand import SparseCooccurrenceRetriever, load_ranker, load_retrievers
from feed_ranker.inference.retrievers import CooccurrenceRetriever
from feed_ranker.models.recall import CosineCooccurrence
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import CandidateScope, Request


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

    def test_auc_resume_matches_uninterrupted_and_checks_recall_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            events().to_csv(raw / "log_standard_fixture.csv", index=False)
            prepare(raw, root / "data", users=0, train_end=10, val_end=15)
            base = dict(
                data=root / "data",
                seeds=[42],
                mf_epochs=1,
                batch_size=4,
                threads=1,
                budget=2,
                recipes=["reference"],
                selection_metric="roc_auc",
                config=Path(__file__).resolve().parents[1] / "configs/kuairand_study.json",
            )
            run(argparse.Namespace(**base, out=root / "first", epochs=1))
            run(
                argparse.Namespace(
                    **base,
                    out=root / "resumed",
                    epochs=3,
                    resume_from=root / "first",
                    recall_from=root / "first",
                )
            )
            run(
                argparse.Namespace(
                    **base,
                    out=root / "continuous",
                    epochs=3,
                    recall_from=root / "first",
                )
            )
            a, b = [root / name / "seed42" / "reference" for name in ("resumed", "continuous")]
            for file in ("best.pt", "last.pt"):
                left, right = [torch.load(p / file, weights_only=True) for p in (a, b)]
                self.assertEqual(left["epoch"], right["epoch"])
                for key in left["model"]:
                    torch.testing.assert_close(
                        left["model"][key], right["model"][key], rtol=0, atol=0
                    )
            history = json.loads((a / "history.json").read_text())
            values = [row["val"]["roc_auc"] for row in history]
            result = json.loads((a / "result.json").read_text())
            self.assertEqual(result["best_epoch"], values.index(max(values)))
            self.assertEqual(result["checkpoint_policy"]["mode"], "max")
            for channel in ("popularity", "cooccurrence", "mf"):
                np.testing.assert_array_equal(
                    np.load(a / f"{channel}_candidates.npz")["items"],
                    np.load(b / f"{channel}_candidates.npz")["items"],
                )
            (root / "first" / "seed42" / "mf_scores.parquet").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Recall artifact hash mismatch"):
                run(
                    argparse.Namespace(
                        **base,
                        out=root / "invalid",
                        epochs=1,
                        recall_from=root / "first",
                    )
                )

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

    def test_sparse_retriever_matches_existing_dense_adapter(self) -> None:
        h = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        users, items = np.array([10, 20]), np.array([100, 200, 300])
        sparse = SparseCooccurrenceRetriever(
            users,
            items,
            SparseCosineCooccurrence([np.flatnonzero(row.numpy()) for row in h], 3),
            "sparse",
        )
        dense = CooccurrenceRetriever(users, items, CosineCooccurrence(h), "dense")
        scope = CandidateScope(np.array([100, 200, 300, 999]))
        for user in [10, 20, 99]:
            q = Request(str(user), user)
            a, b = [r.retrieve(q, scope, 3) for r in [sparse, dense]]
            np.testing.assert_array_equal(np.sort(a.candidates.items), np.sort(b.candidates.items))
            aligned = pd.Series(b.candidates.scores, index=b.candidates.items).reindex(
                a.candidates.items
            )
            np.testing.assert_allclose(a.candidates.scores, aligned.to_numpy(), atol=1e-6)
            again = sparse.retrieve(q, scope, 3)
            np.testing.assert_array_equal(a.candidates.items, again.candidates.items)
            self.assertEqual(a.supported_count, b.supported_count)

    def test_large_sparse_vocabulary_and_invalid_histories(self) -> None:
        model = SparseCosineCooccurrence([np.array([0, 99999]), np.array([99999])], 100000)
        self.assertLess(model.storage_bytes, 4_000_000)
        np.testing.assert_allclose(
            model.score(0, np.array([0, 99999, 50])), [1 / np.sqrt(2), 1 / np.sqrt(2), 0]
        )
        with self.assertRaises(ValueError):
            SparseCosineCooccurrence([np.array([1, 1])], 3)

    def test_temporal_repeats_missing_and_feature_ablation_contract(self) -> None:
        source = events()
        train, val, duplicates = temporal_events(
            pd.concat([source, source.iloc[[0]]]), "long_view", 10, 15
        )
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(val), 5)
        self.assertTrue(train.time_ms.lt(10).all())
        self.assertEqual(val.time_ms.eq(10).sum(), 3)
        encoding = Encoding.fit(train)
        pos, neg = histories(train, encoding, "long_view")
        self.assertEqual(pos[0].tolist(), [0])
        self.assertEqual(neg[0].tolist(), [1])
        for feature_set in ("base", "no_user", "no_item", "no_duration", "duration_band"):
            builder = KuaiRandFeatures(encoding, feature_set)
            batch = builder.build(np.array([1, 99]), np.array([10, 99]))
            self.assertEqual(set(batch), set(builder.config.feature_cols))
            if feature_set == "no_user":
                self.assertNotIn("user_id", batch)
            if feature_set == "no_item":
                self.assertNotIn("video_id", batch)
            if feature_set == "no_duration":
                self.assertEqual(set(batch), {"user_id", "video_id", "constant"})
            data = EventDataset(val, encoding, "long_view", feature_set)
            poisoned = EventDataset(
                val.assign(duration_ms=999999, long_view=1 - val.long_view),
                encoding,
                "long_view",
                feature_set,
            )
            for key in data.features:
                torch.testing.assert_close(data.features[key], poisoned.features[key])
            self.assertEqual(sum(len(b["label"]) for b in data.batches(2, 42)), len(val))
        band = KuaiRandFeatures(
            Encoding(
                np.array([1]),
                np.arange(5),
                np.array([0, 0, 10000, 30000, 60000, 60001], dtype=np.float32),
            ),
            "duration_band",
        )
        np.testing.assert_array_equal(
            band.build(np.ones(5, dtype=int), np.arange(5))["duration_band"], [0, 1, 2, 3, 4]
        )
        with self.assertRaises(ValueError):
            KuaiRandFeatures(encoding, "is_like")

    def test_all_recipes_shared_pipeline_artifacts_and_cache_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            events().to_csv(raw / "log_standard_fixture.csv", index=False)
            prepare(raw, root / "data", users=0, train_end=10, val_end=15)
            train, val, encoding, manifest = load_prepared(root / "data")
            self.assertEqual(manifest["selected_users"], [1, 2])
            args = argparse.Namespace(
                data=root / "data",
                out=root / "study",
                seeds=[42],
                epochs=1,
                mf_epochs=1,
                batch_size=4,
                threads=1,
                budget=2,
                recipes=None,
                config=Path(__file__).resolve().parents[1] / "configs/kuairand_study.json",
            )
            result = run(args)
            self.assertEqual(len(result["runs"]), 7)
            self.assertEqual(result["coverage"]["all_positive_pairs"], 3)
            self.assertEqual(result["coverage"]["warm_positive_pairs"], 2)
            restored = load_retrievers(root / "study" / "seed42")
            for retriever in restored:
                q = Request("fresh-query", 1)
                scope = CandidateScope(np.array([10, 20, 99]))
                retrieval = retriever.retrieve(q, scope, 2)
                self.assertTrue(np.isfinite(retrieval.candidates.scores).all())
                self.assertNotIn(99, retrieval.candidates.items)
            reference = None
            for row in result["runs"]:
                self.assertEqual(set(row["funnel"]), {"popularity", "cooccurrence", "mf"})
                directory = root / "study" / "seed42" / row["recipe"]
                ranker = load_ranker(directory / "inference.pt")
                scores = ranker.score(Request("fresh-query", 1), np.array([10, 99])).scores
                self.assertTrue(np.isfinite(scores).all())
                candidates = np.load(directory / "mf_candidates.npz")["items"]
                if reference is None:
                    reference = candidates
                else:
                    np.testing.assert_array_equal(reference, candidates)
                for metric in row["funnel"].values():
                    self.assertLessEqual(metric["ndcg"], metric["oracle_ndcg"] + 1e-12)
            (root / "data" / "train.parquet").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                load_prepared(root / "data")


if __name__ == "__main__":
    unittest.main()
