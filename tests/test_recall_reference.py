"""Current fitting/scoring retains the frozen pre-refactor recall behavior."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from feed_ranker.artifacts import (
    save_cosine,
    save_mf,
    save_popularity,
)
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import CandidateScope, Request
from feed_ranker.recommendation.retrievers import load_retriever
from feed_ranker.training.bpr import ObservedSampler, fit_mf
from feed_ranker.training.optimizer import OptimizerConfig, build_optimizer


def reference() -> dict[str, Any]:
    return torch.load(Path(__file__).parent / "fixtures/recall_reference.pt", weights_only=True)


def sampler() -> ObservedSampler:
    return ObservedSampler(
        [np.array([0, 1]), np.array([1]), np.array([2])],
        [np.array([2, 3]), np.array([0]), np.array([], dtype=np.int64)],
    )


class RecallReferenceTests(unittest.TestCase):
    def test_current_artifacts_preserve_original_scores_without_event_reads(self) -> None:
        fixture = reference()
        arrays = {n: v.numpy() for n, v in fixture["arrays"].items()}
        users, items = arrays["users"], arrays["items"]
        mf = BPRMF(len(users), len(items), 32)
        mf.load_state_dict(fixture["checkpoint"]["model"])
        cosine = SparseCosineCooccurrence([np.array([0, 1]), np.array([1]), np.array([2])], 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_mf(root / "mf", mf, users, items, arrays["supported"], {})
            save_cosine(root / "cosine", cosine, users, items, {})
            save_popularity(root / "popularity", cosine.counts, users, items, {})
            for kind in ("mf", "cosine", "popularity"):
                path = root / kind
                rng = torch.get_rng_state().clone()
                with patch("pandas.read_parquet", side_effect=AssertionError("Event read")):
                    retriever = load_retriever(path)
                    for user, expected in zip(
                        [1, 2, 3, 99], fixture["expected"][retriever.name], strict=True
                    ):
                        actual = retriever.retrieve(
                            Request(str(user), user),
                            CandidateScope(np.array([10, 20, 30, 40, 999])),
                            50,
                        )
                        np.testing.assert_array_equal(actual.candidates.items, expected["items"])
                        np.testing.assert_allclose(
                            actual.candidates.scores, expected["scores"], rtol=1e-6, atol=1e-7
                        )
                        self.assertEqual(actual.supported_count, expected["supported"])
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_bpr_updates_match_original_weights_curve_and_support(self) -> None:
        fixture = reference()
        torch.manual_seed(42)
        model = BPRMF(3, 4, 32)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3))
        observed = sampler()
        history = fit_mf(model, observed, optimizer, epochs=2, seed=42)
        for actual, expected in zip(history, fixture["history"], strict=True):
            self.assertEqual(
                (actual["epoch"], actual["pairs"]), (expected["epoch"], expected["pairs"])
            )
            np.testing.assert_allclose(
                actual["train_loss"], expected["train_loss"], rtol=1e-6, atol=1e-7
            )
        np.testing.assert_array_equal(observed.supported, fixture["arrays"]["supported"].numpy())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, fixture["checkpoint"]["model"][name], rtol=1e-6, atol=1e-7
            )

    def test_bpr_uses_supplied_model_and_optimizer_without_resetting_rng(self) -> None:
        model = BPRMF(3, 4, 7).eval()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        # A zero learning rate makes replacement with an internal optimizer observable.
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        fit_mf(model, sampler(), optimizer, epochs=1, seed=42, batch_size=2)
        self.assertTrue(model.training)
        self.assertTrue(model.item_updated.all())
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
