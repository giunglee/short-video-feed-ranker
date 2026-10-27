"""Actual artifact scoring and label-free replay agree in the existing pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feed_ranker.data.prepared import fit_features
from feed_ranker.models.artifacts import ModelArtifact, save_artifact
from feed_ranker.models.deep import ModelConfig, build_model
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import CandidateScope, Request
from feed_ranker.recommendation.loading import load_ranker, load_retriever
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.rankers import ModelRanker


class ArtifactRecommendationTests(unittest.TestCase):
    def test_loaded_scores_order_and_support_match_fitted_objects(self) -> None:
        state = fit_features(
            pd.DataFrame(
                {"user_id": [1, 1, 2], "video_id": [10, 20, 20], "duration_ms": [100, 200, 200]}
            )
        )
        config = ModelConfig(("user_code", "item_code", "train_duration_log1p"), hidden=(4,))
        model = build_model(config, state.schema).eval()
        request = Request("fixture", 1)
        scope = CandidateScope(np.array([10, 20]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_artifact(root / "ranker", ModelArtifact("deep", model, "", {}, config, state))
            ranker = load_ranker(root / "ranker")
            direct = ModelRanker(model, state, "fitted")
            np.testing.assert_array_equal(
                ranker.score(request, scope.items).scores, direct.score(request, scope.items).scores
            )
            cosine = SparseCosineCooccurrence([np.array([0, 1]), np.array([1])], 2)
            for kind, fitted in (("cosine", cosine), ("popularity", cosine.counts)):
                save_artifact(
                    root / kind,
                    ModelArtifact(kind, fitted, "", {}, users=state.users, items=state.items),
                )
            retrievers = [load_retriever(root / kind) for kind in ("cosine", "popularity")]
            for policy in ("rr_all", "rrf_all"):
                merged = MergedRetriever(retrievers, policy, 50)
                a = RecommendationPipeline(merged, direct, 2, 2).recommend(request, scope)
                b = RecommendationPipeline(merged, ranker, 2, 2).recommend(request, scope)
                np.testing.assert_array_equal(a.items, b.items)
                np.testing.assert_array_equal(a.retrieved.items, b.retrieved.items)
                self.assertEqual(a.supported_count, b.supported_count)
            with self.assertRaisesRegex(ValueError, "output"):
                load_ranker(root / "ranker", output="not_a_task")


if __name__ == "__main__":
    unittest.main()
