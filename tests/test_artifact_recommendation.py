"""Actual artifact scoring and label-free replay agree in the existing pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feed_ranker.artifacts import (
    CosineArtifact,
    PopularityArtifact,
    RankerArtifact,
    save_artifact,
)
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.models.recall import SparseCosineCooccurrence
from feed_ranker.preprocessing.features import fit_entity_features
from feed_ranker.recommendation.contracts import CandidateScope, Request
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.rankers import ModelRanker, load_ranker
from feed_ranker.recommendation.retrievers import load_retriever


class ArtifactRecommendationTests(unittest.TestCase):
    def test_loaded_scores_order_and_support_match_fitted_objects(self) -> None:
        state = fit_entity_features(
            pd.DataFrame(
                {"user_id": [1, 1, 2], "video_id": [10, 20, 20], "duration_ms": [100, 200, 200]}
            )
        )
        config = DeepConfig(
            dense=("train_duration_log1p",),
            sparse={n: state.vocab_sizes[n] for n in ("user_code", "item_code")},
            hidden=(4,),
        )
        model = DeepModel(config).eval()
        request = Request("fixture", 1)
        scope = CandidateScope(np.array([10, 20]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_artifact(root / "ranker", RankerArtifact(model, state, "", {}))
            ranker = load_ranker(root / "ranker")
            direct = ModelRanker(RankerArtifact(model, state, "fitted", {}))
            np.testing.assert_array_equal(
                ranker.score(request, scope.items).scores, direct.score(request, scope.items).scores
            )
            cosine = SparseCosineCooccurrence([np.array([0, 1]), np.array([1])], 2)
            for kind, fitted in (("cosine", cosine), ("popularity", cosine.counts)):
                save_artifact(
                    root / kind,
                    (PopularityArtifact if kind == "popularity" else CosineArtifact)(
                        fitted, state.user_ids, state.item_ids, "", {}
                    ),
                )
            retrievers = [load_retriever(root / kind) for kind in ("cosine", "popularity")]
            for policy in ("round_robin", "rrf"):
                merged = MergedRetriever(retrievers, policy, source_depth=50)
                a = RecommendationPipeline(merged, direct, 2, 2).recommend(request, scope)
                b = RecommendationPipeline(merged, ranker, 2, 2).recommend(request, scope)
                np.testing.assert_array_equal(a.items, b.items)
                np.testing.assert_array_equal(a.retrieved.items, b.retrieved.items)
                self.assertEqual(a.supported_count, b.supported_count)
            with self.assertRaisesRegex(ValueError, "output"):
                load_ranker(root / "ranker", output="not_a_task")


if __name__ == "__main__":
    unittest.main()
