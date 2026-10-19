"""Feature ablations retain their inputs through legacy conversion and export."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.experiments.convert_inference import convert_checkpoint
from feed_ranker.features.config import DenseFeatureConfig, EmbeddingFeatureConfig, FeatureConfig
from feed_ranker.features.kuairec import KuaiRecFeatureBuilder
from feed_ranker.inference.artifacts import TrainingProvenance, load_ranker
from feed_ranker.models.ranking_factory import ModelSpec, build_model
from feed_ranker.recommendation.contracts import Request
from feed_ranker.training.export import export_ranker


class FeatureArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.config = FeatureConfig(
            dense_configs=[DenseFeatureConfig("video_duration", "log1p_nonnegative")],
            embedding_configs=[EmbeddingFeatureConfig("item_categories", 4, 8, "sum")],
        )
        self.builder = KuaiRecFeatureBuilder(
            self.config,
            {"item_categories": ["1", "2", "3"]},
            pd.DataFrame(index=pd.Index([7], dtype="int64")),
            pd.Series([1000.0, 2000.0, 3000.0], index=[10, 20, 30]),
            item_categories=pd.Series([(1, 2), (3,)], index=[10, 20]),
        )

    def test_category_export_works_without_data_and_preserves_sum_padding(self) -> None:
        features = self.builder.build(np.array([7, 7, 99]), np.array([20, 10, 30]))
        np.testing.assert_array_equal(features["item_categories"], [[3, -1], [1, 2], [0, -1]])
        model = build_model(self.config, ModelSpec("deep", (4,)))
        with torch.no_grad():
            embedding = model.embedding_encoder.embeddings["item_categories"]
            embedding.weight[1:].fill_(1.0)
        pooled = model.embedding_encoder(
            {"item_categories": torch.tensor(features["item_categories"])}
        )
        torch.testing.assert_close(pooled, torch.tensor([[1.0] * 8, [2.0] * 8, [0.0] * 8]))
        pooled.sum().backward()
        self.assertTrue(embedding.weight.grad[0].eq(0).all())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "model"
            export_ranker(
                directory,
                model,
                ModelSpec("deep", (4,)),
                self.builder,
                TrainingProvenance("weights", "data", 42, 2, 2, "user_xauc"),
            )
            with patch("pandas.read_parquet", side_effect=AssertionError("No source data")):
                loaded = load_ranker(directory)
                expected = (
                    model({k: torch.from_numpy(v) for k, v in features.items()})
                    .detach()
                    .flatten()
                    .numpy()
                )
                np.testing.assert_array_equal(loaded.ranker.predict(features), expected)
                self.assertEqual(
                    len(loaded.ranker.score(Request("q", 7), np.array([], dtype=np.int64)).scores),
                    0,
                )

    def test_legacy_feature_conversion_requires_explicit_ordered_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(
                {"raw_user_id": [7, 7], "raw_video_id": [10, 20], "item_categories": [[1, 2], [3]]}
            ).to_parquet(root / "train.parquet")
            pd.DataFrame({"video_id": [10, 20, 30], "median": [1000.0, 2000.0, 3000.0]}).to_parquet(
                root / "duration.parquet"
            )
            write_json(
                root / "meta.json",
                {
                    "vocab": self.builder.vocab,
                    "spec": {"dense": ["video_duration"], "crosses": {}},
                    "splits": {"train": {"sha256": sha256(root / "train.parquet")}},
                },
            )
            model = build_model(self.config, ModelSpec("deep", (4,)))
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": 3,
                    "training": {"seed": 42, "selection_metric": "user_xauc"},
                    "metadata": {
                        "features": ["item_categories"],
                        "recipe": {"architecture": "deep", "hidden": [4]},
                        "target_cap": 2,
                        "data_manifest_sha256": sha256(root / "meta.json"),
                    },
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "resolved"):
                convert_checkpoint(checkpoint, root, root / "duration.parquet", root / "reject")
            invalid = FeatureConfig.from_dict(asdict(self.config))
            invalid.dense_configs[0].transform = "identity"
            with self.assertRaisesRegex(ValueError, "log1p"):
                convert_checkpoint(
                    checkpoint, root, root / "duration.parquet", root / "reject", invalid
                )
            convert_checkpoint(
                checkpoint, root, root / "duration.parquet", root / "converted", self.config
            )
            loaded = load_ranker(root / "converted")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    loaded.ranker.model.state_dict()[key], value, rtol=0, atol=0
                )
            self.assertEqual(loaded.manifest.provenance.source_sha256, sha256(checkpoint))

    def test_invalid_category_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "category codes"):
            KuaiRecFeatureBuilder(
                self.config,
                self.builder.vocab,
                self.builder.profiles,
                self.builder.durations,
                item_categories=pd.Series([(-1,)], index=[10]),
            )
        with self.assertRaisesRegex(ValueError, "pooling"):
            EmbeddingFeatureConfig("item_categories", 4, 8, "mean")


if __name__ == "__main__":
    unittest.main()
