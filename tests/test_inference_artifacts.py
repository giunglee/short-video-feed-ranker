"""Inference artifacts must survive removal of all training/evaluation inputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.recall import VERSION, RecallData
from feed_ranker.evaluation.retrieval import pool_scores
from feed_ranker.experiments.convert_inference import convert_checkpoint, convert_retriever
from feed_ranker.features.config import (
    CrossFeatureConfig,
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
    FeatureConfig,
)
from feed_ranker.features.kuairec import KuaiRecFeatureBuilder
from feed_ranker.inference.artifacts import TrainingProvenance, load_ranker, load_retriever
from feed_ranker.inference.rankers import ModelRanker
from feed_ranker.models.ranking_factory import ModelSpec, build_model
from feed_ranker.models.recall import BPRMF
from feed_ranker.recommendation.contracts import CandidateScope, Request
from feed_ranker.training.export import export_ranker
from feed_ranker.training.recall import fit_baseline


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.request = Request("query", 10)
        self.items = np.array([200, 100])
        self.provenance = TrainingProvenance("checkpoint", "data", 42, 3, 2.0, "mae")

    def features(self, include_age: bool) -> KuaiRecFeatureBuilder:
        vocab = {"user_id": ["10", "20"], "video_id": ["100", "200"]}
        profiles = pd.DataFrame(index=pd.Index([10, 20], dtype="int64"))
        if include_age:
            vocab["age_range"] = ["young", "old"]
            profiles["age_range"] = [1, 2]
        config = FeatureConfig(
            dense_configs=[DenseFeatureConfig("video_duration", "log1p_nonnegative")],
            embedding_configs=[EmbeddingFeatureConfig(n, len(v) + 1, 2) for n, v in vocab.items()],
        )
        return KuaiRecFeatureBuilder(
            config, vocab, profiles, pd.Series([1000.0, 2000.0], index=[100, 200])
        )

    def test_architecture_and_feature_variants_load_without_experiment_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ("linear", "wide", "deep", "wide_deep"):
                for include_age in (False, True):
                    spec = ModelSpec(architecture, (4,) if "deep" in architecture else ())
                    features = self.features(include_age)
                    model = build_model(features.config, spec)
                    before = ModelRanker(model, features, "original").score(
                        self.request, self.items
                    )
                    destination = Path(tmp) / f"{architecture}_{include_age}"
                    export_ranker(destination, model, spec, features, self.provenance)
                    rng = torch.get_rng_state().clone()
                    with patch(
                        "pandas.read_parquet", side_effect=AssertionError("No dataset reads")
                    ):
                        loaded = load_ranker(destination)
                        after = loaded.ranker.score(self.request, self.items)
                    torch.testing.assert_close(torch.get_rng_state(), rng)
                    np.testing.assert_array_equal(before.scores, after.scores)
                    self.assertEqual(
                        "age_range" in loaded.features.build(np.array([10, 10]), self.items),
                        include_age,
                    )
                    self.assertEqual(loaded.manifest.provenance.selection_metric, "mae")
                    with self.assertRaises(FileExistsError):
                        export_ranker(destination, model, spec, features, self.provenance)

    def test_wide_cross_uses_fitted_categories_and_unknown_user_has_no_match(self) -> None:
        features = self.features(True)
        features.config.cross_feature_configs = [CrossFeatureConfig("young_short")]
        builder = KuaiRecFeatureBuilder(
            features.config,
            features.vocab,
            features.profiles,
            features.durations,
            {"young_short": {"age_range": "young", "duration_band": "0–5s"}},
        )
        batch = builder.build(np.array([10, 20, 999]), np.array([100, 100, 100]))
        np.testing.assert_array_equal(batch["young_short"], [1.0, 0.0, 0.0])
        spec = ModelSpec("wide")
        model = build_model(builder.config, spec)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "artifact"
            export_ranker(destination, model, spec, builder, self.provenance)
            loaded = load_ranker(destination)
            np.testing.assert_array_equal(
                loaded.ranker.score(self.request, self.items).scores,
                ModelRanker(model, builder, "original").score(self.request, self.items).scores,
            )

    def test_resolved_checkpoint_config_overrides_legacy_recipe_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "training"
            source.mkdir()
            features, spec = self.features(False), ModelSpec("deep", (4,))
            model = build_model(features.config, spec)
            pd.DataFrame({"raw_user_id": [10, 20]}).to_parquet(source / "train.parquet")
            write_json(
                source / "meta.json",
                {
                    "vocab": features.vocab,
                    "spec": {"crosses": {}},
                    "splits": {"train": {"sha256": sha256(source / "train.parquet")}},
                },
            )
            pd.DataFrame({"video_id": [100, 200], "median": [1000.0, 2000.0]}).to_parquet(
                source / "duration.parquet"
            )
            checkpoint = source / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": 0,
                    "training": {"seed": 42, "selection_metric": "mae"},
                    "metadata": {
                        "recipe": {"feature_set": "profiles"},
                        "model_spec": asdict(spec),
                        "feature_config": asdict(features.config),
                        "data_manifest_sha256": sha256(source / "meta.json"),
                        "target_cap": 2,
                    },
                },
                checkpoint,
            )
            convert_checkpoint(checkpoint, source, source / "duration.parquet", root / "artifact")
            source.rename(root / "unavailable")
            loaded = load_ranker(root / "artifact")
            np.testing.assert_array_equal(
                loaded.ranker.score(self.request, self.items).scores,
                ModelRanker(model, features, "old").score(self.request, self.items).scores,
            )
            self.assertNotIn("age_range", loaded.features.config.embedding_cols)

    def test_corrupt_and_incompatible_artifacts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "artifact"
            features, spec = self.features(True), ModelSpec("deep", (4,))
            export_ranker(
                directory, build_model(features.config, spec), spec, features, self.provenance
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            write_json(manifest_path, {**manifest, "version": "unknown"})
            with self.assertRaisesRegex(ValueError, "version/kind"):
                load_ranker(directory)
            write_json(manifest_path, manifest)
            state = torch.load(directory / "state.pt", weights_only=True)
            state["vocab"]["user_id"].append("30")
            torch.save(state, directory / "state.pt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_ranker(directory)
            manifest["state_sha256"] = sha256(directory / "state.pt")
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "Vocabulary size"):
                load_ranker(directory)
            state["vocab"]["user_id"].pop()
            weight = next(n for n, v in state["model"].items() if v.dtype == torch.float32)
            state["model"][weight] = state["model"][weight].double()
            torch.save(state, directory / "state.pt")
            manifest["state_sha256"] = sha256(directory / "state.pt")
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "weight dtype"):
                load_ranker(directory)

    def test_retrievers_preserve_nontraining_user_rows_and_support_without_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            train = pd.DataFrame(
                {
                    "user_id": [10, 10, 20, 20],
                    "video_id": [100, 200, 200, 300],
                    "outcome": [1, -1, 1, -1],
                }
            )
            val = pd.DataFrame(
                {"user_id": [5, 10, 20], "video_id": [100, 300, 100], "outcome": [1, 0, 1]}
            )
            for split, frame in (("train", train), ("val", val)):
                frame.to_parquet(source / f"{split}.parquet", index=False)
            write_json(
                source / "meta.json",
                {
                    "version": VERSION,
                    "files": {
                        f"{s}.parquet": sha256(source / f"{s}.parquet") for s in ("train", "val")
                    },
                },
            )
            data = RecallData(source)
            expected = {}
            for method in ("popularity", "cooccurrence", "mf"):
                run = source / method
                run.mkdir()
                config = {
                    "method": method,
                    "seed": 42,
                    "dimension": 2,
                    "data_manifest_sha256": sha256(source / "meta.json"),
                }
                write_json(run / "config.json", config)
                if method == "mf":
                    model = BPRMF(len(data.users), len(data.items), 2)
                    model.item_updated[:] = torch.tensor([True, False, True])
                    torch.save({"model": model.state_dict(), "config": config}, run / "best.pt")
                    expected[method] = pool_scores(model, data)
                else:
                    fitted = fit_baseline(data.training, method, "fixture")
                    if method == "popularity":
                        values = fitted.counts[data.pools.items].copy()
                    else:
                        values = fitted.model.score_users(0, len(data.users))[
                            data.pools.users, data.pools.items
                        ]
                    values[values <= 0] = -np.inf
                    expected[method] = values
                convert_retriever(data, run, root / method)
            source.rename(root / "unavailable")
            with patch("pandas.read_parquet", side_effect=AssertionError("No dataset reads")):
                for method in expected:
                    loaded = load_retriever(root / method)
                    for u, user in enumerate(data.users):
                        lo, hi = data.pools.offsets[u : u + 2]
                        scope = CandidateScope(data.items[data.pools.items[lo:hi]])
                        got = loaded.retrieve(Request(str(user), int(user)), scope, 50)
                        finite = np.isfinite(expected[method][lo:hi])
                        np.testing.assert_array_equal(got.candidates.items, scope.items[finite])
                        np.testing.assert_allclose(
                            got.candidates.scores, expected[method][lo:hi][finite]
                        )
                    self.assertEqual(
                        loaded.retrieve(
                            Request("cold", 999), CandidateScope(np.array([100])), 50
                        ).supported_count,
                        1 if method == "popularity" else 0,
                    )
