"""Scoring state round trips, corrupt-state failures and caller RNG isolation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import (
    load_artifact,
    save_cosine,
    save_mf,
    save_popularity,
    save_ranker,
)
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.persistence import sha256
from feed_ranker.preprocessing.features import fit_entity_features


class ModelArtifactTests(unittest.TestCase):
    def test_all_scorers_roundtrip_without_events_and_preserve_rng(self) -> None:
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
        deep = DeepModel(config).eval()
        mf = BPRMF(2, 2, 3)
        mf.item_updated[0] = True
        cosine = SparseCosineCooccurrence([np.array([0, 1]), np.array([1])], 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = torch.get_rng_state().clone()
            # Saving must not deserialize state or initialize a second model.
            with (
                patch("feed_ranker.artifacts.load_artifact", side_effect=AssertionError("reload")),
                patch("feed_ranker.artifacts.DeepModel", side_effect=AssertionError("reconstruct")),
            ):
                save_ranker(root / "deep", deep, state, {})
                save_mf(
                    root / "mf", mf, state.user_ids, state.item_ids, np.array([True, False]), {}
                )
                save_cosine(root / "cosine", cosine, state.user_ids, state.item_ids, {})
                save_popularity(
                    root / "popularity", cosine.counts, state.user_ids, state.item_ids, {}
                )
            for kind in ("deep", "mf", "cosine", "popularity"):
                path = root / kind
                loaded = load_artifact(path)
                torch.testing.assert_close(before, torch.get_rng_state())
                if kind == "deep":
                    batch = {
                        "features": {
                            n: torch.from_numpy(v)
                            for n, v in state.transform(
                                np.array([1, 99]), np.array([10, 99])
                            ).items()
                        }
                    }
                    torch.testing.assert_close(
                        deep(batch["features"])["long_view"],
                        loaded.model(batch["features"])["long_view"],
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(loaded.feature_state.vocab_sizes, state.vocab_sizes)
                elif kind == "mf":
                    for name, value in mf.state_dict().items():
                        torch.testing.assert_close(
                            value, loaded.model.state_dict()[name], rtol=0, atol=0
                        )
                    np.testing.assert_array_equal(loaded.supported_users, [True, False])
                elif kind == "cosine":
                    np.testing.assert_array_equal(
                        cosine.score(0, np.arange(2)), loaded.model.score(0, np.arange(2))
                    )
                else:
                    np.testing.assert_array_equal(cosine.counts, loaded.model)
            with self.assertRaises(FileExistsError):
                save_ranker(root / "deep", deep, state, {})
            path = Path(tmp) / "deep"
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["output_schema"]["long_view"]["meaning"] = "probability"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "meaning"):
                load_artifact(path)
            manifest["output_schema"]["long_view"]["meaning"] = "binary logit"
            config_path = path / "config.json"
            config = json.loads(config_path.read_text())
            config["hidden"] = [5]
            config_path.write_text(json.dumps(config))
            manifest["files"]["config.json"] = sha256(config_path)
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "shape/dtype/value"):
                load_artifact(path)
            (Path(tmp) / "mf" / "weights.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(Path(tmp) / "mf")

    def test_export_failure_never_publishes_partial_or_invalid_state(self) -> None:
        state = fit_entity_features(
            pd.DataFrame({"user_id": [1], "video_id": [10], "duration_ms": [100]})
        )
        model = DeepModel(DeepConfig(dense=("train_duration_log1p",), hidden=(4,)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            with patch.object(state, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_ranker(path, model, state, {})
            self.assertEqual(list(Path(tmp).iterdir()), [])
            state.item_features["train_duration_log1p"][1] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite fitted feature"):
                save_ranker(path, model, state, {})
            self.assertFalse(path.exists())
            state.item_features["train_duration_log1p"][1] = 0
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite model weights"):
                save_ranker(path, model, state, {})
            self.assertFalse(path.exists())

    def test_export_rejects_unsupported_parameter_dtypes(self) -> None:
        state = fit_entity_features(
            pd.DataFrame({"user_id": [1], "video_id": [10], "duration_ms": [100]})
        )
        deep = DeepModel(DeepConfig(dense=("train_duration_log1p",), hidden=(4,))).double()
        mf = BPRMF(1, 1, 2).double()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "require torch.float32"):
                save_ranker(root / "deep", deep, state, {})
            with self.assertRaisesRegex(ValueError, "require torch.float32"):
                save_mf(root / "mf", mf, state.user_ids, state.item_ids, np.array([True]), {})
            with self.assertRaisesRegex(ValueError, "Invalid MF dimension"):
                save_mf(
                    root / "empty",
                    BPRMF(1, 1, 0),
                    state.user_ids,
                    state.item_ids,
                    np.array([True]),
                    {},
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_legacy_artifacts_keep_feature_values_without_training_objectives(self) -> None:
        state = fit_entity_features(
            pd.DataFrame({"user_id": [1, 1], "video_id": [10, 20], "duration_ms": [100, 200]})
        )
        model = DeepModel(DeepConfig(dense=("train_duration_log1p",), hidden=(4,))).eval()
        features = {
            n: torch.from_numpy(v)
            for n, v in state.transform(np.array([1, 99]), np.array([10, 999])).items()
        }
        expected = model(features)["long_view"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            save_ranker(path, model, state, {})
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["version"], "fitted-scorer-v5")
            self.assertNotIn("tasks", manifest)
            # v4 embedded configuration and stored feature files at the top level.
            manifest["model_config"] = json.loads((path / "config.json").read_text())
            (path / "config.json").unlink()
            for feature_file in (path / "features").iterdir():
                feature_file.rename(path / feature_file.name)
            (path / "features").rmdir()
            manifest["files"] = {
                name: sha256(path / name)
                for name in ("weights.pt", "feature-state.npz", "feature-tables.json")
            }
            manifest["version"] = "fitted-scorer-v4"
            (path / "manifest.json").write_text(json.dumps(manifest))
            restored = load_artifact(path)
            torch.testing.assert_close(
                restored.model(features)["long_view"], expected, rtol=0, atol=0
            )
            # Recreate the stored flat ownership map used by existing v2/v3 scorers.
            definition_path = path / "feature-tables.json"
            definition = json.loads(definition_path.read_text())
            definition["entities"] = {
                **dict.fromkeys(definition.pop("user_features"), "user"),
                **dict.fromkeys(definition.pop("item_features"), "item"),
            }
            definition_path.write_text(json.dumps(definition))
            manifest["files"]["feature-tables.json"] = sha256(definition_path)
            # The preceding format stores training supervision separately from weights.
            manifest["version"] = "fitted-scorer-v2"
            manifest["tasks"] = [
                {
                    "output": "long_view",
                    "target": "long_view",
                    "loss": "bce_logits",
                    "mask": None,
                    "weight": None,
                    "task_weight": 2.0,
                }
            ]
            for version in ("fitted-scorer-v2", "fitted-scorer-v3"):
                manifest["version"] = version
                (path / "manifest.json").write_text(json.dumps(manifest))
                restored = load_artifact(path)
                restored_features = {
                    n: torch.from_numpy(v)
                    for n, v in restored.feature_state.transform(
                        np.array([1, 99]), np.array([10, 999])
                    ).items()
                }
                for name in features:
                    torch.testing.assert_close(
                        features[name], restored_features[name], rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    restored.model(restored_features)["long_view"], expected, rtol=0, atol=0
                )
            manifest["version"] = "unsupported"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "version/kind"):
                load_artifact(path)


if __name__ == "__main__":
    unittest.main()
