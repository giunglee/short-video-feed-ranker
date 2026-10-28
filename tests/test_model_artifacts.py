"""Scoring state round trips, corrupt-state failures and caller RNG isolation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import (
    CosineArtifact,
    MFArtifact,
    PopularityArtifact,
    RankerArtifact,
    load_artifact,
    save_artifact,
)
from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
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
        artifacts = [RankerArtifact(deep, state, "", {})]
        artifacts.extend(
            [
                MFArtifact(mf, state.user_ids, state.item_ids, np.array([True, False]), "", {}),
                CosineArtifact(cosine, state.user_ids, state.item_ids, "", {}),
                PopularityArtifact(cosine.counts, state.user_ids, state.item_ids, "", {}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            for artifact in artifacts:
                path = Path(tmp) / artifact.kind
                before = torch.get_rng_state().clone()
                save_artifact(path, artifact)
                loaded = load_artifact(path)
                torch.testing.assert_close(before, torch.get_rng_state())
                if artifact.kind == "deep":
                    batch = {"features": state.gather(np.array([1, 99]), np.array([10, 99]))}
                    torch.testing.assert_close(
                        deep(batch["features"])["long_view"],
                        loaded.model(batch["features"])["long_view"],
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(loaded.feature_state.vocab_sizes, state.vocab_sizes)
                elif artifact.kind == "mf":
                    for name, value in mf.state_dict().items():
                        torch.testing.assert_close(
                            value, loaded.model.state_dict()[name], rtol=0, atol=0
                        )
                    np.testing.assert_array_equal(loaded.supported_users, [True, False])
                elif artifact.kind == "cosine":
                    np.testing.assert_array_equal(
                        cosine.score(0, np.arange(2)), loaded.model.score(0, np.arange(2))
                    )
                else:
                    np.testing.assert_array_equal(cosine.counts, loaded.model)
                with self.assertRaises(FileExistsError):
                    save_artifact(path, artifact)
            path = Path(tmp) / "deep"
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["output_schema"]["long_view"]["meaning"] = "probability"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "meaning"):
                load_artifact(path)
            manifest["output_schema"]["long_view"]["meaning"] = "binary logit"
            manifest["model_config"]["hidden"] = [5]
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "shape/dtype/value"):
                load_artifact(path)
            (Path(tmp) / "mf" / "weights.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(Path(tmp) / "mf")

    def test_v2_artifact_scores_without_reconstructing_training_objective(self) -> None:
        state = fit_entity_features(
            pd.DataFrame({"user_id": [1, 1], "video_id": [10, 20], "duration_ms": [100, 200]})
        )
        model = DeepModel(DeepConfig(dense=("train_duration_log1p",), hidden=(4,))).eval()
        features = state.gather(np.array([1, 99]), np.array([10, 999]))
        expected = model(features)["long_view"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            save_artifact(path, RankerArtifact(model, state, "", {}))
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["version"], "fitted-scorer-v3")
            self.assertNotIn("tasks", manifest)
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
            (path / "manifest.json").write_text(json.dumps(manifest))
            restored = load_artifact(path)
            torch.testing.assert_close(
                restored.model(features)["long_view"], expected, rtol=0, atol=0
            )
            manifest["version"] = "unsupported"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "version/kind"):
                load_artifact(path)


if __name__ == "__main__":
    unittest.main()
