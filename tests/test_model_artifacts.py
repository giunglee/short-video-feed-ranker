"""Scoring state round trips, corrupt-state failures and caller RNG isolation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.prepared import fit_features
from feed_ranker.models.artifacts import ModelArtifact, load_artifact, save_artifact
from feed_ranker.models.deep import ModelConfig, build_model
from feed_ranker.models.recall import BPRMF
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence


class ModelArtifactTests(unittest.TestCase):
    def test_all_scorers_roundtrip_without_events_and_preserve_rng(self) -> None:
        state = fit_features(
            pd.DataFrame(
                {"user_id": [1, 1, 2], "video_id": [10, 20, 20], "duration_ms": [100, 200, 200]}
            )
        )
        config = ModelConfig(("user_code", "item_code", "train_duration_log1p"), hidden=(4,))
        deep = build_model(config, state.schema).eval()
        mf = BPRMF(2, 2, 3)
        mf.item_updated[0] = True
        cosine = SparseCosineCooccurrence([np.array([0, 1]), np.array([1])], 2)
        artifacts = [ModelArtifact("deep", deep, "", {}, config=config, feature_state=state)]
        for kind, scorer in (("mf", mf), ("cosine", cosine), ("popularity", cosine.counts)):
            artifacts.append(
                ModelArtifact(
                    kind,
                    scorer,
                    "",
                    {},
                    users=state.users,
                    items=state.items,
                    supported_users=np.array([True, False]),
                )
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
                        deep(batch)["long_view"], loaded.model(batch)["long_view"], rtol=0, atol=0
                    )
                    self.assertEqual(loaded.feature_state.schema, state.schema)
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
            (Path(tmp) / "mf" / "weights.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(Path(tmp) / "mf")


if __name__ == "__main__":
    unittest.main()
