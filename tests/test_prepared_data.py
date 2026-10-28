"""Public fixtures for materialization, projection and train/score parity."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.instances import EventDataset, make_loader
from feed_ranker.data.prepared import PreparedDataset
from feed_ranker.preprocessing.kuairand import prepare_kuairand


def raw_fixture(directory: Path) -> Path:
    raw = directory / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "video_id": [10, 20, 20, 30, 10, 40, 20, 30],
            "time_ms": [5, 6, 5, 6, 10, 11, 10, 11],
            "duration_ms": [100, 200, 200, 300, 999, 999, 999, 999],
            "long_view": [1, 0, 1, 0, 1, 0, 1, 0],
            "is_click": [1, 1, 1, 0, 1, 0, 1, 1],
            "is_rand": [0] * 8,
            "tab": [1] * 8,
        }
    ).to_csv(raw / "log_standard_fixture.csv", index=False)
    return raw


class PreparedDataTests(unittest.TestCase):
    def test_snapshot_duration_is_distinct_and_cold_metadata_is_not_warm(self) -> None:
        from feed_ranker.preprocessing.features import fit_features
        from feed_ranker.preprocessing.metadata import ANONYMOUS

        train = pd.DataFrame(
            {"user_id": [1, 1], "video_id": [10, 20], "duration_ms": [100.0, 200.0]}
        )
        basic = pd.DataFrame(
            {
                "video_id": [10, 20, 40],
                "author_id": [7, 8, 7],
                "video_duration": [110.0, 220.0, 440.0],
            }
        )
        profiles = pd.DataFrame({"user_id": [1, 99], **{n: [0, 999] for n in ANONYMOUS}})
        state = fit_features(train, basic, profiles)
        values = state.gather(np.array([1, 99]), np.array([10, 40]))
        self.assertEqual(values["item_code"].tolist(), [1, 0])
        self.assertEqual(values["author_code"].tolist(), [1, 1])
        self.assertEqual(values["onehot_feat0"].tolist(), [1, 0])
        torch.testing.assert_close(
            values["train_duration_log1p"], torch.log1p(torch.tensor([100.0, 0.0])), rtol=0, atol=0
        )
        torch.testing.assert_close(
            values["metadata_duration_log1p"],
            torch.log1p(torch.tensor([110.0, 440.0])),
            rtol=0,
            atol=0,
        )

    def test_projection_preserves_distinct_events_and_train_only_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = raw_fixture(root)
            file = raw / "log_standard_fixture.csv"
            frame = pd.read_csv(file)
            # Different unselected outcome: retained even when projected rows match.
            other = frame.iloc[[0]].copy()
            other["is_click"] = 0
            pd.concat([frame, other, frame.iloc[[0]]]).to_csv(file, index=False)
            data = prepare_kuairand(raw, root / "data", users=0, train_end=10, val_end=15)
            data.validate()
            self.assertEqual(len(data.read_split("train")), 5)
            self.assertEqual(data.manifest["duplicates_removed"], 1)
            self.assertEqual(list(data.read_split("validation", ["long_view"])), ["long_view"])
            state = data.load_feature_state()
            values = state.gather(np.array([1, 1, 99]), np.array([10, 40, 999]))
            self.assertEqual(values["item_code"].tolist(), [1, 0, 0])
            torch.testing.assert_close(
                values["train_duration_log1p"],
                torch.log1p(torch.tensor([100.0, 0.0, 0.0])),
                atol=0,
                rtol=0,
            )
            self.assertEqual(values["train_duration_missing"].tolist(), [0, 1, 1])
            with self.assertRaises(ValueError):
                data.read_split("test")

    def test_state_read_is_independent_and_vectorized_batches_align(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = prepare_kuairand(
                raw_fixture(root),
                root / "data",
                users=0,
                train_end=10,
                val_end=15,
                targets=("long_view", "is_click"),
            )
            frame = data.read_split("train")
            with patch("pandas.read_parquet", side_effect=AssertionError("event read")):
                state = PreparedDataset(root / "data").load_feature_state()
            names = ("user_code", "item_code", "train_duration_log1p")
            dataset = EventDataset(frame, state, names, ("long_view", "is_click"))
            before = torch.get_rng_state().clone()
            batches = list(make_loader(dataset, 3, seed=42))
            torch.testing.assert_close(before, torch.get_rng_state())
            order = np.random.default_rng(42).permutation(len(frame))
            self.assertEqual([len(b["targets"]["long_view"]) for b in batches], [3, 1])
            np.testing.assert_array_equal(
                torch.cat([b["metadata"]["event_id"] for b in batches]),
                frame.event_id.to_numpy()[order],
            )
            for batch in batches:
                expected = state.gather(
                    batch["metadata"]["user_id"].numpy(),
                    batch["metadata"]["video_id"].numpy(),
                    names,
                )
                for name in names:
                    torch.testing.assert_close(
                        batch["features"][name], expected[name], atol=0, rtol=0
                    )
            (root / "data" / "train.parquet").write_bytes(b"corrupted")
            PreparedDataset(root / "data").load_feature_state()
            with self.assertRaisesRegex(ValueError, "changed"):
                data.read_split("train")


if __name__ == "__main__":
    unittest.main()
