"""Public fixtures for materialization, projection and train/score parity."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.dataset import EventDataset, make_loader
from feed_ranker.data.parquet import ParquetReader
from feed_ranker.persistence import sha256
from feed_ranker.preprocessing.features import FeatureState
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
        from feed_ranker.preprocessing.features import ANONYMOUS, fit_entity_features

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
        state = fit_entity_features(train, basic, profiles)
        values = state.transform(np.array([1, 99]), np.array([10, 40]))
        self.assertEqual(values["item_code"].tolist(), [1, 0])
        self.assertEqual(values["author_code"].tolist(), [1, 1])
        self.assertEqual(values["onehot_feat0"].tolist(), [1, 0])
        torch.testing.assert_close(
            torch.from_numpy(values["train_duration_log1p"]),
            torch.log1p(torch.tensor([100.0, 0.0])),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.from_numpy(values["metadata_duration_log1p"]),
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
            values = state.transform(np.array([1, 1, 99]), np.array([10, 40, 999]))
            self.assertEqual(values["item_code"].tolist(), [1, 0, 0])
            torch.testing.assert_close(
                torch.from_numpy(values["train_duration_log1p"]),
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
                state = ParquetReader(root / "data").load_feature_state()
            names = ("user_code", "item_code", "train_duration_log1p")
            dataset = EventDataset(frame, names, ("long_view", "is_click"))
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
                expected = state.transform(
                    batch["metadata"]["user_id"].numpy(),
                    batch["metadata"]["video_id"].numpy(),
                    names,
                )
                for name in names:
                    torch.testing.assert_close(
                        batch["features"][name], torch.from_numpy(expected[name]), atol=0, rtol=0
                    )
            (root / "data" / "train.parquet").write_bytes(b"corrupted")
            ParquetReader(root / "data").load_feature_state()
            with self.assertRaisesRegex(ValueError, "changed"):
                data.read_split("train")

    def test_materialized_rows_load_and_batch_without_feature_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reader = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            names = tuple(reader.feature_dtypes)
            with (
                patch.object(FeatureState, "load", side_effect=AssertionError("state read")),
                patch.object(FeatureState, "transform", side_effect=AssertionError("lookup")),
            ):
                columns = [*reader.metadata_dtypes, *names, *reader.targets]
                frame = reader.read_split("validation", columns)
                dataset = EventDataset(frame, names, reader.targets)
                batches = list(make_loader(dataset, 3))
                for name in names:
                    np.testing.assert_array_equal(
                        torch.cat([b["features"][name] for b in batches]), frame[name].to_numpy()
                    )
                self.assertEqual(dataset[1]["features"]["item_code"].ndim, 0)
                expected = dataset[0]["features"]["item_code"].clone()
                frame.loc[0, "item_code"] = 123
                torch.testing.assert_close(dataset[0]["features"]["item_code"], expected)

    def test_reader_projects_columns_and_reads_only_declared_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_kuairand(raw_fixture(root), root / "data", users=0, train_end=10, val_end=15)
            with patch("pandas.read_parquet", side_effect=AssertionError("eager read")):
                reader = ParquetReader(root / "data")
            with self.assertRaisesRegex(ValueError, "not declared"):
                reader.read_split("test")
            for columns in ([], ["item_code", "item_code"], ["unknown"]):
                with self.assertRaisesRegex(ValueError, "unique known"):
                    reader.read_split("train", columns)
            columns = ["train_duration_log1p", "item_code", "long_view"]
            with patch("pandas.read_parquet", wraps=pd.read_parquet) as read:
                frame = reader.read_split("train", columns)
                self.assertEqual(list(frame), columns)
                self.assertEqual(read.call_args.kwargs["columns"], columns)
            # An explicit synthetic test split exercises the reader, not model evaluation.
            test = reader.read_split("validation")
            test["event_id"] += 100
            test["time_ms"] += 10
            test.to_parquet(root / "data/test.parquet", index=False)
            manifest = reader.manifest
            manifest["splits"]["test"] = {"start": 15}
            manifest["files"]["test.parquet"] = sha256(root / "data/test.parquet")
            (root / "data/manifest.json").write_text(json.dumps(manifest))
            reader = ParquetReader(root / "data")
            pd.testing.assert_frame_equal(reader.read_split("test"), test)
            reader.validate()

    def test_full_audit_rejects_invalid_materialized_values(self) -> None:
        for column, value, message in (
            ("item_code", 999, "outside fitted vocabulary"),
            ("long_view", 2.0, "binary target"),
            ("train_duration_log1p", np.nan, "invalid values"),
            ("time_ms", 10, "Temporal boundaries"),
            ("warm_item", False, "warm_item disagrees"),
        ):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reader = prepare_kuairand(
                    raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
                )
                frame = reader.read_split("train")
                frame.loc[0, column] = value
                frame.to_parquet(root / "data/train.parquet", index=False)
                manifest = reader.manifest
                manifest["files"]["train.parquet"] = sha256(root / "data/train.parquet")
                (root / "data/manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    ParquetReader(root / "data").validate()

    def test_reader_rejects_event_only_format_and_wrong_column_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reader = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            frame = reader.read_split("train")
            frame["item_code"] = frame.item_code.astype(np.float32)
            frame.to_parquet(root / "data/train.parquet", index=False)
            reader.manifest["files"]["train.parquet"] = sha256(root / "data/train.parquet")
            with self.assertRaisesRegex(ValueError, "dtype"):
                reader.read_split("train", ["item_code"])
            reader.manifest["version"] = "kuairand-prepared-v3"
            (root / "data/manifest.json").write_text(json.dumps(reader.manifest))
            with self.assertRaisesRegex(ValueError, "regenerate"):
                ParquetReader(root / "data")

    def test_manifest_roles_and_temporal_bounds_cannot_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reader = prepare_kuairand(
                raw_fixture(root), root / "data", users=0, train_end=10, val_end=15
            )
            original = json.dumps(reader.manifest)
            for section, key, value, message in (
                ("feature_dtypes", "long_view", "float32", "feature column"),
                ("vocab_sizes", "item_code", 0, "feature column"),
                ("splits", "validation", {"start": 9, "end": 15}, "temporal protocol"),
                ("splits", "test", {"start": 14}, "temporal protocol"),
            ):
                manifest = json.loads(original)
                manifest[section][key] = value
                (root / "data/manifest.json").write_text(json.dumps(manifest))
                with self.subTest(section=section, key=key):
                    with self.assertRaisesRegex(ValueError, message):
                        ParquetReader(root / "data")


if __name__ == "__main__":
    unittest.main()
