"""Candidate feature construction requires training metadata, never watch labels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from feed_ranker.experiments.recall_rank import candidate_features


class RecallRankTests(unittest.TestCase):
    def test_training_metadata_preserves_candidate_identity_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame({"video_id": [10, 20], "median": [10000.0, 30000.0]}).to_parquet(
                directory / "item_duration.parquet", index=False
            )
            data = SimpleNamespace(
                directory=directory,
                users=np.array([7, 8]),
                items=np.array([10, 20]),
                pools=SimpleNamespace(users=np.array([0, 0, 1]), items=np.array([1, 0, 1])),
            )
            actual = candidate_features(data)
            self.assertEqual(actual.columns.tolist(), ["user_id", "video_id", "video_duration"])
            self.assertEqual(actual.video_id.tolist(), [20, 10, 20])
            self.assertEqual(actual.user_id.tolist(), [7, 7, 8])
            self.assertEqual(actual.video_duration.tolist(), [30000.0, 10000.0, 30000.0])
            data.items = np.array([10, 30])
            with self.assertRaisesRegex(ValueError, "Missing/invalid"):
                candidate_features(data)


if __name__ == "__main__":
    unittest.main()
