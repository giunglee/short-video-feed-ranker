"""Fresh raw audit retains missingness and population distinctions."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from test_prepared_data import raw_fixture

from experiments.eda_kuairand import audit
from feed_ranker.data.profiling import profile_fields


class RawAuditTests(unittest.TestCase):
    def test_fresh_audit_and_explicit_prefix_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = raw_fixture(root) / "log_standard_fixture.csv"
            frame = pd.read_csv(raw)
            pd.concat([frame, frame.iloc[[0]]]).to_csv(raw, index=False)
            result = audit([raw], root / "audit", train_end=10, val_end=15)
            self.assertEqual(result["events"]["duplicates_removed"], 1)
            self.assertEqual(
                result["events"]["windows"]["validation_train_users"]["warm_item_events"], 3
            )
            self.assertEqual(
                result["events"]["windows"]["validation_train_users"]["cold_item_events"], 1
            )
            self.assertTrue((root / "audit" / "cohorts.png").exists())
            self.assertFalse(result["tables"]["events"]["truncated"])
            bounded = audit([raw], root / "bounded", max_rows=3, train_end=10, val_end=15)
            self.assertTrue(bounded["tables"]["events"]["truncated"])
            with self.assertRaises(FileExistsError):
                audit([raw], root / "audit")

    def test_null_sentinel_and_invalid_are_not_conflated(self) -> None:
        fields = profile_fields(
            pd.DataFrame(
                {"long_view": [None, -124, 2, 0, 1], "duration_ms": [None, -124, -1, 0, 100]}
            )
        )
        for field in fields:
            self.assertEqual(field["null"], 1)
            self.assertEqual(field["sentinel"], 1)
            self.assertEqual(field["invalid_nonmissing"], 1)
            self.assertEqual(field["zero"], 1)


if __name__ == "__main__":
    unittest.main()
