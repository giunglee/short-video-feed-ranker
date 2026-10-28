"""Run public-fixture commands and matched studies through actual fitted artifacts."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_prepared_data import raw_fixture


def configuration(data: Path) -> dict:
    return {
        "data": str(data),
        "model": {
            "features": [
                "train_duration_log1p",
                "train_duration_missing",
                "user_code",
                "item_code",
            ],
            "hidden": [4],
        },
        "tasks": [{"output": "long_view", "target": "long_view"}],
        "loading": {"batch_size": 3},
        "training": {
            "epochs": 1,
            "seed": 42,
            "selection_metric": "long_view/roc_auc",
            "selection_mode": "max",
        },
    }


class WorkflowTests(unittest.TestCase):
    def test_commands_work_without_external_study_imports_or_scoring_event_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = raw_fixture(root)
            env = {
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                "OMP_NUM_THREADS": "1",
            }

            def command(*arguments: str) -> None:
                completed = subprocess.run(
                    [sys.executable, "-m", "feed_ranker.cli", *arguments],
                    cwd=root,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            command(
                "prepare",
                "--raw",
                str(raw),
                "--out",
                "data",
                "--users",
                "0",
                "--train-end",
                "10",
                "--val-end",
                "15",
            )
            command("validate-data", "--data", "data")
            (root / "job.json").write_text(json.dumps(configuration(root / "data")))
            command("train", "--config", "job.json", "--out", "deep")
            command(
                "fit-recall", "--data", "data", "--out", "recall", "--epochs", "1", "--seed", "42"
            )
            arguments = (
                "--ranker",
                "deep/model",
                "--recall",
                "recall/popularity",
                "recall/cosine",
                "recall/mf",
                "--budget",
                "2",
            )
            command("evaluate", *arguments, "--data", "data", "--out", "evaluation")
            metrics = json.loads((root / "evaluation/metrics.json").read_text())
            self.assertEqual(metrics["metrics"]["queries"], 2)
            (root / "data").rename(root / "unavailable")
            (root / "requests.json").write_text(
                json.dumps([{"query_id": "q", "user_id": 1, "items": [10, 20, 30]}])
            )
            command(
                "recommend", *arguments, "--requests", "requests.json", "--out", "recommendation"
            )
            self.assertTrue((root / "recommendation/recommendation_trace.parquet").exists())


if __name__ == "__main__":
    unittest.main()
