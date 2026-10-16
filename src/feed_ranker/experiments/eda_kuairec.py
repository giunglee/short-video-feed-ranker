"""Descriptive big/small audit; small-label summaries must not select model features.

Prefix samples are smoke checks, not representative estimates. Duration cohorts
use milliseconds; repeated big-matrix events remain separate observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.schema import KEYS

AUDIT_COLUMNS = [*KEYS, "watch_ratio", "video_duration", "play_duration", "date", "timestamp"]
DURATION_BINS_MS = [0, 5_000, 10_000, 20_000, 40_000, 60_000, np.inf]


def audit_matrices(raw_dir: Path, output: Path, rows: int | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Use a new EDA output path")
    if rows is not None and rows <= 0:
        raise ValueError("rows must be positive")
    reports = {}
    pairs = {}
    for name in ("big", "small"):
        reports[name], pairs[name] = analyze(raw_dir / f"{name}_matrix.csv", rows)
        print(f"{name}: {reports[name]['rows']} rows audited", flush=True)

    result = {
        "purpose": "Descriptive audit, not test-driven model selection",
        "prefix_rows": rows,
        "matrices": reports,
        **_coverage_summary(pairs["big"], pairs["small"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def analyze(path: Path, rows: int | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(path, usecols=AUDIT_COLUMNS, nrows=rows)
    report = {
        "rows": len(frame),
        "users": int(frame.user_id.nunique()),
        "items": int(frame.video_id.nunique()),
        "missing_date": int(frame.date.isna().sum()),
        "missing_timestamp": int(frame.timestamp.isna().sum()),
        **_label_summary(frame.watch_ratio),
        **_pair_summary(frame),
        "mean_of_user_label_means": float(frame.groupby("user_id").watch_ratio.mean().mean()),
        "ratio_formula_max_abs_difference": _ratio_error(frame),
        "duration_ms_cohorts": _duration_cohorts(frame),
        "source_sha256": sha256(path),
    }
    return report, frame[KEYS]


def _label_summary(labels: pd.Series) -> dict[str, Any]:
    values = labels.to_numpy()
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No finite labels")
    quantiles = np.quantile(finite, [0, 0.5, 0.9, 0.99, 1]).tolist()
    return {
        "invalid_labels": int((~np.isfinite(values)).sum()),
        "negative_labels": int((finite < 0).sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "quantiles": dict(zip(["min", "p50", "p90", "p99", "max"], quantiles)),
        "zero_fraction": float((finite == 0).mean()),
        "above_one_fraction": float((finite > 1).mean()),
    }


def _pair_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    counts = frame.groupby(KEYS, sort=False).size()
    return {
        "distinct_pairs": len(counts),
        "repeated_pairs": int((counts > 1).sum()),
        "repeat_rows_beyond_first": int((counts - 1).sum()),
        "mean_of_pair_label_means": float(frame.groupby(KEYS).watch_ratio.mean().mean()),
    }


def _ratio_error(frame: pd.DataFrame) -> float | None:
    expected = frame.play_duration / frame.video_duration.replace(0, np.nan)
    valid = np.isfinite(expected) & np.isfinite(frame.watch_ratio)
    if not valid.any():
        return None
    return float((frame.loc[valid, "watch_ratio"] - expected[valid]).abs().max())


def _duration_cohorts(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    duration = pd.cut(frame.video_duration, DURATION_BINS_MS)
    cohorts = frame.groupby(duration, observed=True).watch_ratio.agg(["count", "mean", "median"])
    return {
        str(interval): {name: float(value) for name, value in metrics.items()}
        for interval, metrics in cohorts.to_dict("index").items()
    }


def _coverage_summary(big: pd.DataFrame, small: pd.DataFrame) -> dict[str, int]:
    big_pairs = pd.MultiIndex.from_frame(big)
    small_pairs = pd.MultiIndex.from_frame(small)
    return {
        "pair_overlap": int(small_pairs.isin(big_pairs).sum()),
        "small_users_absent_big": len(set(small.user_id) - set(big.user_id)),
        "small_items_absent_big": len(set(small.video_id) - set(big.video_id)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/kuairec"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, help="Prefix smoke sample only; not representative")
    args = parser.parse_args()
    if args.rows is not None and args.rows <= 0:
        parser.error("--rows must be positive")
    return args


def main() -> None:
    args = _parse_args()
    audit_matrices(args.raw_dir, args.output, args.rows)


if __name__ == "__main__":
    main()
