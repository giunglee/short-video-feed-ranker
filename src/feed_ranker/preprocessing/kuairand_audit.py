"""Descriptive public-data audits; these functions never fit a model or encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = (
    "long_view",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)
NONNEGATIVE = {
    "user_id",
    "video_id",
    "time_ms",
    "duration_ms",
    "play_time_ms",
    "video_duration",
    "profile_stay_time",
    "comment_stay_time",
}


def read_csv_sample(paths: list[Path], max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Deterministic prefix, explicitly bounded per table; zero requests all rows."""
    if max_rows < 0 or not paths:
        raise ValueError("Require source files and nonnegative row limit")
    frames = []
    count = 0
    for path in sorted(paths):
        frame = pd.read_csv(path, nrows=None if max_rows == 0 else max_rows - count + 1)
        frames.append(frame)
        count += len(frame)
        if max_rows and count > max_rows:
            return pd.concat(frames, ignore_index=True).iloc[:max_rows].copy(), True
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("Cannot audit empty sources")
    return result, False


def profile_fields(frame: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for name in frame:
        column = frame[name]
        missing = column.isna()
        sentinel = column.astype("string").str.upper().isin(["UNKNOWN", "-124", "-124.0"])
        numeric = pd.to_numeric(column, errors="coerce")
        numeric_field = (
            pd.api.types.is_numeric_dtype(column) or name in NONNEGATIVE or name in LABELS
        )
        known = ~missing & ~sentinel
        invalid = (
            known & ~np.isfinite(numeric) if numeric_field else pd.Series(False, index=frame.index)
        )
        rule = "finite numeric" if numeric_field else "no domain rule asserted"
        if name in LABELS or name == "is_rand":
            invalid |= known & ~numeric.isin([0, 1])
            rule = "observed binary outside {0,1}"
        elif name in NONNEGATIVE:
            invalid |= known & numeric.lt(0)
            rule = "nonfinite or negative numeric"
        valid = known & ~invalid
        values = numeric[valid] if numeric_field else None
        frequencies = column.astype("string").fillna("<NULL>").value_counts().head(10)
        results.append(
            {
                "field": name,
                "dtype": str(column.dtype),
                "rows": len(frame),
                "null": int(missing.sum()),
                "sentinel": int(sentinel.sum()),
                "invalid_nonmissing": int(invalid.sum()),
                "invalid_rule": rule,
                "distinct_nonnull": int(column.nunique()),
                "zero": int(numeric.eq(0).sum()) if numeric_field else None,
                "distribution": None
                if values is None or values.empty
                else {
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "max": float(values.max()),
                },
                "top_values": {str(k): int(v) for k, v in frequencies.items()},
            }
        )
    return results


def profile_events(frame: pd.DataFrame, train_end: int, val_end: int) -> dict[str, Any]:
    required = {"user_id", "video_id", "time_ms", "tab", "is_rand", "long_view"}
    if not required <= set(frame) or train_end >= val_end:
        raise ValueError("Require event identity, traffic, long_view and ordered cutoffs")
    unique = frame.drop_duplicates()
    numeric = unique[list(required)].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(True, index=unique.index)
    for name in ("user_id", "video_id", "time_ms"):
        values = numeric[name]
        valid &= np.isfinite(values) & values.ge(0) & values.eq(np.floor(values))
    valid &= numeric.time_ms.gt(0)
    scoped = numeric.loc[valid & numeric.tab.eq(1) & numeric.is_rand.eq(0)]
    train = scoped.loc[scoped.time_ms.lt(train_end)]
    validation = scoped.loc[scoped.time_ms.ge(train_end) & scoped.time_ms.lt(val_end)]
    cohort_validation = validation.loc[validation.user_id.isin(train.user_id)]
    windows = {}
    for name, window in (
        ("train", train),
        ("validation_all", validation),
        ("validation_train_users", cohort_validation),
    ):
        warm = window.video_id.isin(train.video_id)
        valid_label = window.long_view.isin([0, 1])
        positive = window.long_view.eq(1)
        windows[name] = {
            "events": len(window),
            "users": int(window.user_id.nunique()),
            "items": int(window.video_id.nunique()),
            "warm_item_events": int(warm.sum()),
            "cold_item_events": int((~warm).sum()),
            "cold_user_events": int((~window.user_id.isin(train.user_id)).sum()),
            "valid_labels": int(valid_label.sum()),
            "invalid_labels": int((~valid_label).sum()),
            "positive_events": int(positive.sum()),
            "positive_pairs": len(window.loc[positive, ["user_id", "video_id"]].drop_duplicates()),
            "warm_positive_pairs": len(
                window.loc[positive & warm, ["user_id", "video_id"]].drop_duplicates()
            ),
        }
    traffic = []
    for name, source in (("raw", frame), ("deduplicated", unique)):
        for (tab, randomized), group in source.groupby(["tab", "is_rand"], dropna=False):
            traffic.append(
                {"view": name, "tab": str(tab), "is_rand": str(randomized), "events": len(group)}
            )
    labels = {}
    for name in LABELS:
        if name in unique:
            values = pd.to_numeric(unique[name], errors="coerce")
            valid_label = values.isin([0, 1])
            labels[name] = {
                "valid": int(valid_label.sum()),
                "invalid_or_missing": int((~valid_label).sum()),
                "positive_rate": float(values[valid_label].mean()) if valid_label.any() else None,
            }
    ties = scoped.groupby(["user_id", "time_ms"]).size()
    return {
        "raw_events": len(frame),
        "unique_events": len(unique),
        "duplicates_removed": len(frame) - len(unique),
        "invalid_identity_or_time": int((~valid).sum()),
        "traffic": traffic,
        "labels": labels,
        "repeated_pair_events": int(scoped.duplicated(["user_id", "video_id"]).sum()),
        "timestamp_tie_groups": int(ties.gt(1).sum()),
        "windows": windows,
        "events_after_validation": int(scoped.time_ms.ge(val_end).sum()),
        "time_convention": "UTC+8 explicit convention; timestamp groups are not split",
    }
