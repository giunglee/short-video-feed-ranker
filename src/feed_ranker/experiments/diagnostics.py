"""Local validation traces and cohort metrics from public prepared data.

Activity thresholds use training users only. Cohort metrics are descriptive;
small cohorts and sampled concordance do not establish statistically clear gains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.kuairec_dataset import KuaiRecDataset
from feed_ranker.training.evaluation import predict_regression
from feed_ranker.training.metrics import user_xauc, xauc


def activity_groups(counts: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    """Zero/unseen is separate; tied counts always receive the same group."""
    low, high = thresholds
    return pd.Series(
        np.select(
            [counts.eq(0), counts.le(low), counts.le(high)],
            ["unseen", "low", "medium"],
            default="high",
        ),
        index=counts.index,
    )


@torch.no_grad()
def export_validation_trace(
    model: nn.Module,
    dataset: KuaiRecDataset,
    checkpoint: Path,
    output: Path,
    *,
    feature_allowlist: list[str],
    device: str = "cpu",
    metric_seed: int = 0,
    user_profiles: Path | None = None,
    split: str = "val",
    label_cap: float | None = None,
) -> dict[str, Any]:
    """Export scores/labels, raw entity IDs and explicitly allowed prepared features.

    Optional public profiles add only user_active_degree, for diagnosis rather than
    model input. Caller must restore the supplied checkpoint into model first.
    """
    if split not in ("val", "test"):
        raise ValueError("Trace split must be val or test")
    if label_cap is not None and (not np.isfinite(label_cap) or label_cap <= 0):
        raise ValueError("label_cap must be finite and positive")
    if set(feature_allowlist) - set(dataset.config.feature_cols):
        raise ValueError("Trace allowlist contains non-model features")
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    chunks = []
    for batch in dataset.loader(split, 8192):
        size = len(batch["label"])
        frame = pd.DataFrame(
            {
                "raw_user_id": batch["raw_user_id"].numpy(),
                "raw_video_id": batch["raw_video_id"].numpy(),
                "label": batch["label"].numpy(),
                "prediction": predict_regression(model, batch["feature"], size, device)
                .cpu()
                .numpy(),
            }
        )
        for name in feature_allowlist:
            values = batch["feature"][name].cpu().numpy()
            frame["feature__" + name] = (
                [v[v >= 0].tolist() for v in values] if values.ndim == 2 else values
            )
        chunks.append(frame)
    trace = pd.concat(chunks, ignore_index=True)
    trace_name = "validation" if split == "val" else "test"
    trace.index.name = f"{trace_name}_row"
    train = pd.read_parquet(
        dataset.directory / "train.parquet", columns=["raw_user_id", "raw_video_id"]
    )
    user_counts = train.groupby("raw_user_id").size()
    video_counts = train.groupby("raw_video_id").size()
    thresholds = tuple(float(v) for v in user_counts.quantile([1 / 3, 2 / 3]))
    trace["train_user_events"] = trace.raw_user_id.map(user_counts).fillna(0).astype("int64")
    trace["train_video_events"] = trace.raw_video_id.map(video_counts).fillna(0).astype("int64")
    trace["train_activity"] = activity_groups(trace.train_user_events, thresholds)
    trace["video_support"] = pd.cut(
        trace.train_video_events, [-1, 0, 10, 100, 1000, np.inf]
    ).astype(str)
    trace["label_tail"] = np.where(trace.label.ge(5), "ratio >=5", "ratio <5")
    cohorts = ["train_activity", "video_support", "label_tail"]
    if "video_duration" in feature_allowlist:
        trace["duration_band"] = pd.cut(
            trace.feature__video_duration / 1000,
            [0, 5, 10, 20, 40, 60, np.inf],
            include_lowest=True,
        ).astype(str)
        cohorts.append("duration_band")
    if user_profiles is not None:
        profiles = pd.read_csv(user_profiles, usecols=["user_id", "user_active_degree"])
        if profiles.user_id.duplicated().any():
            raise ValueError("Profile IDs must be unique")
        trace["snapshot_activity"] = trace.raw_user_id.map(
            profiles.set_index("user_id").user_active_degree
        ).fillna("<missing>")
        cohorts.append("snapshot_activity")
    trace["evaluation_label"] = (
        trace.label if label_cap is None else trace.label.clip(upper=label_cap)
    )
    trace.to_parquet(output / f"{trace_name}_trace.parquet", index=True)
    if label_cap is not None:
        write_json(
            output / "prediction_summary.json",
            {
                "raw_label_quantiles": {
                    str(q): float(v)
                    for q, v in trace.label.quantile([0, 0.5, 0.9, 0.99, 1]).items()
                },
                "prediction_quantiles": {
                    str(q): float(v)
                    for q, v in trace.prediction.quantile([0, 0.5, 0.9, 0.99, 1]).items()
                },
                "fraction_prediction_below_zero": float(trace.prediction.lt(0).mean()),
                "fraction_prediction_above_cap": float(trace.prediction.gt(label_cap).mean()),
                "fraction_labels_at_cap": float(trace.label.ge(label_cap).mean()),
            },
        )
    results = []
    for dimension in ["overall", *cohorts]:
        groups = (
            [("all", trace)]
            if dimension == "overall"
            else trace.groupby(dimension, observed=True, sort=True)
        )
        for name, frame in groups:
            pred = torch.tensor(frame.prediction.to_numpy())
            label = torch.tensor(frame.evaluation_label.to_numpy())
            users = torch.tensor(frame.raw_user_id.to_numpy())
            results.append(
                {
                    "dimension": dimension,
                    "cohort": str(name),
                    "rows": len(frame),
                    "users": frame.raw_user_id.nunique(),
                    "mae": float((pred - label).abs().double().mean()),
                    "bias": float((pred - label).double().mean()),
                    "xauc": xauc(pred, label, seed=metric_seed),
                    **user_xauc(pred, label, users, seed=metric_seed),
                }
            )
    pd.DataFrame(results).to_csv(output / "cohort_metrics.csv", index=False)
    manifest = {
        "checkpoint_sha256": sha256(checkpoint),
        "data_manifest_sha256": sha256(dataset.directory / "meta.json"),
        "split": split,
        "label_cap": label_cap,
        "feature_allowlist": feature_allowlist,
        "feature_encoding": "prepared IDs/lists; vocabulary in data manifest; duration milliseconds before model log1p",
        "activity_thresholds": thresholds,
        "activity_basis": "tertiles across distinct training users by deduplicated event count",
        "metric_seed": metric_seed,
        "global_pairs": 100000,
        "pairs_per_user": 256,
        "snapshot_profile_sha256": sha256(user_profiles) if user_profiles else None,
        "limitations": [
            "Snapshot activity is not point-in-time verified",
            "Trace is local, not a serving log",
            "Cohort XAUC compares rows within each cohort; user_XAUC macro-averages eligible users",
            "Capped tail-only cohorts may have no unequal labels: XAUC is then undefined, not zero",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest
