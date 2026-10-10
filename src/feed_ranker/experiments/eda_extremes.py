"""Event-tail and entity-level diagnostics; thresholds flag observations, not errors.

All split comparisons are descriptive. No labels are removed, reweighted, or
promoted into model-selection rules by these summaries.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tail_audit(frames: dict[str, pd.DataFrame], threshold: float = 5.0) -> pd.DataFrame:
    """Contrast common/tail events and their error shares at one fixed prediction."""
    prediction = frames["big train"].watch_ratio.median()
    rows = []
    for split, frame in frames.items():
        residual = frame.watch_ratio - prediction
        absolute_total = residual.abs().sum()
        squared_total = residual.pow(2).sum()
        tail = frame.watch_ratio.ge(threshold)
        for name, mask in [(f"ratio < {threshold:g}", ~tail), (f"ratio >= {threshold:g}", tail)]:
            selected = frame.loc[mask]
            rows.append(
                {
                    "split": split,
                    "segment": name,
                    "events": len(selected),
                    "event_fraction": len(selected) / len(frame),
                    "users": selected.user_id.nunique(),
                    "videos": selected.video_id.nunique(),
                    "missing_date_events": selected.date.isna().sum(),
                    "play_seconds_p50": selected.play_duration.median() / 1000,
                    "play_seconds_p90": selected.play_duration.quantile(0.9) / 1000,
                    "play_seconds_p99": selected.play_duration.quantile(0.99) / 1000,
                    "play_seconds_max": selected.play_duration.max() / 1000,
                    "video_seconds_p50": selected.video_duration.median() / 1000,
                    "video_seconds_p90": selected.video_duration.quantile(0.9) / 1000,
                    "ratio_p50": selected.watch_ratio.median(),
                    "absolute_error_share": residual.loc[mask].abs().sum() / absolute_total
                    if absolute_total
                    else np.nan,
                    "squared_error_share": residual.loc[mask].pow(2).sum() / squared_total
                    if squared_total
                    else np.nan,
                    "reference_prediction": prediction,
                }
            )
    return pd.DataFrame(rows)


def logging_checks(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check observable constraints, without assuming session or autoplay semantics."""
    rows = []
    for split, frame in frames.items():
        signature = [
            "user_id",
            "video_id",
            "timestamp",
            "play_duration",
            "video_duration",
            "watch_ratio",
        ]
        dated = frame.loc[frame.timestamp.notna()]
        expected = frame.play_duration / frame.video_duration.where(frame.video_duration > 0)
        error = (frame.watch_ratio - expected).abs()
        rows.append(
            {
                "split": split,
                "negative_play_duration": frame.play_duration.lt(0).sum(),
                "nonpositive_video_duration": frame.video_duration.le(0).sum(),
                "nonfinite_play_duration": (~np.isfinite(frame.play_duration)).sum(),
                "nonfinite_video_duration": (~np.isfinite(frame.video_duration)).sum(),
                "nonfinite_ratio": (~np.isfinite(frame.watch_ratio)).sum(),
                "missing_timestamp": frame.timestamp.isna().sum(),
                "repeated_event_signature_beyond_first": dated.duplicated(signature).sum(),
                "ratio_absolute_discrepancy_max": error.max(),
            }
        )
    return pd.DataFrame(rows)


def entity_statistics(frame: pd.DataFrame, entity: str, threshold: float = 5.0) -> pd.DataFrame:
    if entity not in ("user_id", "video_id"):
        raise ValueError("entity must be user_id or video_id")
    partner = "video_id" if entity == "user_id" else "user_id"
    groups = frame.groupby(entity, sort=True)
    result = groups.agg(
        events=("watch_ratio", "size"),
        partners=(partner, "nunique"),
        ratio_mean=("watch_ratio", "mean"),
        ratio_median=("watch_ratio", "median"),
        ratio_max=("watch_ratio", "max"),
        video_duration_mean_ms=("video_duration", "mean"),
        play_duration_mean_ms=("play_duration", "mean"),
    )
    result["tail_events"] = (
        frame.loc[frame.watch_ratio.ge(threshold)]
        .groupby(entity)
        .size()
        .reindex(result.index, fill_value=0)
    )
    result["tail_fraction"] = result.tail_events / result.events
    return result


def concentration_summary(entities: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for split, dimensions in entities.items():
        for entity, frame in dimensions.items():
            ordered = frame.sort_values("tail_events", ascending=False, kind="stable")
            top = ordered.head(max(1, int(np.ceil(len(ordered) * 0.01))))
            total = frame.tail_events.sum()
            rows.append(
                {
                    "split": split,
                    "entity": entity,
                    "entities": len(frame),
                    "entities_with_tail": frame.tail_events.gt(0).sum(),
                    "top_1pct_entity_count": len(top),
                    "top_1pct_tail_share": top.tail_events.sum() / total if total else np.nan,
                    "same_entities_all_event_share": top.events.sum() / frame.events.sum(),
                    "equal_entity_mean_ratio": frame.ratio_mean.mean(),
                    "events_per_entity_p50": frame.events.median(),
                    "events_per_entity_p99": frame.events.quantile(0.99),
                }
            )
    return pd.DataFrame(rows)


def plot_entity_distributions(entities: dict[str, dict[str, pd.DataFrame]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, entity in enumerate(["user_id", "video_id"]):
        for split, dimensions in entities.items():
            frame = dimensions[entity]
            for col, metric in enumerate(["events", "ratio_mean", "tail_fraction"]):
                values = np.sort(frame[metric].to_numpy())
                axes[row, col].plot(
                    values, np.arange(1, len(values) + 1) / len(values) * 100, label=split
                )
                axes[row, col].set(
                    ylabel=f"{entity}: cumulative entities (%)",
                    xlabel=metric,
                    title=f"{entity}: equal-entity distribution of {metric}",
                )
                if metric in ("events", "ratio_mean"):
                    axes[row, col].set_xscale("symlog", linthresh=1)
        axes[row, 0].legend(fontsize=7)
    axes[0, 0].set_xlabel("Recorded events per user (symlog; linear below 1)")
    axes[1, 0].set_xlabel("Recorded events per video (symlog; linear below 1)")
    for row in range(2):
        axes[row, 1].set_xlabel("Entity mean watch ratio (symlog; linear within ±1)")
        axes[row, 2].set_xlabel("Fraction of entity events with ratio >=5 (0–1)")
    fig.tight_layout()
    return fig


def plot_tail_concentration(entities: dict[str, dict[str, pd.DataFrame]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, entity in zip(axes, ["user_id", "video_id"]):
        for split, dimensions in entities.items():
            counts = dimensions[entity].tail_events.sort_values(ascending=False).to_numpy()
            if counts.sum() == 0:
                continue
            ax.plot(
                np.arange(len(counts) + 1) / len(counts) * 100,
                np.r_[0, counts.cumsum() / counts.sum() * 100],
                label=split,
            )
        ax.plot([0, 100], [0, 100], "--", color="gray", label="Uniform tail-event reference")
        ax.set(
            title=f"Tail-event concentration by {entity}",
            xlabel="Entities ranked by tail-event count (%)",
            ylabel="Cumulative events with ratio >=5 (%)",
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_tail_durations(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(len(frames), 2, figsize=(13, 10))
    for row, (split, frame) in enumerate(frames.items()):
        tail = frame.watch_ratio.ge(5)
        for col, feature in enumerate(["play_duration", "video_duration"]):
            for name, mask in [("ratio <5", ~tail), ("ratio >=5", tail)]:
                values = np.sort(frame.loc[mask, feature].to_numpy() / 1000)
                if not len(values):
                    continue
                # Deterministic quantiles thin only the displayed curve; statistics use all rows.
                indices = np.unique(
                    np.linspace(0, len(values) - 1, min(len(values), 1000)).astype(int)
                )
                axes[row, col].plot(values[indices], (indices + 1) / len(values) * 100, label=name)
            axes[row, col].set(
                title=f"{split}: {feature}",
                xlabel="Seconds (symlog; linear within ±1)",
                ylabel="Cumulative segment events (%)",
                xscale="symlog",
            )
            axes[row, col].set_xscale("symlog", linthresh=1)
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    return fig


def daily_tail_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in frames.items():
        dates = pd.to_datetime(frame.date.dropna().astype("int64").astype(str), format="%Y%m%d")
        calendar = pd.Index(
            pd.date_range(dates.min(), dates.max()).strftime("%Y%m%d").astype("int64"), name="date"
        )
        events = frame.groupby("date").size().reindex(calendar, fill_value=0)
        tail = (
            frame.loc[frame.watch_ratio.ge(5)]
            .groupby("date")
            .size()
            .reindex(calendar, fill_value=0)
        )
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "events": events,
                    "tail_events": tail,
                    "tail_fraction": tail / events.replace(0, np.nan),
                }
            ).reset_index()
        )
    return pd.concat(rows, ignore_index=True)


def plot_daily_tail(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    for split, group in table.groupby("split", sort=False):
        dates = pd.to_datetime(group.date.astype("int64").astype(str), format="%Y%m%d")
        ax.plot(dates, group.tail_fraction * 100, marker=".", label=split)
    ax.set(
        title="Daily frequency of watch_ratio >=5; gaps mean no dated rows",
        xlabel="Raw interaction date",
        ylabel="Recorded events that day with ratio >=5 (%)",
    )
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def duplicate_sensitivity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hypothetical signature deduplication across all labels; never modify input rows."""
    signature = [
        "user_id",
        "video_id",
        "timestamp",
        "play_duration",
        "video_duration",
        "watch_ratio",
    ]
    rows = []
    for split, frame in frames.items():
        repeated = frame.timestamp.notna() & frame.duplicated(signature)
        tail = frame.watch_ratio.ge(5)
        kept = frame.loc[~repeated]
        rows.append(
            {
                "split": split,
                "original_events": len(frame),
                "removed_events": int(repeated.sum()),
                "removed_tail_events": int((repeated & tail).sum()),
                "fraction_tail_removed": (repeated & tail).sum() / tail.sum()
                if tail.any()
                else np.nan,
                "fraction_common_removed": (repeated & ~tail).sum() / (~tail).sum()
                if (~tail).any()
                else np.nan,
                "original_tail_fraction": tail.mean(),
                "deduplicated_tail_fraction": kept.watch_ratio.ge(5).mean(),
                "original_label_mean": frame.watch_ratio.mean(),
                "deduplicated_label_mean": kept.watch_ratio.mean(),
                "original_label_max": frame.watch_ratio.max(),
                "deduplicated_label_max": kept.watch_ratio.max(),
            }
        )
    return pd.DataFrame(rows)
