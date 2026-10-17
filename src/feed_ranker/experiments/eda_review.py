"""Descriptive KuaiRec EDA for notebook review, with train-only feature screening.

Cross scores are associations in observed data, not causal effects or estimates of
model improvement. Profile snapshots require a separate availability review.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feed_ranker.data.kuairec_preprocessor import split_pairs
from feed_ranker.data.schema import KEYS

DURATION_EDGES = [0, 5000, 10000, 20000, 40000, 60000, np.inf]
DURATION_LABELS = ["0–5s", "5–10s", "10–20s", "20–40s", "40–60s", ">60s"]
PROFILE_CANDIDATES = [
    "gender",
    "age_range",
    "user_active_degree",
    "register_days_range",
    "follow_user_num_range",
    "fans_user_num_range",
    "fre_city_level",
    "platform",
]


def load_matrices(raw: Path) -> dict[str, pd.DataFrame]:
    columns = [*KEYS, "watch_ratio", "video_duration", "play_duration", "date", "timestamp"]
    big = pd.read_csv(raw / "big_matrix.csv", usecols=columns)
    held_out = split_pairs(big, seed=2026, val_ratio=0.1)
    small = pd.read_csv(raw / "small_matrix.csv", usecols=columns)
    return {
        "big train": big.loc[~held_out].copy(),
        "big validation": big.loc[held_out].copy(),
        "small (descriptive only)": small,
    }


def label_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        y = frame.watch_ratio
        rows.append(
            {
                "split": name,
                "events": len(frame),
                "users": frame.user_id.nunique(),
                "videos": frame.video_id.nunique(),
                "missing_date": frame.date.isna().sum(),
                "invalid_label": (~np.isfinite(y)).sum(),
                "negative_label": (y < 0).sum(),
                "zero_fraction": (y == 0).mean(),
                "above_one_fraction": (y > 1).mean(),
                "mean": y.mean(),
                "median": y.median(),
                "p90": y.quantile(0.90),
                "p99": y.quantile(0.99),
                "p999": y.quantile(0.999),
                "max": y.max(),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def plot_labels(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.linspace(0, 5, 101)
    thresholds = np.geomspace(0.01, 1000, 180)
    for name, frame in frames.items():
        y = frame.watch_ratio.to_numpy()
        counts, _ = np.histogram(y, bins=bins)
        axes[0].plot((bins[1:] + bins[:-1]) / 2, counts / len(y) * 100, label=name)
        ordered = np.sort(y)
        survival = (len(y) - np.searchsorted(ordered, thresholds, side="right")) / len(y)
        axes[1].plot(thresholds, np.where(survival > 0, survival * 100, np.nan), label=name)
    axes[0].set(
        title="Watch-ratio body: values >5 outside this view",
        xlabel="Watch ratio (unitless)",
        ylabel="All events per 0.05-wide bin (%)",
    )
    axes[1].set(
        title="Full tail: fraction strictly above threshold",
        xlabel="Watch-ratio threshold (log scale)",
        ylabel="Events above threshold (%, log scale)",
        xscale="log",
        yscale="log",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def tail_sensitivity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    median = frames["big train"].watch_ratio.median()
    rows = []
    for name, frame in frames.items():
        y = frame.watch_ratio.to_numpy()
        error = np.abs(y - median)
        for cutoff in [1, 2, 5, 10, 20, 100]:
            mask = y > cutoff
            rows.append(
                {
                    "split": name,
                    "threshold": cutoff,
                    "event_fraction": mask.mean(),
                    "absolute_error_share": error[mask].sum() / error.sum(),
                    "mean_after_capping_for_diagnosis_only": np.minimum(y, cutoff).mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_tail_sensitivity(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, rows in table.groupby("split", sort=False):
        axes[0].plot(rows.threshold, rows.event_fraction * 100, marker="o", label=name)
        axes[1].plot(rows.threshold, rows.absolute_error_share * 100, marker="o", label=name)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Watch ratio > threshold (log scale)")
        ax.legend(fontsize=8)
    axes[0].set(title="Tail support", ylabel="Events (%)")
    axes[1].set(
        title="Tail share of absolute error: train-median predictor",
        ylabel="Total absolute error (%)",
    )
    fig.tight_layout()
    return fig


def duration_groups(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.cut(frame.video_duration, DURATION_EDGES, labels=DURATION_LABELS, include_lowest=True)
        .astype("string")
        .fillna("missing/negative")
    )


def duration_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        grouped = frame.groupby(duration_groups(frame), observed=True).watch_ratio.agg(
            ["count", "mean", "median"]
        )
        grouped["fraction"] = grouped["count"] / len(frame)
        grouped["split"] = name
        rows.append(grouped.rename_axis("duration").reset_index())
    return pd.concat(rows, ignore_index=True)


def plot_duration(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, rows in table.groupby("split", sort=False):
        rows = rows.set_index("duration").reindex(DURATION_LABELS)
        for ax, metric in zip(axes, ["fraction", "mean", "median"]):
            ax.plot(DURATION_LABELS, rows[metric], marker="o", label=name)
            ax.tick_params(axis="x", rotation=40)
            ax.set(
                xlabel="Video duration (seconds)", ylabel=metric, title=f"Duration cohort: {metric}"
            )
    axes[0].set_ylabel("Fraction of events (0–1)")
    axes[1].set_ylabel("Mean watch ratio (unitless)")
    axes[2].set_ylabel("Median watch ratio (unitless)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_daily(frames: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, frame in frames.items():
        daily = frame.groupby("date").watch_ratio.agg(["count", "mean"])
        daily = daily.reindex(_calendar_days(frame.date))
        daily["count"] = daily["count"].fillna(0)
        dates = pd.to_datetime(daily.index.astype("int64").astype(str), format="%Y%m%d")
        axes[0].plot(dates, daily["mean"], label=name)
        axes[1].plot(dates, daily["count"], label=name)
    axes[0].set(
        title="Daily label means (not a temporal evaluation split)",
        ylabel="Mean watch ratio (unitless)",
    )
    axes[1].set(
        title="Number of recorded rows on each date (missing-date rows excluded)",
        ylabel="Recorded rows per day",
        xlabel="Interaction date",
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].ticklabel_format(axis="y", style="plain")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def repetition_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = frame.groupby(KEYS, sort=False).watch_ratio.agg(["size", "mean", "min", "max"])
    groups = pd.cut(
        pairs["size"], [0, 1, 2, 3, 5, 10, np.inf], labels=["1", "2", "3", "4–5", "6–10", ">10"]
    )
    support = pairs.groupby(groups, observed=False).agg(
        pairs=("size", "size"), events=("size", "sum"), mean_range=("max", "mean")
    )
    ranges = (pairs["max"] - pairs["min"]).groupby(groups, observed=False).mean()
    support["mean_within_pair_label_range"] = ranges
    support = support.drop(columns="mean_range")
    weights = pd.DataFrame(
        {
            "weighting": ["event weighted", "equal pair weight", "equal user weight"],
            "mean_watch_ratio": [
                frame.watch_ratio.mean(),
                pairs["mean"].mean(),
                frame.groupby("user_id").watch_ratio.mean().mean(),
            ],
        }
    )
    return support, weights


def load_profiles(raw: Path, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw profile values take precedence, matching preprocessor column precedence."""
    raw_users = pd.read_csv(raw / "user_features_raw.csv").set_index("user_id")
    encoded = (
        pd.read_csv(raw / "user_features.csv")
        .rename(columns={f"onehot_feat{i}": f"user_onehot_feat{i}" for i in range(18)})
        .set_index("user_id")
    )
    if not raw_users.index.is_unique or not encoded.index.is_unique:
        raise ValueError("Profile IDs must be unique")
    supplemental = [name for name in encoded.columns if name not in raw_users.columns]
    profiles = raw_users.join(encoded[supplemental], how="outer").reindex(
        sorted(train.user_id.unique())
    )
    counts = train.groupby("user_id").size().reindex(profiles.index)
    inventory = []
    for name in profiles:
        missing = profiles[name].isna() | profiles[name].astype("string").str.strip().eq("").fillna(
            False
        )
        inventory.append(
            {
                "feature": name,
                "source": "user_features_raw.csv" if name in raw_users else "user_features.csv",
                "levels": profiles[name].nunique(),
                "missing_user_fraction": missing.mean(),
                "missing_event_fraction": counts[missing].sum() / counts.sum(),
                "availability": "snapshot timing unresolved; not approved for model use",
            }
        )
        profiles[name] = profiles[name].astype("string").mask(missing, "<missing>")
    return profiles, pd.DataFrame(inventory)


def user_duration_cells(train: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Sufficient sums for exact event-weighted means, without expanding profile joins."""
    columns = train[["user_id", "watch_ratio"]].copy()
    columns["duration_band"] = duration_groups(train)
    columns["period"] = np.where(train.date <= 20200720, "through July 20", "after July 20")
    columns.loc[train.date.isna(), "period"] = "undated"
    cells = (
        columns.groupby(["user_id", "duration_band", "period"], observed=True)
        .watch_ratio.agg(n="size", total="sum")
        .reset_index()
    )
    return cells.merge(profiles.reset_index(), on="user_id", how="left", validate="many_to_one")


def category_lifts(cells: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    mean = cells.total.sum() / cells.n.sum()
    for column in columns:
        grouped = cells.groupby(column, observed=True).agg(
            events=("n", "sum"), total=("total", "sum"), users=("user_id", "nunique")
        )
        grouped["mean"] = grouped.total / grouped.events
        grouped["label_lift"] = grouped["mean"] / mean
        grouped["feature"] = column
        rows.append(grouped.drop(columns="total").rename_axis("level").reset_index())
    return pd.concat(rows, ignore_index=True)


def cross_statistics(cells: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Legacy multiplicative lift plus additive departure, using observed marginals.

    Neither reference fits an additive model. Correlated inputs, selection and
    omitted features can create departures even without useful model interactions.
    """
    if cells.empty:
        raise ValueError("Cannot summarize empty cells")
    overall = cells.total.sum() / cells.n.sum()
    mean_a = cells.groupby(a, observed=True)[["n", "total"]].sum().eval("total / n")
    mean_b = cells.groupby(b, observed=True)[["n", "total"]].sum().eval("total / n")
    joint = (
        cells.groupby([a, b], observed=True)
        .agg(events=("n", "sum"), total=("total", "sum"), users=("user_id", "nunique"))
        .reset_index()
    )
    joint["mean"] = joint.total / joint.events
    expected_product = joint[a].map(mean_a).astype(float) * joint[b].map(mean_b).astype(float)
    joint["interaction_lift"] = joint["mean"] * overall / expected_product.replace(0, np.nan)
    joint["additive_departure"] = (
        joint["mean"]
        - joint[a].map(mean_a).astype(float)
        - joint[b].map(mean_b).astype(float)
        + overall
    )
    return joint.drop(columns="total")


def scan_crosses(
    cells: pd.DataFrame, columns: list[str], min_events: int = 2000, min_users: int = 50
) -> pd.DataFrame:
    tables = []
    for a, b in combinations(columns, 2):
        table = cross_statistics(cells, a, b)
        table["feature_a"], table["feature_b"] = a, b
        table = table.rename(columns={a: "level_a", b: "level_b"})
        for period, suffix in [("through July 20", "early"), ("after July 20", "late")]:
            part = cross_statistics(cells.loc[cells.period.eq(period)], a, b)
            part = part.rename(
                columns={
                    a: "level_a",
                    b: "level_b",
                    **{
                        c: f"{c}_{suffix}"
                        for c in ["events", "users", "interaction_lift", "additive_departure"]
                    },
                }
            )
            table = table.merge(
                part.drop(columns="mean"),
                on=["level_a", "level_b"],
                how="left",
                validate="one_to_one",
            )
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result["supported"] = (result.events >= min_events) & (result.users >= min_users)
    result["period_supported"] = (
        result.supported
        & (result.events_early >= min_events)
        & (result.events_late >= min_events)
        & (result.users_early >= min_users)
        & (result.users_late >= min_users)
    )
    result["abs_log_lift"] = np.abs(
        np.log(result.interaction_lift.where(result.interaction_lift > 0))
    )
    return result.sort_values("abs_log_lift", ascending=False)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    r, k = table.shape
    if n <= 1 or min(r, k) < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    phi = ((table - expected) ** 2 / expected).sum() / n
    corrected = max(0.0, phi - (k - 1) * (r - 1) / (n - 1))
    denominator = min(r - 1 - (r - 1) ** 2 / (n - 1), k - 1 - (k - 1) ** 2 / (n - 1))
    return float(np.sqrt(corrected / denominator)) if denominator > 0 else 0.0


def redundancy(profiles: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    aliases = []
    for a, b in combinations(columns, 2):
        matrix.loc[a, b] = matrix.loc[b, a] = cramers_v(profiles[a], profiles[b])
        pairs = profiles[[a, b]].drop_duplicates()
        if len(pairs) == profiles[a].nunique() == profiles[b].nunique():
            aliases.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "mapping": "one-to-one on observed train users; missing treated as category",
                }
            )
    return matrix, pd.DataFrame(aliases, columns=["feature_a", "feature_b", "mapping"])


def plot_matrix(
    matrix: pd.DataFrame, title: str, cmap: str = "viridis", center: bool = False
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.38), max(5, len(matrix) * 0.34)))
    values = matrix.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), 1e-8) if center else 1
    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
        vmin=-limit if center else 0,
        vmax=limit,
    )
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set(
        title=title, xlabel=matrix.columns.name or "Feature", ylabel=matrix.index.name or "Feature"
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Additive departure (watch-ratio units)"
        if center
        else "Bias-corrected Cramér’s V (0–1)",
    )
    fig.tight_layout()
    return fig


def plot_cross_screen(table: pd.DataFrame) -> plt.Figure:
    eligible = (
        table.loc[table.period_supported & np.isfinite(table.abs_log_lift)].head(18).iloc[::-1]
    )
    if eligible.empty:
        raise ValueError("No cells meet the declared cross support thresholds")
    labels = [
        f"{r.feature_a}={r.level_a} × {r.feature_b}={r.level_b}" for r in eligible.itertuples()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(eligible) * 0.4)), sharey=True)
    for suffix, marker in [("early", "o"), ("late", "x")]:
        axes[0].scatter(
            np.log2(eligible[f"interaction_lift_{suffix}"]),
            range(len(eligible)),
            marker=marker,
            label=suffix,
        )
    axes[0].axvline(0, color="gray", lw=1)
    axes[0].set_yticks(range(len(eligible)), labels=labels, fontsize=8)
    axes[0].set(
        title="Largest full-train deviations: descriptive screening only",
        xlabel="log2(joint lift / product of marginal lifts)",
        ylabel="Category combination",
    )
    axes[0].legend()
    axes[1].barh(range(len(eligible)), eligible.users, color="steelblue")
    axes[1].set(
        title="Distinct-user support per combination", xlabel="Distinct train users (not events)"
    )
    fig.tight_layout()
    return fig


def daily_duration_summary(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily event-weighted measurements and train-only duration-composition reference.

    The reference fixes each bucket's mean to its all-train value and varies only
    daily bucket proportions. It is descriptive, not a forecast or causal effect.
    Undated events stay in other analyses but cannot be assigned to a day here.
    """
    summaries = []
    for name, frame in frames.items():
        daily = frame.groupby("date").agg(
            events=("watch_ratio", "size"),
            users=("user_id", "nunique"),
            videos=("video_id", "nunique"),
            label_mean=("watch_ratio", "mean"),
            duration_mean_ms=("video_duration", "mean"),
            duration_median_ms=("video_duration", "median"),
            duration_p10_ms=("video_duration", lambda x: x.quantile(0.1)),
            duration_p90_ms=("video_duration", lambda x: x.quantile(0.9)),
        )
        daily = daily.reindex(_calendar_days(frame.date))
        daily[["events", "users", "videos"]] = daily[["events", "users", "videos"]].fillna(0)
        daily["split"] = name
        summaries.append(daily.reset_index())
    train = frames["big train"].copy()
    train["duration_band"] = duration_groups(train)
    bucket_means = train.groupby("duration_band", observed=True).watch_ratio.mean()
    counts = pd.crosstab(train.date, train.duration_band).reindex(
        _calendar_days(train.date), fill_value=0
    )
    shares = counts.div(counts.sum(axis=1), axis=0)
    reference = shares.mul(bucket_means.reindex(shares.columns), axis=1).sum(axis=1, min_count=1)
    result = pd.concat(summaries, ignore_index=True)
    result["duration_mix_reference"] = np.nan
    selected = result["split"].eq("big train")
    result.loc[selected, "duration_mix_reference"] = result.loc[selected, "date"].map(reference)
    return result, shares


def plot_daily_duration(summary: pd.DataFrame, shares: pd.DataFrame) -> plt.Figure:
    """Separate duration levels, distribution mix, and label-composition reference."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for name, rows in summary.groupby("split", sort=False):
        dates = pd.to_datetime(rows.date.astype("int64").astype(str), format="%Y%m%d")
        axes[0, 0].plot(dates, rows.duration_mean_ms / 1000, label=name)
    axes[0, 0].set(title="Daily mean duration of observed events", ylabel="Mean duration (seconds)")
    axes[0, 0].legend(fontsize=8)
    train = summary.loc[summary["split"].eq("big train")]
    dates = pd.to_datetime(train.date.astype("int64").astype(str), format="%Y%m%d")
    axes[0, 1].fill_between(
        dates,
        train.duration_p10_ms / 1000,
        train.duration_p90_ms / 1000,
        alpha=0.2,
        label="p10–p90",
    )
    axes[0, 1].plot(dates, train.duration_median_ms / 1000, label="Median")
    axes[0, 1].plot(dates, train.duration_mean_ms / 1000, label="Mean", linestyle="--")
    axes[0, 1].set(title="Big train: daily duration distribution", ylabel="Duration (seconds)")
    axes[0, 1].legend(fontsize=8)
    bands = [name for name in DURATION_LABELS if name in shares.columns]
    bands += [name for name in shares.columns if name not in bands]
    share_dates = pd.to_datetime(shares.index.astype("int64").astype(str), format="%Y%m%d")
    axes[1, 0].stackplot(
        share_dates, *[shares[name].to_numpy() * 100 for name in bands], labels=bands
    )
    axes[1, 0].set(
        title="Big train: duration mix among each day’s events",
        ylabel="Daily events (%)",
        ylim=(0, 100),
    )
    axes[1, 0].legend(fontsize=7, ncol=3, loc="lower left")
    axes[1, 1].plot(dates, train.label_mean, label="Observed daily mean")
    axes[1, 1].plot(
        dates,
        train.duration_mix_reference,
        label="Fixed bucket means × daily shares",
        linestyle="--",
    )
    axes[1, 1].set(
        title="Big train: label mean versus duration-composition reference",
        ylabel="Mean watch ratio (unitless)",
    )
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Interaction date")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def _calendar_days(values: pd.Series) -> pd.Index:
    """Retain missing calendar days as gaps instead of interpolating observations."""
    valid = values.dropna().astype("int64").astype(str)
    dates = pd.to_datetime(valid, format="%Y%m%d")
    if dates.empty:
        raise ValueError("Daily analysis requires at least one dated event")
    days = pd.date_range(dates.min(), dates.max(), freq="D")
    return pd.Index(days.strftime("%Y%m%d").astype("int64"), name="date")
