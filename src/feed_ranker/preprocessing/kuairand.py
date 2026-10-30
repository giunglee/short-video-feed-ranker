"""Temporal binary-event adapter for public KuaiRand standard logs.

Dataset: https://kuairand.com/ (Gao et al., CIKM 2022).
This protocol is separate from KuaiRec's pair-disjoint big/small matrix protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.data.parquet import VERSION, ParquetReader
from feed_ranker.persistence import sha256, write_json
from feed_ranker.preprocessing.features import fit_entity_features

TRAIN_END = 1651161600000  # 2022-04-29 00:00 UTC+8, exclusive
VAL_END = 1651420800000  # 2022-05-02 00:00 UTC+8, exclusive


def temporal_events(
    frame: pd.DataFrame,
    label: str,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Full-record deduplication only; never invent an ordering within timestamp ties."""
    if label not in ("long_view", "is_click") or train_end >= val_end:
        raise ValueError("Require long_view/is_click and train_end < val_end")
    required = ["user_id", "video_id", "time_ms", "duration_ms", "is_rand", "tab", label]
    if frame[required].isna().any().any():
        raise ValueError("Missing required event values")
    for name in required:
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"Nonfinite or noninteger {name}")
    if (frame[["user_id", "video_id", "duration_ms"]] < 0).any().any():
        raise ValueError("Negative ID/duration")
    if not frame[label].isin([0, 1]).all() or not frame.is_rand.eq(0).all():
        raise ValueError("Expected binary labels and standard traffic only")
    if not frame.tab.eq(1).all() or not frame.time_ms.gt(0).all():
        raise ValueError("Expected tab1 and positive timestamps")
    unique = frame.drop_duplicates().sort_values(["time_ms", "user_id", "video_id"])
    removed = len(frame) - len(unique)
    # Event ID is a locator in this prepared dataset, not a claim of intra-tie order.
    unique = unique.reset_index(drop=True).reset_index(names="event_id")
    train = unique.loc[unique.time_ms.lt(train_end)].copy()
    val = unique.loc[unique.time_ms.ge(train_end) & unique.time_ms.lt(val_end)].copy()
    if train.empty or val.empty:
        raise ValueError("Both training and validation must contain events")
    return train, val, removed


def read_cohort(
    raw: Path,
    *,
    label: str = "long_view",
    users: int = 32,
    cohort_seed: int = 2026,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read complete source records before projection; select users from train only."""
    if users < 0 or label not in ("long_view", "is_click") or train_end >= val_end:
        raise ValueError("Invalid cohort size, label, or temporal bounds")
    files = sorted(raw.glob("log_standard*.csv"))
    if not files:
        raise FileNotFoundError(f"No log_standard*.csv in {raw}")
    source_hashes = {path.name: sha256(path) for path in files}
    training_users: set[int] = set()
    for path in files:
        for chunk in pd.read_csv(
            path, chunksize=250_000, usecols=["user_id", "time_ms", "tab", "is_rand"]
        ):
            scoped = chunk.loc[chunk.tab.eq(1) & chunk.is_rand.eq(0)]
            values = scoped[["user_id", "time_ms"]].to_numpy(dtype=float)
            if (
                not np.isfinite(values).all()
                or (values != np.floor(values)).any()
                or scoped.user_id.lt(0).any()
                or scoped.time_ms.le(0).any()
            ):
                raise ValueError("Invalid user ID/timestamp in standard tab1 traffic")
            keep = chunk.tab.eq(1) & chunk.is_rand.eq(0) & chunk.time_ms.lt(train_end)
            training_users.update(chunk.loc[keep, "user_id"].tolist())
    available = np.array(sorted(training_users), dtype=np.int64)
    if not len(available) or users > len(available):
        raise ValueError(f"Requested {users} users but only {len(available)} training users")
    selected = np.sort(
        np.random.default_rng(cohort_seed).choice(
            available,
            users or len(available),
            replace=False,
        )
    )
    frames = []
    for path in files:
        for chunk in pd.read_csv(path, chunksize=250_000):
            keep = (
                chunk.user_id.isin(selected)
                & chunk.tab.eq(1)
                & chunk.is_rand.eq(0)
                & chunk.time_ms.lt(val_end)
            )
            frames.append(chunk.loc[keep])
    train, val, duplicates = temporal_events(
        pd.concat(frames, ignore_index=True), label, train_end, val_end
    )
    return (
        train,
        val,
        {
            "source_sha256": source_hashes,
            "selected_users": selected.tolist(),
            "duplicates_removed": duplicates,
        },
    )


def prepare_kuairand(
    raw: Path,
    out: Path,
    *,
    targets: tuple[str, ...] = ("long_view",),
    users: int = 32,
    cohort_seed: int = 2026,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
    basic: Path | None = None,
    profiles: Path | None = None,
) -> ParquetReader:
    if not targets or len(set(targets)) != len(targets) or set(targets) - {"long_view", "is_click"}:
        raise ValueError("Configure unique supported binary target columns")
    train, val, source = read_cohort(
        raw,
        label=targets[0],
        users=users,
        cohort_seed=cohort_seed,
        train_end=train_end,
        val_end=val_end,
    )
    for frame in (train, val):
        for target in targets:
            if target not in frame or not frame[target].isin([0, 1]).all():
                raise ValueError(f"{target}: expected observed binary targets")
    state = fit_entity_features(
        train,
        None if basic is None else pd.read_csv(basic),
        None if profiles is None else pd.read_csv(profiles),
    )
    out.mkdir(parents=True, exist_ok=False)
    state.save(out)
    keys = ["event_id", "user_id", "video_id", "time_ms"]
    for name, frame in (("train", train), ("validation", val)):
        features = state.transform(
            frame.user_id.to_numpy(dtype=np.int64), frame.video_id.to_numpy(dtype=np.int64)
        )
        frame = frame[keys + list(targets)].astype(
            {**{k: "int64" for k in keys}, **{t: "float32" for t in targets}}
        )
        frame = frame.assign(**features, warm_item=features["item_code"] > 0)
        frame.to_parquet(out / f"{name}.parquet", index=False)
    files = ["train.parquet", "validation.parquet", "feature-state.npz", "feature-tables.json"]
    write_json(
        out / "manifest.json",
        {
            "version": VERSION,
            "metadata_dtypes": {**dict.fromkeys(keys, "int64"), "warm_item": "bool"},
            "feature_dtypes": {
                name: "int64" if name in state.vocab_sizes else "float32"
                for name in state.feature_names
            },
            "vocab_sizes": state.vocab_sizes,
            "feature_descriptions": state.descriptions,
            "targets": list(targets),
            "splits": {
                "train": {"end": train_end},
                "validation": {"start": train_end, "end": val_end},
            },
            **source,
            "train_end": train_end,
            "val_end": val_end,
            "cohort_seed": cohort_seed,
            "traffic": "standard/tab1",
            "dedup": "complete source records before projection",
            "snapshot_sha256": {p.name: sha256(p) for p in (basic, profiles) if p is not None},
            "files": {name: sha256(out / name) for name in files},
        },
    )
    reader = ParquetReader(out)
    reader.validate()
    return reader
