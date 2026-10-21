"""Temporal binary-event adapter for public KuaiRand standard logs.

Dataset: https://kuairand.com/ (Gao et al., CIKM 2022).
This protocol is separate from KuaiRec's pair-disjoint big/small matrix protocol.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.schema import RecommendationBatch
from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures

VERSION = "kuairand-temporal-binary-v1"
TRAIN_END = 1651161600000  # 2022-04-29 00:00 UTC+8, exclusive
VAL_END = 1651420800000  # 2022-05-02 00:00 UTC+8, exclusive


class EventDataset(Dataset):
    """Event BCE labels [B] and feature tensors; repeats keep their own labels."""

    def __init__(
        self, frame: pd.DataFrame, encoding: Encoding, label: str, feature_set: str = "base"
    ) -> None:
        builder = KuaiRandFeatures(encoding, feature_set)
        self.features = {
            name: torch.from_numpy(value)
            for name, value in builder.build(
                frame.user_id.to_numpy(), frame.video_id.to_numpy()
            ).items()
        }
        self.labels = torch.tensor(frame[label].to_numpy(), dtype=torch.float32)
        self.users = torch.tensor(frame.user_id.to_numpy(), dtype=torch.long)
        self.items = torch.tensor(frame.video_id.to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> RecommendationBatch:
        return {
            "feature": {key: value[index] for key, value in self.features.items()},
            "label": self.labels[index],
            "raw_user_id": self.users[index],
            "raw_video_id": self.items[index],
        }

    def batches(self, size: int, seed: int | None = None) -> Iterator[RecommendationBatch]:
        """Vectorized CPU indexing; avoid Python per-event collation on million-row data."""
        if size < 1:
            raise ValueError("Batch size must be positive")
        order = (
            np.arange(len(self))
            if seed is None
            else np.random.default_rng(seed).permutation(len(self))
        )
        for start in range(0, len(order), size):
            ids = torch.from_numpy(order[start : start + size])
            yield {
                "feature": {name: values[ids] for name, values in self.features.items()},
                "label": self.labels[ids],
                "raw_user_id": self.users[ids],
                "raw_video_id": self.items[ids],
            }


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


def prepare(
    raw: Path,
    out: Path,
    *,
    label: str = "long_view",
    users: int = 32,
    cohort_seed: int = 2026,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, Encoding, dict[str, Any]]:
    """Select users using training traffic, read their complete pre-val-end logs.

    Intended for 1K development cohorts; --users 0 selects all training users.
    Chunked reading bounds ingestion; selected cohort events are materialized in RAM.
    """
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
    encoding = Encoding.fit(train)
    out.mkdir(parents=True, exist_ok=False)
    columns = ["event_id", "user_id", "video_id", "time_ms", label]
    # Validation raw duration/outcomes are deliberately not cached as feature inputs.
    for name, frame in [("train", train), ("val", val)]:
        frame[columns].to_parquet(out / f"{name}.parquet", index=False)
    np.savez(
        out / "encoding.npz", users=encoding.users, items=encoding.items, duration=encoding.duration
    )
    manifest: dict[str, Any] = {
        "version": VERSION,
        "label": label,
        "source_sha256": source_hashes,
        "source_directory": str(raw.resolve()),
        "cohort_seed": cohort_seed,
        "selected_users": selected.tolist(),
        "train_end_exclusive_ms": train_end,
        "val_end_exclusive_ms": val_end,
        "traffic": "standard/tab1",
        "dedup": "identical complete source records",
        "duplicates_removed": duplicates,
        "features": "train-only IDs; train positive-duration item median; missing flag",
        "repeat_policy": "retain repeat events; repeat items eligible for recall",
        "splits": {
            name: {
                "events": len(f),
                "users": int(f.user_id.nunique()),
                "items": int(f.video_id.nunique()),
                "positive_rate": float(f[label].mean()),
                "min_time_ms": int(f.time_ms.min()),
                "max_time_ms": int(f.time_ms.max()),
            }
            for name, f in [("train", train), ("val", val)]
        },
        "files": {
            name: sha256(out / name) for name in ["train.parquet", "val.parquet", "encoding.npz"]
        },
    }
    write_json(out / "manifest.json", manifest)
    return train[columns], val[columns], encoding, manifest


def histories(
    train: pd.DataFrame,
    encoding: Encoding,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Positive-ever vs observed never-positive items; no contradictory BPR pair."""
    pairs = train.groupby(["user_id", "video_id"], as_index=False)[label].max()
    positive, negative = [], []
    grouped = {int(u): f for u, f in pairs.groupby("user_id")}
    for user in encoding.users:
        rows = grouped[int(user)]
        ids = encoding.encode(rows.video_id.to_numpy(), encoding.items) - 1
        positive.append(ids[rows[label].to_numpy() == 1])
        negative.append(ids[rows[label].to_numpy() == 0])
    return positive, negative


def load_prepared(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, Encoding, dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Unsupported KuaiRand preprocessing version")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    with np.load(directory / "encoding.npz", allow_pickle=False) as state:
        encoding = Encoding(state["users"], state["items"], state["duration"])
    train, val = [pd.read_parquet(directory / f"{split}.parquet") for split in ["train", "val"]]
    if (
        not train.time_ms.lt(manifest["train_end_exclusive_ms"]).all()
        or not val.time_ms.ge(manifest["train_end_exclusive_ms"]).all()
        or not val.time_ms.lt(manifest["val_end_exclusive_ms"]).all()
    ):
        raise ValueError("Prepared temporal boundaries do not match manifest")
    return train, val, encoding, manifest
