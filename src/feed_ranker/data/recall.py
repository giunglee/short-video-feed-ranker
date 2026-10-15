"""First-observed pair data for KuaiRec recall; never reads the small matrix.

Public data: https://kuairec.com/. Labels are project utility heuristics, not
validated satisfaction or causal outcomes. Ranking split membership is inherited.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.kuairec_preprocessor import deduplicate_signatures

KEYS = ["user_id", "video_id"]
VERSION = "recall-first-watch-v1"


def classify(duration_ms: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Return int8 negative=-1, neutral=0, positive=1, with disjoint boundaries."""
    duration = np.asarray(duration_ms, dtype=np.float64) / 1000
    ratio = np.asarray(ratio, dtype=np.float64)
    if duration.shape != ratio.shape or not np.isfinite(ratio).all():
        raise ValueError("Duration/ratio shape mismatch or nonfinite ratio")
    if not np.isfinite(duration).all() or (duration <= 0).any() or (ratio < 0).any():
        raise ValueError("Require positive finite duration and nonnegative ratio")
    positive = np.minimum(1, np.sqrt(10 / duration))
    negative = np.minimum(0.3, 0.5 * positive)
    return np.where(ratio >= positive, 1, np.where(ratio < negative, -1, 0)).astype(np.int8)


def first_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Chronological first, or original-row first for pairs with any missing time."""
    if frame.empty or frame.source_row.duplicated().any():
        raise ValueError("Require nonempty events with unique stable source_row")
    times = frame.timestamp.to_numpy(dtype=float)
    if np.isinf(times).any():
        raise ValueError("Infinite timestamps are invalid")
    missing = frame.timestamp.isna().groupby([frame[k] for k in KEYS]).transform("any")
    ordered = frame.assign(_order_time=frame.timestamp.where(~missing, 0)).sort_values(
        [*KEYS, "_order_time", "source_row"], kind="stable"
    )
    first = ordered.drop_duplicates(KEYS).drop(columns="_order_time").copy()
    missing_pairs = frame.loc[missing, KEYS].drop_duplicates()
    known = frame.loc[~missing]
    earliest = known.groupby(KEYS).timestamp.transform("min")
    tied = known.loc[known.timestamp.eq(earliest)].groupby(KEYS).size().gt(1).sum()
    first["outcome"] = classify(first.video_duration.to_numpy(), first.watch_ratio.to_numpy())
    return first, {"missing_time_pairs": len(missing_pairs), "tied_first_time_pairs": int(tied)}


def prepare(raw_big: Path, ranking_data: Path, output: Path) -> None:
    """Reuse exact cleaned ranking train/val membership; produce separate recall files."""
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((ranking_data / "meta.json").read_text())
    if metadata.get("preprocessing_version") != "signature-dedup-v2":
        raise ValueError("Recall requires signature-dedup-v2 ranking data")
    raw_hash = sha256(raw_big)
    if raw_hash != metadata["sources"]["big_matrix.csv"]:
        raise ValueError("Raw big matrix differs from ranking source")
    memberships = []
    hashes = {}
    for split in ("train", "val"):
        path = ranking_data / f"{split}.parquet"
        hashes[split] = sha256(path)
        if hashes[split] != metadata["splits"][split]["sha256"]:
            raise ValueError(f"Changed ranking {split}")
        pairs = pd.read_parquet(path, columns=["raw_user_id", "raw_video_id"])
        pairs.columns = KEYS
        memberships.append(pairs.drop_duplicates().assign(split=split))
    membership = pd.concat(memberships, ignore_index=True)
    if membership.duplicated(KEYS).any():
        raise ValueError("Training/validation pair overlap")
    columns = [*KEYS, "timestamp", "play_duration", "video_duration", "watch_ratio"]
    frame = pd.read_csv(raw_big, usecols=columns)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame, removed = deduplicate_signatures(frame)
    frame = frame.merge(membership, on=KEYS, validate="many_to_one", how="left")
    if frame.split.isna().any():
        raise ValueError("Raw pair missing from ranking membership")
    for split in ("train", "val"):
        if (
            int(frame.split.eq(split).sum())
            != pq.ParquetFile(ranking_data / f"{split}.parquet").metadata.num_rows
        ):
            raise ValueError(f"Cleaned event count mismatch in {split}")
    # Label validity is checked before reducing events, including nonselected events.
    classify(frame.video_duration.to_numpy(), frame.watch_ratio.to_numpy())
    duration = (
        frame.loc[frame.split.eq("train")]
        .groupby("video_id")
        .video_duration.agg(["median", "min", "max", "size"])
        .reset_index()
    )
    first, audit = first_events(frame)
    del frame, membership, memberships
    output.mkdir(parents=True)
    duration.to_parquet(output / "item_duration.parquet", index=False)
    files = {}
    counts = {}
    for split in ("train", "val"):
        selected = first.loc[first.split.eq(split)].drop(columns="split")
        path = output / f"{split}.parquet"
        selected.to_parquet(path, index=False)
        files[path.name] = sha256(path)
        counts[split] = {str(int(k)): int(v) for k, v in selected.outcome.value_counts().items()}
    files["item_duration.parquet"] = sha256(output / "item_duration.parquet")
    write_json(
        output / "meta.json",
        {
            "version": VERSION,
            "raw_big_sha256": raw_hash,
            "ranking_manifest_sha256": sha256(ranking_data / "meta.json"),
            "ranking_split_sha256": hashes,
            "split_seed": metadata["split_seed"],
            "source_sha256": sha256(Path(__file__)),
            "removed_duplicates": removed,
            "first_event_audit": audit,
            "outcomes": counts,
            "files": files,
            "label": {
                "anchor_seconds": 10,
                "exponent": 0.5,
                "negative_cap": 0.3,
                "negative_positive_fraction": 0.5,
            },
            "duration_conflicts_over_1s": int((duration["max"] - duration["min"]).gt(1000).sum()),
        },
    )


@dataclass
class Pools:
    """Flattened observed validation pools; offsets delimit raw-user ordered rows."""

    users: np.ndarray
    items: np.ndarray
    relevant: np.ndarray
    offsets: np.ndarray


class RecallData:
    """Compact identity mappings and positive/negative sets; no test interface."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.meta = json.loads((directory / "meta.json").read_text())
        if self.meta["version"] != VERSION:
            raise ValueError("Unsupported recall dataset")
        for name, expected in self.meta["files"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Recall data changed: {name}")
        train = pd.read_parquet(directory / "train.parquet")
        val = pd.read_parquet(directory / "val.parquet")
        if train.duplicated(KEYS).any() or val.duplicated(KEYS).any():
            raise ValueError("Recall requires one row per pair")
        if pd.MultiIndex.from_frame(val[KEYS]).isin(pd.MultiIndex.from_frame(train[KEYS])).any():
            raise ValueError("Train/validation pair leakage")
        self.users = np.union1d(train.user_id.unique(), val.user_id.unique())
        self.items = np.sort(train.video_id.unique())
        self.positive = self._histories(train.loc[train.outcome.eq(1)])
        self.negative = self._histories(train.loc[train.outcome.eq(-1)])
        self.seen = self._histories(train)
        self.supported = np.array(
            [bool(len(p) and len(n)) for p, n in zip(self.positive, self.negative, strict=True)]
        )
        eligible = val.video_id.isin(self.items)
        excluded = val.loc[~eligible]
        val = val.loc[eligible].sort_values(KEYS)
        vu = np.searchsorted(self.users, val.user_id.to_numpy())
        self.pools = Pools(
            users=np.repeat(np.arange(len(self.users)), np.bincount(vu, minlength=len(self.users))),
            items=np.searchsorted(self.items, val.video_id.to_numpy()),
            relevant=val.outcome.eq(1).to_numpy(),
            offsets=np.r_[0, np.cumsum(np.bincount(vu, minlength=len(self.users)))],
        )
        train_users = np.isin(self.users, train.user_id.unique())
        lengths = np.array([len(p) for p in self.positive])
        neg_lengths = np.array([len(n) for n in self.negative])
        self.audit: dict[str, Any] = {
            "users": len(self.users),
            "catalog_items": len(self.items),
            "train_pairs": len(train),
            "eligible_validation_pairs": len(val),
            "excluded_validation_pairs": len(excluded),
            "excluded_validation_positives": int(excluded.outcome.eq(1).sum()),
            "excluded_validation_users": int(excluded.user_id.nunique()),
            "mf_supported_users": int(self.supported.sum()),
            "unseen_training_users": int((~train_users).sum()),
            "train_users_without_positive": int((train_users & (lengths == 0)).sum()),
            "train_users_without_negative": int((train_users & (neg_lengths == 0)).sum()),
            "positive_history_quantiles": np.quantile(lengths, [0, 0.1, 0.5, 0.9, 1]).tolist(),
            "negative_history_quantiles": np.quantile(neg_lengths, [0, 0.1, 0.5, 0.9, 1]).tolist(),
            "negative_counts": neg_lengths.tolist(),
            "training_outcomes": {
                str(int(k)): int(v) for k, v in train.outcome.value_counts().items()
            },
        }

    def _histories(self, frame: pd.DataFrame) -> list[np.ndarray]:
        result = [np.empty(0, dtype=np.int64) for _ in self.users]
        for user, group in frame.groupby("user_id", sort=False):
            result[int(np.searchsorted(self.users, user))] = np.searchsorted(
                self.items, np.sort(group.video_id.to_numpy())
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-big", type=Path, required=True)
    parser.add_argument("--ranking-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw_big, args.ranking_data, args.output)


if __name__ == "__main__":
    main()
