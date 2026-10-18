"""Train-derived KuaiRec candidate features and frozen vocabulary encoding."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feed_ranker.data.kuairec_preprocessor import duration_band

PROFILES = ["age_range", "follow_user_num_range"]


def pair_codes(frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """Collision-free packing for the nonnegative uint32 KuaiRec identifiers."""
    values = []
    for name in ("user_id", "video_id"):
        v = frame[prefix + name].to_numpy()
        if v.dtype.kind not in "iu" or (v < 0).any() or (v >= 2**32).any():
            raise ValueError("Pair IDs must be uint32-compatible integers")
        values.append(v.astype(np.uint64))
    return (values[0] << np.uint64(32)) | values[1]


def training_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Require a single encoded profile per training user; never aggregate conflicts."""
    unique = frame[["raw_user_id", *PROFILES]].drop_duplicates()
    if unique.raw_user_id.duplicated().any() or unique[PROFILES].isna().any().any():
        raise ValueError("Conflicting or missing training profile codes")
    for name in PROFILES:
        if unique[name].dtype.kind not in "iu" or unique[name].lt(0).any():
            raise ValueError("Profile codes must be nonnegative integers")
    return unique.set_index("raw_user_id")


def encode_candidates(
    raw: pd.DataFrame,
    metadata: dict[str, Any],
    profiles: pd.DataFrame,
    mappings: dict[str, dict[str, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Build only declared model inputs; held-out labels and event context are ignored."""
    duration = raw.video_duration.to_numpy(dtype=np.float32)
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Require positive finite candidate durations")
    result = {"video_duration": duration}
    for name, values in {
        "user_id": raw.user_id,
        "video_id": raw.video_id,
        "duration_band": duration_band(raw.video_duration),
    }.items():
        mapping = (
            mappings[name]
            if mappings is not None
            else {str(v): i + 1 for i, v in enumerate(metadata["vocab"][name])}
        )
        result[name] = values.astype("string").map(mapping).fillna(0).to_numpy(dtype=np.int64)
    aligned = profiles.reindex(raw.user_id.to_numpy())
    for name in PROFILES:
        result[name] = aligned[name].fillna(0).to_numpy(dtype=np.int64)
        if (result[name] >= len(metadata["vocab"][name]) + 1).any():
            raise ValueError("Training profile code exceeds frozen vocabulary")
    return result


class KuaiRecFeatures:
    """Fit lookups once; build only requested pairs, without validation outcomes."""

    def __init__(
        self, metadata: dict[str, Any], profiles: pd.DataFrame, durations: pd.Series
    ) -> None:
        if (
            durations.index.has_duplicates
            or not np.isfinite(durations).all()
            or (durations <= 0).any()
        ):
            raise ValueError("Missing/invalid training-derived candidate duration")
        self.metadata, self.profiles, self.durations = metadata, profiles.copy(), durations.copy()

        self.mappings = {
            name: {str(v): i + 1 for i, v in enumerate(values)}
            for name, values in metadata.get("vocab", {}).items()
        }

    @classmethod
    def load(
        cls, metadata: dict[str, Any], training_path: Path, duration_path: Path
    ) -> "KuaiRecFeatures":
        parts = [
            b.to_pandas().drop_duplicates()
            for b in pq.ParquetFile(training_path).iter_batches(
                batch_size=131072, columns=["raw_user_id", *PROFILES]
            )
        ]
        profiles = training_profiles(pd.concat(parts, ignore_index=True))
        durations = pd.read_parquet(duration_path).set_index("video_id")["median"]
        return cls(metadata, profiles, durations)

    def raw(self, users: np.ndarray, items: np.ndarray) -> pd.DataFrame:
        if users.shape != items.shape:
            raise ValueError("Candidate user/item shape mismatch")
        durations = self.durations.reindex(items).to_numpy(dtype=np.float32)
        if not np.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError("Missing/invalid training-derived candidate duration")
        return pd.DataFrame({"user_id": users, "video_id": items, "video_duration": durations})

    def build(self, users: np.ndarray, items: np.ndarray) -> dict[str, np.ndarray]:
        return encode_candidates(
            self.raw(users, items), self.metadata, self.profiles, self.mappings
        )
