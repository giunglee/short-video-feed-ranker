"""Label-free candidate features from a resolved input contract and fitted lookups.

FeatureConfig describes the model inputs; this builder owns raw-ID encoding and
lookup validation. Numeric transforms remain inside the shared PyTorch encoder.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feed_ranker.features.config import FeatureConfig, FeatureSpec


class KuaiRecFeatureBuilder:
    """Own fitted vocabularies, encoded user attributes, and train-median durations.

    No dataset paths or labels are needed. Unknown IDs/profile values use code 0;
    a required missing candidate duration is an error, never a label-derived fill.
    """

    def __init__(
        self,
        config: FeatureConfig,
        vocab: dict[str, list[str]],
        profiles: pd.DataFrame,
        durations: pd.Series,
        crosses: dict[str, dict[str, str | int | float]] | None = None,
        item_categories: pd.Series | None = None,
    ) -> None:
        self.config = FeatureConfig.from_dict(asdict(config))
        self.vocab = deepcopy(vocab)
        self.crosses = deepcopy(crosses or {})
        FeatureSpec(config.embedding_cols, config.dense_cols, self.crosses).validate()
        if set(self.crosses) != set(config.cross_feature_cols):
            raise ValueError("Cross definitions differ from model inputs")
        pooled = {c.name for c in config.embedding_configs if c.pooling is not None}
        if pooled - {"item_categories"} or (
            "item_categories" in config.embedding_cols and "item_categories" not in pooled
        ):
            raise ValueError("Candidate item categories require sum pooling")
        self.item_categories = None
        if pooled:
            if item_categories is None or item_categories.index.has_duplicates:
                raise ValueError("Require unique training item category lookups")
            if item_categories.index.dtype.kind not in "iu" or (item_categories.index < 0).any():
                raise ValueError("Invalid category lookup identities")
            if "item_categories" not in vocab:
                raise ValueError("Missing item category vocabulary")
            for codes in item_categories:
                values = np.asarray(codes)
                if (
                    values.ndim != 1
                    or not len(values)
                    or values.dtype.kind not in "iu"
                    or (values < 0).any()
                    or (values > len(vocab["item_categories"])).any()
                ):
                    raise ValueError("Invalid item category codes")
            self.item_categories = item_categories.map(tuple).copy()
        elif item_categories is not None:
            raise ValueError("Undeclared item category lookup")
        self.cross_inputs = {n for conditions in self.crosses.values() for n in conditions}
        self.profile_names = sorted(
            (set(config.embedding_cols) | {n for c in self.crosses.values() for n in c})
            - {"user_id", "video_id", "duration_band", "item_categories"}
        )
        if set(config.dense_cols) - {"video_duration"}:
            raise ValueError("Unsupported candidate numeric feature")
        self.needs_duration = bool(config.dense_cols) or "duration_band" in (
            set(config.embedding_cols) | {n for c in self.crosses.values() for n in c}
        )
        needed = set(config.embedding_cols) | set(self.profile_names)
        if not needed <= set(vocab):
            raise ValueError("Missing input vocabulary")
        self.mappings = {}
        for name in needed:
            values = [str(v) for v in vocab[name]]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate vocabulary values: {name}")
            self.mappings[name] = {v: i + 1 for i, v in enumerate(values)}
        for field in config.embedding_configs:
            if field.vocab_size != len(vocab[field.name]) + 1:
                raise ValueError(f"Vocabulary size mismatch: {field.name}")
        if set(profiles.columns) != set(self.profile_names):
            raise ValueError("User lookup columns differ from declared features")
        if profiles.index.has_duplicates or (
            len(profiles) and (profiles.index.dtype.kind not in "iu" or (profiles.index < 0).any())
        ):
            raise ValueError("Invalid user lookup identities")
        for name in self.profile_names:
            values = profiles[name].to_numpy()
            if (
                values.dtype.kind not in "iu"
                or (values < 0).any()
                or (values > len(vocab[name])).any()
            ):
                raise ValueError(f"Invalid profile codes: {name}")
        if (
            durations.index.has_duplicates
            or durations.index.dtype.kind not in "iu"
            or (durations.index < 0).any()
            or not np.isfinite(durations).all()
            or (durations <= 0).any()
        ):
            raise ValueError("Missing/invalid training-derived candidate duration")
        self.profiles, self.durations = profiles.copy(), durations.copy()

    def raw(self, users: np.ndarray, items: np.ndarray) -> pd.DataFrame:
        if users.ndim != 1 or users.shape != items.shape:
            raise ValueError("Candidate user/item shape mismatch")
        raw = pd.DataFrame({"user_id": users, "video_id": items})
        if self.needs_duration:
            raw["video_duration"] = self.durations.reindex(items).to_numpy(dtype=np.float32)
        return raw

    def encode(self, raw: pd.DataFrame) -> dict[str, np.ndarray]:
        """Encode declared inputs only, ignoring event labels and supplied profiles."""
        values = {"user_id": raw.user_id, "video_id": raw.video_id}
        for name, ids in values.items():
            if ids.dtype.kind not in "iu" or (ids < 0).any():
                raise ValueError(f"Invalid raw identity: {name}")
        result = {}
        if self.needs_duration:
            duration = raw.video_duration.to_numpy(dtype=np.float32)
            if not np.isfinite(duration).all() or (duration <= 0).any():
                raise ValueError("Missing/invalid candidate duration")
            values["duration_band"] = duration_band(raw.video_duration)
            if "video_duration" in self.config.dense_cols:
                result["video_duration"] = duration
        aligned = self.profiles.reindex(raw.user_id.to_numpy())
        for name in self.profile_names:
            codes = aligned[name].fillna(0).to_numpy(dtype=np.int64)
            if name in self.config.embedding_cols:
                result[name] = codes
            # Cross conditions use raw profile categories, not encoded integers.
            if name in self.cross_inputs:
                values[name] = (
                    pd.Series([None, *self.vocab[name]]).iloc[codes].reset_index(drop=True)
                )
        if self.item_categories is not None:
            sequences = self.item_categories.reindex(raw.video_id).map(
                lambda v: v if isinstance(v, tuple) else (0,)
            )
            result["item_categories"] = pad_categories(sequences.tolist())
        for name in self.config.embedding_cols:
            if name not in result:
                result[name] = (
                    values[name]
                    .astype("string")
                    .map(self.mappings[name])
                    .fillna(0)
                    .to_numpy(dtype=np.int64)
                )
        for name, conditions in self.crosses.items():
            if not conditions:
                raise ValueError("Empty cross definition")
            mask = np.ones(len(raw), dtype=bool)
            for field, target in conditions.items():
                mask &= (
                    values[field]
                    .astype("string")
                    .eq(str(target))
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
            result[name] = mask.astype(np.float32)
        return result

    def build(self, users: np.ndarray, items: np.ndarray) -> dict[str, np.ndarray]:
        return self.encode(self.raw(users, items))


def fit_candidate_features(
    config: FeatureConfig, metadata: dict[str, Any], training_path: Path, duration_path: Path
) -> KuaiRecFeatureBuilder:
    """Extract only declared user attributes from training and frozen item durations."""
    crosses = {n: metadata["spec"]["crosses"][n] for n in config.cross_feature_cols}
    profile_names = sorted(
        (set(config.embedding_cols) | {n for c in crosses.values() for n in c})
        - {"user_id", "video_id", "duration_band", "item_categories"}
    )
    columns = ["raw_user_id", *profile_names]
    parts = [
        b.to_pandas().drop_duplicates()
        for b in pq.ParquetFile(training_path).iter_batches(batch_size=131072, columns=columns)
    ]
    profiles = pd.concat(parts, ignore_index=True).drop_duplicates()
    if profiles.raw_user_id.duplicated().any() or profiles.isna().any().any():
        raise ValueError("Conflicting or missing training profile codes")
    profiles = profiles.set_index("raw_user_id")
    durations = pd.read_parquet(duration_path).set_index("video_id")["median"]
    names = set(config.embedding_cols) | set(profile_names)
    categories = None
    if "item_categories" in config.embedding_cols:
        parts = []
        for batch in pq.ParquetFile(training_path).iter_batches(
            batch_size=131072, columns=["raw_video_id", "item_categories"]
        ):
            frame = batch.to_pandas()
            frame["item_categories"] = frame.item_categories.map(tuple)
            parts.append(frame.drop_duplicates())
        items = pd.concat(parts, ignore_index=True).drop_duplicates()
        if items.raw_video_id.duplicated().any():
            raise ValueError("Conflicting training item category codes")
        categories = items.set_index("raw_video_id").item_categories
    return KuaiRecFeatureBuilder(
        config, {n: metadata["vocab"][n] for n in names}, profiles, durations, crosses, categories
    )


def duration_band(values: pd.Series) -> pd.Series:
    """Fixed duration bins in milliseconds; boundaries match the descriptive EDA."""
    return pd.cut(
        values,
        [0, 5000, 10000, 20000, 40000, 60000, np.inf],
        labels=["0–5s", "5–10s", "10–20s", "20–40s", "40–60s", ">60s"],
        include_lowest=True,
    ).astype("string")


def pad_categories(sequences: list[tuple[int, ...]]) -> np.ndarray:
    """Keep OOV=0 distinct from masked padding=-1; produce [B, L] int64."""
    width = max((len(v) for v in sequences), default=1)
    result = np.full((len(sequences), width), -1, dtype=np.int64)
    for row, values in enumerate(sequences):
        result[row, : len(values)] = values
    return result
