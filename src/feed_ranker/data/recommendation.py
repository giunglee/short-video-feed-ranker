"""Label-free candidate features from a resolved input contract and fitted lookups.

FeatureConfig describes the model inputs; this builder owns raw-ID encoding and
lookup validation. Numeric transforms remain inside the shared PyTorch encoder.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd

from feed_ranker.data.config import FeatureConfig, FeatureSpec
from feed_ranker.data.kuairec_preprocessor import duration_band


class KuaiRecFeatures:
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
    ) -> None:
        self.config = FeatureConfig.from_dict(asdict(config))
        self.vocab = deepcopy(vocab)
        self.crosses = deepcopy(crosses or {})
        FeatureSpec(config.embedding_cols, config.dense_cols, self.crosses).validate()
        if set(self.crosses) != set(config.cross_feature_cols):
            raise ValueError("Cross definitions differ from model inputs")
        self.cross_inputs = {n for conditions in self.crosses.values() for n in conditions}
        self.profile_names = sorted(
            (set(config.embedding_cols) | {n for c in self.crosses.values() for n in c})
            - {"user_id", "video_id", "duration_band"}
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
