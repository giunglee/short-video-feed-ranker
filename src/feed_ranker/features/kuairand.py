"""Strict temporal candidate features fitted only on training events.

Snapshot attributes and current-event outcomes are intentionally outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from feed_ranker.features.config import DenseFeatureConfig, EmbeddingFeatureConfig, FeatureConfig


@dataclass
class Encoding:
    """Training-only categorical vocabularies and frozen item-duration features."""

    users: np.ndarray
    items: np.ndarray
    duration: np.ndarray  # [items+1], zero is missing/OOV, in milliseconds

    @classmethod
    def fit(cls, train: pd.DataFrame) -> Encoding:
        if train.empty:
            raise ValueError("Training events are empty")
        users = np.sort(train.user_id.unique())
        items = np.sort(train.video_id.unique())
        medians = train.loc[train.duration_ms.gt(0)].groupby("video_id").duration_ms.median()
        duration = np.r_[0.0, medians.reindex(items).fillna(0).to_numpy()].astype(np.float32)
        return cls(users, items, duration)

    @staticmethod
    def encode(values: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
        return pd.Index(vocabulary).get_indexer(values).astype(np.int64) + 1

    def features(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        # Explicit allowlist: no event outcome or validation duration is consumed.
        users = self.encode(frame.user_id.to_numpy(), self.users)
        items = self.encode(frame.video_id.to_numpy(), self.items)
        duration = self.duration[items]
        return {
            "user_id": torch.from_numpy(users),
            "video_id": torch.from_numpy(items),
            "duration_ms": torch.from_numpy(duration),
            "duration_missing": torch.from_numpy((duration == 0).astype(np.float32)),
        }

    def config(self) -> FeatureConfig:
        # reasonable defaults, not from any specific production config
        return FeatureConfig(
            dense_configs=[
                DenseFeatureConfig("duration_ms", "log1p_nonnegative"),
                DenseFeatureConfig("duration_missing"),
            ],
            embedding_configs=[
                EmbeddingFeatureConfig("user_id", len(self.users) + 1, 8),
                EmbeddingFeatureConfig("video_id", len(self.items) + 1, 8),
            ],
        )


FEATURE_SETS = ("base", "no_user", "no_item", "no_duration", "duration_band")


class KuaiRandFeatures:
    """Train/serve feature parity using the same fitted Encoding and input recipe."""

    def __init__(self, encoding: Encoding, feature_set: str = "base") -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown strict feature set: {feature_set}")
        self.encoding, self.feature_set = encoding, feature_set
        self.users = pd.Index(encoding.users)
        self.items = pd.Index(encoding.items)
        self.config = encoding.config()
        removed = {"no_user": "user_id", "no_item": "video_id"}.get(feature_set)
        self.config.embedding_configs = [
            c for c in self.config.embedding_configs if c.name != removed
        ]
        if feature_set == "no_duration":
            # Constant zero satisfies the existing Deep numeric interface without information.
            self.config.dense_configs = [DenseFeatureConfig("constant")]
        if feature_set == "duration_band":
            self.config.embedding_configs.append(EmbeddingFeatureConfig("duration_band", 5, 8))

    def build(self, users: np.ndarray, items: np.ndarray) -> dict[str, np.ndarray]:
        if users.shape != items.shape or users.ndim != 1:
            raise ValueError("Require aligned one-dimensional raw IDs")
        user_ids, item_ids = self.users.get_indexer(users) + 1, self.items.get_indexer(items) + 1
        duration = self.encoding.duration[item_ids]
        band = np.searchsorted([10_000, 30_000, 60_000], duration, side="left") + 1
        band[duration == 0] = 0
        values = {
            "user_id": user_ids,
            "video_id": item_ids,
            "duration_ms": duration,
            "duration_missing": (duration == 0).astype(np.float32),
            "constant": np.zeros(len(items), dtype=np.float32),
            "duration_band": band,
        }
        return {name: values[name] for name in self.config.feature_cols}
