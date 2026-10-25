"""Explicit snapshot assumptions for KuaiRand side-feature experiments.

Public field definitions: https://kuairand.com/ . Author/duration mappings are
assumed available at serving; anonymous user snapshots have unknown semantics and
unverified historical availability. No monthly statistics or outcomes are inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

ANONYMOUS = tuple(f"onehot_feat{i}" for i in range(18))


def _vocabulary(values: pd.Series) -> np.ndarray:
    valid = values.dropna()
    if not np.isfinite(valid.to_numpy(dtype=float)).all() or (valid % 1 != 0).any():
        raise ValueError("Categorical snapshot values must be finite integers or null")
    return np.sort(valid.loc[valid.ne(-124)].unique()).astype(np.int64)


@dataclass
class MetadataEncoding:
    """Frozen lookups; vocabularies fit only training users/items, zero is OOV."""

    users: np.ndarray
    videos: np.ndarray
    author_codes: np.ndarray
    basic_duration: np.ndarray
    anonymous_codes: np.ndarray
    author_vocabulary: np.ndarray
    anonymous_vocabularies: list[np.ndarray]

    @classmethod
    def fit(
        cls, users: np.ndarray, items: np.ndarray, basic: pd.DataFrame, profiles: pd.DataFrame
    ) -> MetadataEncoding:
        if basic.video_id.duplicated().any() or profiles.user_id.duplicated().any():
            raise ValueError("Metadata entity keys must be unique")
        if basic.video_id.isna().any() or profiles.user_id.isna().any():
            raise ValueError("Metadata entity keys cannot be null")
        video = basic.set_index("video_id").sort_index()
        training = video.reindex(items)
        authors = _vocabulary(training.author_id)
        # Zero and all other legitimate codes remain categories; -124/null are unknown.
        codes = pd.Index(authors).get_indexer(video.author_id) + 1
        duration = video.video_duration.to_numpy(dtype=np.float64)
        duration = np.where(np.isfinite(duration) & (duration > 0), duration, 0).astype(np.float32)
        profile = profiles.set_index("user_id").reindex(users)
        vocabularies = [_vocabulary(profile[name]) for name in ANONYMOUS]
        anonymous = np.column_stack(
            [
                pd.Index(vocab).get_indexer(profile[name]) + 1
                for name, vocab in zip(ANONYMOUS, vocabularies, strict=True)
            ]
        )
        return cls(
            users.copy(),
            video.index.to_numpy(dtype=np.int64),
            np.r_[0, codes],
            np.r_[np.float32(0), duration],
            np.vstack([np.zeros((1, len(ANONYMOUS)), dtype=np.int64), anonymous]),
            authors,
            vocabularies,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            **{
                name: torch.from_numpy(getattr(self, name))
                for name in (
                    "users",
                    "videos",
                    "author_codes",
                    "basic_duration",
                    "anonymous_codes",
                    "author_vocabulary",
                )
            },
            "anonymous_vocabularies": [torch.from_numpy(v) for v in self.anonymous_vocabularies],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MetadataEncoding:
        return cls(
            **{
                name: [v.numpy() for v in value]
                if name == "anonymous_vocabularies"
                else value.numpy()
                for name, value in state.items()
            }
        )
