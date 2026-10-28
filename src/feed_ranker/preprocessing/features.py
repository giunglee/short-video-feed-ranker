"""Fit entity features from training events and explicitly supplied snapshots."""

from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.feature_tables import FeatureTables
from feed_ranker.preprocessing.metadata import ANONYMOUS, MetadataEncoding


def _duration_columns(values: np.ndarray) -> dict[str, np.ndarray]:
    """Derive numeric and categorical duration columns without changing fitted state."""
    bands = np.searchsorted([10_000, 30_000, 60_000], values, side="left").astype(np.int64) + 1
    bands[values == 0] = 0
    return {
        # Preserve the original torch float32 transformation exactly.
        "log1p": torch.log1p(torch.from_numpy(values).clamp(min=0)).numpy(),
        "missing": (values == 0).astype(np.float32),
        "band": bands,
    }


def fit_features(
    train: pd.DataFrame, basic: pd.DataFrame | None = None, profiles: pd.DataFrame | None = None
) -> FeatureTables:
    users = np.sort(train.user_id.unique()).astype(np.int64)
    warm_items = np.sort(train.video_id.unique()).astype(np.int64)
    if not len(users) or not len(warm_items):
        raise ValueError("Training events cannot be empty")
    if (basic is None) != (profiles is None):
        raise ValueError("Provide both item and user snapshots, or neither")
    metadata = None if basic is None else MetadataEncoding.fit(users, warm_items, basic, profiles)
    items = warm_items if metadata is None else np.union1d(warm_items, metadata.videos)
    columns = {
        "user_code": np.arange(len(users) + 1, dtype=np.int64),
        "item_code": np.r_[0, pd.Index(warm_items).get_indexer(items) + 1],
    }
    entities = {"user_code": "user", "item_code": "item"}
    vocab_sizes = {"user_code": len(users) + 1, "item_code": len(warm_items) + 1}
    descriptions: dict[str, dict[str, Any]] = {
        "user_code": {"missing": "zero OOV"},
        "item_code": {"missing": "zero OOV"},
    }

    medians = train.loc[train.duration_ms.gt(0)].groupby("video_id").duration_ms.median()
    duration_sources = {
        "train_duration": (
            np.r_[0, medians.reindex(items).fillna(0)].astype(np.float32),
            "train-fitted item median",
        )
    }
    if metadata is not None:
        rows = pd.Index(metadata.videos).get_indexer(items) + 1
        duration_sources["metadata_duration"] = (
            np.r_[0, metadata.basic_duration[rows]].astype(np.float32),
            "snapshot; historical availability unverified",
        )
    duration_descriptions = {
        "log1p": {
            "units": "log1p(milliseconds)",
            "transform": "torch-float32-log1p-v1",
            "missing": "zero with separate indicator",
        },
        "missing": {"missing": "one indicates missing"},
        "band": {"transform": "duration-bands-v1", "missing": "zero OOV/missing"},
    }
    for prefix, (duration, availability) in duration_sources.items():
        for suffix, values in _duration_columns(duration).items():
            name = f"{prefix}_{suffix}"
            columns[name] = values
            entities[name] = "item"
            descriptions[name] = {**duration_descriptions[suffix], "availability": availability}
        vocab_sizes[f"{prefix}_band"] = 5

    if metadata is not None:
        columns["author_code"] = np.r_[0, metadata.author_codes[rows]]
        entities["author_code"] = "item"
        vocab_sizes["author_code"] = len(metadata.author_vocabulary) + 1
        descriptions["author_code"] = {
            "missing": "zero OOV",
            "availability": "snapshot; historical availability unverified",
        }
        for i, name in enumerate(ANONYMOUS):
            columns[name] = metadata.anonymous_codes[:, i].copy()
            entities[name] = "user"
            vocab_sizes[name] = len(metadata.anonymous_vocabularies[i]) + 1
            descriptions[name] = {
                "missing": "zero null/sentinel/OOV",
                "availability": "anonymous snapshot; historical availability unverified",
            }
    return FeatureTables(users, items, columns, entities, vocab_sizes, descriptions)
