"""Fit user/item feature lookups from training events and optional snapshots.

Public field definitions: https://kuairand.com/. Categorical vocabularies use
training entities only. Snapshot historical availability is unverified; monthly
statistics and outcomes are excluded. Runtime lookup lives in data.entity_features.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.data.entity_features import EntityFeatureLookup

ANONYMOUS = tuple(f"onehot_feat{i}" for i in range(18))


def _snapshot_rows(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame[key].duplicated().any():
        raise ValueError("Metadata entity keys must be unique")
    if frame[key].isna().any():
        raise ValueError("Metadata entity keys cannot be null")
    return frame.set_index(key).sort_index()


def _encode_categories(training: pd.Series, values: pd.Series) -> tuple[np.ndarray, int]:
    """Fit compact embedding codes; reserve zero for null/sentinel/unseen values."""
    valid = training.dropna()
    if not np.isfinite(valid.to_numpy(dtype=float)).all() or (valid % 1 != 0).any():
        raise ValueError("Categorical snapshot values must be finite integers or null")
    vocabulary = np.sort(valid.loc[valid.ne(-124)].unique()).astype(np.int64)
    # Raw zero is a legitimate category; it is distinct from encoded unknown zero.
    category_codes = pd.Index(vocabulary).get_indexer(values) + 1
    return np.r_[0, category_codes], len(vocabulary) + 1


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


def fit_entity_features(
    train: pd.DataFrame, basic: pd.DataFrame | None = None, profiles: pd.DataFrame | None = None
) -> EntityFeatureLookup:
    """Materialize attributes once per entity, independently of model input selection.

    Snapshot-only items have lookup rows but retain item_code=0. Their metadata
    can still be known. Fitting uses no validation events or outcome columns.
    """
    user_ids = np.sort(train.user_id.unique()).astype(np.int64)
    warm_item_ids = np.sort(train.video_id.unique()).astype(np.int64)
    if not len(user_ids) or not len(warm_item_ids):
        raise ValueError("Training events cannot be empty")
    if (basic is None) != (profiles is None):
        raise ValueError("Provide both item and user snapshots, or neither")
    item_ids = warm_item_ids
    if basic is not None and profiles is not None:
        basic = _snapshot_rows(basic, "video_id")
        profiles = _snapshot_rows(profiles, "user_id").reindex(user_ids)
        item_ids = np.union1d(warm_item_ids, basic.index.to_numpy(dtype=np.int64))

    columns = {
        "user_code": np.arange(len(user_ids) + 1, dtype=np.int64),
        "item_code": np.r_[0, pd.Index(warm_item_ids).get_indexer(item_ids) + 1],
    }
    entities = {"user_code": "user", "item_code": "item"}
    vocab_sizes = {"user_code": len(user_ids) + 1, "item_code": len(warm_item_ids) + 1}
    descriptions: dict[str, dict[str, Any]] = {
        "user_code": {"missing": "zero OOV"},
        "item_code": {"missing": "zero OOV"},
    }
    medians = train.loc[train.duration_ms.gt(0)].groupby("video_id").duration_ms.median()
    duration_sources = {
        "train_duration": (
            np.r_[0, medians.reindex(item_ids).fillna(0)].astype(np.float32),
            "train-fitted item median",
        )
    }
    if basic is not None:
        duration = basic.video_duration.reindex(item_ids).to_numpy(dtype=np.float64)
        duration = np.where(np.isfinite(duration) & (duration > 0), duration, 0)
        duration_sources["metadata_duration"] = (
            np.r_[0, duration].astype(np.float32),
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

    if basic is not None and profiles is not None:
        columns["author_code"], vocab_sizes["author_code"] = _encode_categories(
            basic.author_id.reindex(warm_item_ids), basic.author_id.reindex(item_ids)
        )
        entities["author_code"] = "item"
        descriptions["author_code"] = {
            "missing": "zero OOV",
            "availability": "snapshot; historical availability unverified",
        }
        for name in ANONYMOUS:
            columns[name], vocab_sizes[name] = _encode_categories(profiles[name], profiles[name])
            entities[name] = "user"
            descriptions[name] = {
                "missing": "zero null/sentinel/OOV",
                "availability": "anonymous snapshot; historical availability unverified",
            }
    return EntityFeatureLookup(user_ids, item_ids, columns, entities, vocab_sizes, descriptions)
