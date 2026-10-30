"""Fit user/item feature lookups from training events and optional snapshots.

Public field definitions: https://kuairand.com/. Categorical vocabularies use
training entities only. Snapshot historical availability is unverified; monthly
statistics and outcomes are excluded. Fitted state is reused by offline example
materialization and candidate scoring; Dataset loading performs no feature joins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.persistence import write_json

ANONYMOUS = tuple(f"onehot_feat{i}" for i in range(18))


@dataclass
class FeatureState:
    """Frozen features for offline example preparation and candidate scoring.

    Each group follows its own sorted raw IDs, with an unknown fallback at row
    zero. Lookup positions are not embedding codes: snapshot-only items have
    metadata rows but item_code=0. Training batches do not use this state.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_features: dict[str, np.ndarray]
    item_features: dict[str, np.ndarray]
    vocab_sizes: dict[str, int]
    descriptions: dict[str, dict[str, Any]]
    _user_index: pd.Index = field(init=False, repr=False)
    _item_index: pd.Index = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if set(self.user_features) & set(self.item_features):
            raise ValueError("User and item feature names must be disjoint")
        if set(self.vocab_sizes) - set(self.feature_names):
            raise ValueError("Vocabulary definitions require known features")
        for keys, columns in (
            (self.user_ids, self.user_features),
            (self.item_ids, self.item_features),
        ):
            if keys.dtype != np.int64 or keys.ndim != 1 or (keys < 0).any():
                raise ValueError("Entity keys must be nonnegative int64 vectors")
            if len(keys) and not np.all(keys[1:] > keys[:-1]):
                raise ValueError("Entity keys must be sorted and unique")
            for name, values in columns.items():
                categorical = name in self.vocab_sizes
                dtype = np.int64 if categorical else np.float32
                if values.shape != (len(keys) + 1,) or values.dtype != dtype:
                    raise ValueError(f"{name}: feature shape/dtype mismatch")
                if not np.isfinite(values).all():
                    raise ValueError(f"{name}: nonfinite fitted feature")
                if categorical and (
                    type(self.vocab_sizes[name]) is not int
                    or self.vocab_sizes[name] < 1
                    or (values < 0).any()
                    or (values >= self.vocab_sizes[name]).any()
                ):
                    raise ValueError(f"{name}: categorical code outside fitted vocabulary")
        self._user_index = pd.Index(self.user_ids)
        self._item_index = pd.Index(self.item_ids)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.user_features, *self.item_features)

    def transform(
        self, user_ids: np.ndarray, item_ids: np.ndarray, names: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        """Assemble aligned feature columns without fitting or reading source data."""
        if (
            user_ids.ndim != 1
            or user_ids.shape != item_ids.shape
            or user_ids.dtype != np.int64
            or item_ids.dtype != np.int64
        ):
            raise ValueError("Require aligned int64 raw user/item vectors")
        if (user_ids < 0).any() or (item_ids < 0).any():
            raise ValueError("Raw IDs cannot be negative")
        selected = self.feature_names if names is None else names
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - set(self.feature_names)
        ):
            raise ValueError("Require unique known feature names")
        values = {}
        for index, ids, columns in (
            (self._user_index, user_ids, self.user_features),
            (self._item_index, item_ids, self.item_features),
        ):
            rows = index.get_indexer(ids) + 1
            for name in selected:
                if name in columns:
                    values[name] = columns[name][rows]
        return {name: values[name] for name in selected}

    def save(self, directory: Path) -> None:
        np.savez(
            directory / "feature-state.npz",
            users=self.user_ids,
            items=self.item_ids,
            **self.user_features,
            **self.item_features,
        )
        write_json(
            directory / "feature-tables.json",
            {
                "user_features": list(self.user_features),
                "item_features": list(self.item_features),
                "vocab_sizes": self.vocab_sizes,
                "descriptions": self.descriptions,
            },
        )

    @classmethod
    def load(cls, directory: Path) -> FeatureState:
        definition = json.loads((directory / "feature-tables.json").read_text())
        # Existing fitted-scorer v2/v3 packages store the same arrays with a flat map.
        if "entities" in definition:
            owners = definition.pop("entities")
            if set(owners.values()) - {"user", "item"}:
                raise ValueError("Invalid legacy feature lookup definitions")
            definition["user_features"] = [n for n, owner in owners.items() if owner == "user"]
            definition["item_features"] = [n for n, owner in owners.items() if owner == "item"]
        users, items = definition.pop("user_features"), definition.pop("item_features")
        if len(set(users + items)) != len(users + items):
            raise ValueError("Feature names must be unique")
        with np.load(directory / "feature-state.npz", allow_pickle=False) as state:
            if set(state.files) != {"users", "items", *users, *items}:
                raise ValueError("Unexpected fitted-state arrays")
            return cls(
                state["users"],
                state["items"],
                {n: state[n] for n in users},
                {n: state[n] for n in items},
                **definition,
            )


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
) -> FeatureState:
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

    user_features = {"user_code": np.arange(len(user_ids) + 1, dtype=np.int64)}
    item_features = {
        "item_code": np.r_[0, pd.Index(warm_item_ids).get_indexer(item_ids) + 1],
    }
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
            item_features[name] = values
            descriptions[name] = {**duration_descriptions[suffix], "availability": availability}
        vocab_sizes[f"{prefix}_band"] = 5

    if basic is not None and profiles is not None:
        item_features["author_code"], vocab_sizes["author_code"] = _encode_categories(
            basic.author_id.reindex(warm_item_ids), basic.author_id.reindex(item_ids)
        )
        descriptions["author_code"] = {
            "missing": "zero OOV",
            "availability": "snapshot; historical availability unverified",
        }
        for name in ANONYMOUS:
            user_features[name], vocab_sizes[name] = _encode_categories(
                profiles[name], profiles[name]
            )
            descriptions[name] = {
                "missing": "zero null/sentinel/OOV",
                "availability": "anonymous snapshot; historical availability unverified",
            }
    return FeatureState(user_ids, item_ids, user_features, item_features, vocab_sizes, descriptions)
