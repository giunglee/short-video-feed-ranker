"""
Shared data / feature / label configs.

  DenseFeatureConfig      — numeric columns (optional log1p)
  EmbeddingFeatureConfig  — high-card categoricals → nn.Embedding
  CrossFeatureConfig      — ETL 0/1 columns selected for the wide path

Used by: KuaiRecDataset, feature encoders, experiments.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from feed_ranker.data.schema import KEYS, LABEL


@dataclass
class DenseFeatureConfig:
    name: str = ""
    transform: str = "identity"

    def __post_init__(self) -> None:
        if self.transform not in ("identity", "log1p_nonnegative"):
            raise ValueError(f"Unsupported dense transform: {self.transform}")


@dataclass
class EmbeddingFeatureConfig:
    """Categorical id → dense embedding (deep path / embed-linear baselines)."""

    name: str = ""
    # Fill from meta.json via FeatureConfig.apply_vocab before building encoders
    vocab_size: int | None = None
    embedding_dim: int = 8


@dataclass
class CrossFeatureConfig:
    """
    Precomputed wide-path binary column (float 0/1 in parquet).

    Columns are materialized in KuaiRecPreprocessor; experiments select names.
    """

    name: str = ""


@dataclass
class FeatureConfig:
    dense_configs: list[DenseFeatureConfig] = field(default_factory=list)
    embedding_configs: list[EmbeddingFeatureConfig] = field(default_factory=list)
    cross_feature_configs: list[CrossFeatureConfig] = field(default_factory=list)

    @property
    def dense_cols(self) -> list[str]:
        return [c.name for c in self.dense_configs]

    @property
    def embedding_cols(self) -> list[str]:
        return [c.name for c in self.embedding_configs]

    @property
    def cross_feature_cols(self) -> list[str]:
        return [c.name for c in self.cross_feature_configs]

    @property
    def categorical_cols(self) -> list[str]:
        """Columns stored as long ids (embedding sources)."""
        return self.embedding_cols

    @property
    def feature_cols(self) -> list[str]:
        return list(
            dict.fromkeys([*self.embedding_cols, *self.cross_feature_cols, *self.dense_cols])
        )

    @property
    def dense_dims(self) -> int:
        return len(self.dense_configs)

    @property
    def embedding_dims(self) -> int:
        return sum(c.embedding_dim for c in self.embedding_configs)

    @property
    def cross_feature_dims(self) -> int:
        return len(self.cross_feature_configs)

    @property
    def deep_input_dims(self) -> int:
        """Dense + concatenated embedding dims (deep / embed-linear)."""
        return self.dense_dims + self.embedding_dims

    @property
    def wide_input_dims(self) -> int:
        """Dense + binary cross dims (paper-wide linear)."""
        return self.dense_dims + self.cross_feature_dims

    @property
    def feature_dims(self) -> int:
        """Alias for deep_input_dims (LinearRegression / DeepNN)."""
        return self.deep_input_dims

    def apply_vocab(self, vocab_sizes: dict[str, int]) -> FeatureConfig:
        """Fill vocab_size on embedding configs from preprocessor meta."""
        needed = self.categorical_cols
        missing = [n for n in needed if n not in vocab_sizes]
        if missing:
            raise KeyError(
                f"meta.json missing vocab sizes for: {missing}. "
                f"Available keys: {sorted(vocab_sizes)}"
            )
        for cfg in self.embedding_configs:
            cfg.vocab_size = int(vocab_sizes[cfg.name])
        return self


@dataclass
class LabelConfig:
    """Single target column for v1 (e.g. watch_ratio)."""

    name: str = ""

    @property
    def label_col(self) -> str:
        if not self.name:
            raise ValueError("LabelConfig.name must be set")
        return self.name


@dataclass
class DataConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)

    @property
    def feature_cols(self) -> list[str]:
        return self.feature.feature_cols

    @property
    def label_col(self) -> str:
        return self.label.label_col

    @property
    def columns(self) -> list[str]:
        """Parquet columns to load (features + label)."""
        return list(dict.fromkeys([*self.feature.feature_cols, self.label_col]))


@dataclass
class FeatureSpec:
    """Explicit model inputs; cross conditions compare raw values before encoding."""

    categorical: list[str]
    dense: list[str]
    crosses: dict[str, dict[str, str | int | float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> FeatureSpec:
        if not isinstance(spec, dict):
            raise ValueError("Feature specification must be a JSON object")
        if set(spec) - {"categorical", "dense", "crosses"}:
            raise ValueError("Unknown feature-specification fields")
        if not {"categorical", "dense"} <= set(spec):
            raise ValueError("Feature specification requires categorical and dense lists")
        features = cls(**deepcopy(spec))
        features.validate()
        return features

    @property
    def required_columns(self) -> set[str]:
        cross_inputs = {name for conditions in self.crosses.values() for name in conditions}
        return set(self.categorical + self.dense) | cross_inputs

    def validate(self) -> None:
        for name, columns in (("categorical", self.categorical), ("dense", self.dense)):
            if not isinstance(columns, list) or not all(
                isinstance(column, str) and column.strip() for column in columns
            ):
                raise ValueError(f"{name} must be a list of nonempty column names")
        if not isinstance(self.crosses, dict):
            raise ValueError("crosses must map output names to equality conditions")
        for name, conditions in self.crosses.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Cross names must be nonempty strings")
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError(f"Cross {name!r} must contain equality conditions")
            for column, value in conditions.items():
                if not isinstance(column, str) or not column.strip():
                    raise ValueError(f"Cross {name!r} has an invalid input column")
                if type(value) not in (str, int, float) or (
                    isinstance(value, float) and not math.isfinite(value)
                ):
                    raise ValueError(f"Cross {name!r}: expected a string or finite number")

        output_names = self.categorical + self.dense + list(self.crosses)
        if len(set(output_names)) != len(output_names):
            raise ValueError("Feature names must be unique")

        all_names = self.required_columns | set(self.crosses)
        if any(name.startswith("raw_") for name in all_names):
            raise ValueError("raw_* names are reserved for evaluation metadata")
        if all_names & {LABEL, "play_duration", "date", "timestamp", "time"}:
            raise ValueError("Label/time columns cannot be selected as model features")
        if set(self.crosses) & set(KEYS):
            raise ValueError("Cross names cannot overwrite entity IDs")


def resolve_feature_config(metadata: dict[str, Any], model: str, use_ids: bool) -> FeatureConfig:
    spec = metadata["spec"]
    categorical = [
        name for name in spec["categorical"] if use_ids or name not in ("user_id", "video_id")
    ]
    embeddings = []
    if model != "wide_nn":
        # reasonable default, not from any specific production config
        embeddings = [
            EmbeddingFeatureConfig(name, len(metadata["vocab"][name]) + 1, 8)
            for name in categorical
        ]
    crosses = []
    if model in ("wide_nn", "wide_deep_nn"):
        crosses = [CrossFeatureConfig(name) for name in spec.get("crosses", {})]
    features = FeatureConfig(
        dense_configs=[DenseFeatureConfig(name, "log1p_nonnegative") for name in spec["dense"]],
        embedding_configs=embeddings,
        cross_feature_configs=crosses,
    )
    if not features.dense_configs:
        raise ValueError("This experiment requires at least one dense feature")
    return features
