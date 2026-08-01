"""
Shared data / feature / label configs.

  DenseFeatureConfig      — numeric columns (optional log1p)
  EmbeddingFeatureConfig  — high-card categoricals → nn.Embedding
  CrossFeatureConfig      — ETL 0/1 columns selected for the wide path

Used by: KuaiRecDataset, feature encoders, experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DenseFeatureConfig:
    name: str = ""
    normalize: bool = False


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
            dict.fromkeys(
                [*self.embedding_cols, *self.cross_feature_cols, *self.dense_cols]
            )
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
