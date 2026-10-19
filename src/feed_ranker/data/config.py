"""Dataset columns and target configuration."""

from dataclasses import dataclass, field

from feed_ranker.features.config import FeatureConfig


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
