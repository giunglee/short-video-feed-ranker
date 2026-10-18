"""Ranking recipes and model construction shared by training and inference."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.data.config import (
    CrossFeatureConfig,
    DataConfig,
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
    FeatureConfig,
    LabelConfig,
)
from feed_ranker.data.kuairec_dataset import KuaiRecDataset
from feed_ranker.models.components.feature_encoder import EmbeddingEncoder
from feed_ranker.models.wide_deep import DeepNN, WideDeepNN, WideNN

ARCHITECTURES = ("linear", "wide", "deep", "wide_deep")


@dataclass(frozen=True)
class Recipe:
    architecture: str
    hidden: tuple[int, ...] = ()
    lr: float = 0.001
    weight_decay: float = 0.0
    feature_set: str = "profiles"
    cross_family: str = "both"
    embedding_dim: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(self.hidden))
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Unknown architecture")
        if self.feature_set not in ("profiles", "ids_duration"):
            raise ValueError("Unknown feature set")
        if self.cross_family not in ("both", "age", "follow", "none"):
            raise ValueError("Unknown cross family")
        if self.architecture in ("deep", "wide_deep") and (
            not self.hidden or any(type(d) is not int or d <= 0 for d in self.hidden)
        ):
            raise ValueError("Deep models require positive hidden widths")
        if self.architecture in ("linear", "wide") and self.hidden:
            raise ValueError("Linear models have no hidden layers")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0:
            raise ValueError("Embedding width must be a positive integer")


def feature_config(metadata: dict[str, Any], recipe: Recipe) -> FeatureConfig:
    names = ["user_id", "video_id"]
    if recipe.feature_set == "profiles":
        names += ["age_range", "follow_user_num_range", "duration_band"]
    missing = set(names) - set(metadata["vocab"])
    if missing:
        raise ValueError(f"Prepared data lacks categorical features: {sorted(missing)}")
    crosses = []
    if recipe.architecture in ("wide", "wide_deep") and recipe.feature_set == "profiles":
        for name, conditions in metadata["spec"].get("crosses", {}).items():
            # Main effects are supplied by categorical coefficients, not duplicated
            # by the older one-hot columns stored in this prepared extract.
            if set(conditions) == {"age_range", "duration_band"}:
                family = "age"
            elif set(conditions) == {"follow_user_num_range", "duration_band"}:
                family = "follow"
            else:
                continue
            if recipe.cross_family in ("both", family):
                crosses.append(CrossFeatureConfig(name))
        if recipe.cross_family != "none" and not crosses:
            raise ValueError("Requested interaction family has no prepared columns")
    return FeatureConfig(
        dense_configs=[DenseFeatureConfig("video_duration", "log1p_nonnegative")],
        embedding_configs=[
            EmbeddingFeatureConfig(name, len(metadata["vocab"][name]) + 1, recipe.embedding_dim)
            for name in names
        ],
        cross_feature_configs=crosses,
    )


def build_architecture(features: FeatureConfig, recipe: Recipe, seed: int) -> nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dims = [features.deep_input_dims, *recipe.hidden, 1]
    if recipe.architecture in ("linear", "wide"):
        model = WideNN(features.wide_input_dims, 1, features, categorical_main_effects=True)
    elif recipe.architecture == "deep":
        model = DeepNN(features, dims, 0.0)
    else:
        model = WideDeepNN(features, dims, 0.0, categorical_main_effects=True)
    # Unseen categorical IDs have no estimated identity effect. Freeze their zero
    # vectors consistently in all architectures, instead of random OOV predictions.
    for module in model.modules():
        if isinstance(module, EmbeddingEncoder):
            for embedding in module.embeddings.values():
                embedding.padding_idx = 0
                with torch.no_grad():
                    embedding.weight[0].zero_()
    return model


def make_dataset(directory: Path, recipe: Recipe) -> KuaiRecDataset:
    import json

    metadata = json.loads((directory / "meta.json").read_text())
    return KuaiRecDataset(
        DataConfig(feature_config(metadata, recipe), LabelConfig("watch_ratio")), directory
    )
