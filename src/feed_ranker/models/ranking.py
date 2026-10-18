"""Ranking model specification and tensor construction, shared by train and inference."""

from dataclasses import dataclass

import torch
from torch import nn

from feed_ranker.data.config import FeatureConfig
from feed_ranker.models.components.feature_encoder import EmbeddingEncoder
from feed_ranker.models.wide_deep import DeepNN, WideDeepNN, WideNN

ARCHITECTURES = ("linear", "wide", "deep", "wide_deep")


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    hidden: tuple[int, ...] = ()
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(self.hidden))
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Unknown ranking architecture")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if self.architecture in ("deep", "wide_deep"):
            if not self.hidden or any(type(n) is not int or n <= 0 for n in self.hidden):
                raise ValueError("Deep models require positive hidden widths")
        elif self.hidden:
            raise ValueError("Linear models have no hidden layers")


def build_model(features: FeatureConfig, spec: ModelSpec) -> nn.Module:
    """Construct shared modules; caller owns training seeds and checkpoint loading."""
    dims = [features.deep_input_dims, *spec.hidden, 1]
    if spec.architecture in ("linear", "wide"):
        model = WideNN(features.wide_input_dims, 1, features, categorical_main_effects=True)
    elif spec.architecture == "deep":
        model = DeepNN(features, dims, spec.dropout)
    else:
        model = WideDeepNN(features, dims, spec.dropout, categorical_main_effects=True)
    for module in model.modules():
        if isinstance(module, EmbeddingEncoder):
            for embedding in module.embeddings.values():
                embedding.padding_idx = 0
                with torch.no_grad():
                    embedding.weight[0].zero_()
    return model
