"""
models/wide_deep.py

Paper: Cheng et al., "Wide & Deep Learning for Recommender Systems", DLRS 2016
Link:  https://arxiv.org/abs/1606.07792

  LinearRegression — embeddings + dense → Linear (simplest baseline)
  WideNN           — binary crosses + dense → Linear (paper-wide)
  DeepNN           — embeddings + dense → MLP (paper-deep)
  WideDeepNN       — wide_logit + deep_logit (paper W&D; sum, not one shared head)

Inputs are named [B] tensors; each model returns regression predictions [B, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from feed_ranker.data.config import FeatureConfig
from feed_ranker.models.components.feature_encoder import (
    CategoricalMainEffects,
    CrossFeatureEncoder,
    DenseFeatureEncoder,
    EmbeddingEncoder,
)
from feed_ranker.models.components.mlp import MLP


class LinearRegression(nn.Module):
    """
    Linear baseline for Cheng et al. (2016): embeddings + dense → one Linear.
    Cross binaries belong on the wide path.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FeatureConfig,
    ) -> None:
        super().__init__()
        if not config.embedding_configs or not config.dense_configs:
            raise ValueError("LinearRegression expects embedding_configs and dense_configs")
        self.config = config
        self.dense_encoder = DenseFeatureEncoder(config.dense_configs)
        self.embedding_encoder = EmbeddingEncoder(config.embedding_configs)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.dense_encoder({n: features[n] for n in self.config.dense_cols})
        emb = self.embedding_encoder({n: features[n] for n in self.config.embedding_cols})
        return self.linear(torch.cat([dense, emb], dim=-1))


class WideNN(nn.Module):
    """Cheng et al. (2016), Wide & Deep: optional crosses + dense → Linear."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FeatureConfig,
        *,
        categorical_main_effects: bool = False,
    ) -> None:
        super().__init__()
        if not config.dense_configs:
            raise ValueError("WideNN expects dense_configs")
        self.config = config
        self.dense_encoder = DenseFeatureEncoder(config.dense_configs)
        self.cross_encoder = CrossFeatureEncoder(config.cross_feature_configs)
        self.linear = nn.Linear(input_dim, output_dim)
        self.main_effects = (
            CategoricalMainEffects(config.embedding_configs) if categorical_main_effects else None
        )
        if categorical_main_effects and output_dim != 1:
            raise ValueError("Categorical main effects require scalar regression")

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.dense_encoder({n: features[n] for n in self.config.dense_cols})
        if self.config.cross_feature_cols:
            crosses = self.cross_encoder(
                {name: features[name] for name in self.config.cross_feature_cols}
            )
        else:
            crosses = dense.new_empty((dense.shape[0], 0))
        prediction = self.linear(torch.cat([dense, crosses], dim=-1))
        if self.main_effects is not None:
            prediction = prediction + self.main_effects(features)
        return prediction


class DeepNN(nn.Module):
    """Cheng et al. (2016), Wide & Deep: embeddings + dense → MLP."""

    def __init__(
        self,
        config: FeatureConfig,
        nn_dims: list[int] | None = None,
        dropout_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if not config.embedding_configs or not config.dense_configs:
            raise ValueError("DeepNN expects embedding_configs and dense_configs")
        self.config = config
        self.dense_encoder = DenseFeatureEncoder(config.dense_configs)
        self.embedding_encoder = EmbeddingEncoder(config.embedding_configs)
        if nn_dims is None:
            nn_dims = [config.deep_input_dims, 16, 8, 1]
        self.mlp = MLP(nn_dims, dropout_ratio)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.dense_encoder({n: features[n] for n in self.config.dense_cols})
        emb = self.embedding_encoder({n: features[n] for n in self.config.embedding_cols})
        return self.mlp(torch.cat([dense, emb], dim=-1))


class WideDeepNN(nn.Module):
    """
    Cheng et al. (2016), Wide & Deep: two towers, sum logits.

      logit = wide_linear(crosses || dense) + deep_mlp(embed || dense)
    """

    def __init__(
        self,
        config: FeatureConfig,
        nn_dims: list[int] | None = None,
        dropout_ratio: float = 0.5,
        *,
        categorical_main_effects: bool = False,
    ) -> None:
        super().__init__()
        if not config.embedding_configs or not config.dense_configs:
            raise ValueError("WideDeepNN expects embedding_configs and dense_configs")
        self.config = config
        self.dense_encoder = DenseFeatureEncoder(config.dense_configs)
        self.cross_encoder = CrossFeatureEncoder(config.cross_feature_configs)
        self.embedding_encoder = EmbeddingEncoder(config.embedding_configs)
        self.wide_linear = nn.Linear(config.wide_input_dims, 1)
        if nn_dims is None:
            nn_dims = [config.deep_input_dims, 16, 8, 1]
        if nn_dims[-1] != 1:
            raise ValueError(f"deep nn_dims must end with 1 logit, got {nn_dims}")
        self.deep_mlp = MLP(nn_dims, dropout_ratio)
        self.main_effects = (
            CategoricalMainEffects(config.embedding_configs) if categorical_main_effects else None
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.dense_encoder({n: features[n] for n in self.config.dense_cols})
        if self.config.cross_feature_cols:
            crosses = self.cross_encoder(
                {name: features[name] for name in self.config.cross_feature_cols}
            )
        else:
            crosses = dense.new_empty((dense.shape[0], 0))
        emb = self.embedding_encoder({n: features[n] for n in self.config.embedding_cols})
        wide_logit = self.wide_linear(torch.cat([dense, crosses], dim=-1))
        deep_logit = self.deep_mlp(torch.cat([dense, emb], dim=-1))
        if self.main_effects is not None:
            wide_logit = wide_logit + self.main_effects(features)
        return wide_logit + deep_logit
