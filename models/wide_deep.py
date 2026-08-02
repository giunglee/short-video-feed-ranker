"""
models/wide_deep.py

Paper: Cheng et al., "Wide & Deep Learning for Recommender Systems", DLRS 2016
Link:  https://arxiv.org/abs/1606.07792

  LinearRegression — embeddings + dense → Linear (simplest baseline)
  WideNN           — binary crosses + dense → Linear (paper-wide)
  DeepNN           — embeddings + dense → MLP (paper-deep)
  WideDeepNN       — wide_logit + deep_logit (paper W&D; sum, not one shared head)

Prerequisites: components/feature_encoder.py, components/mlp.py, training/trainer.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data.config import FeatureConfig
from models.components.feature_encoder import (
    CrossFeatureEncoder,
    DenseFeatureEncoder,
    EmbeddingEncoder,
)
from models.components.mlp import MLP


class LinearRegression(nn.Module):
    """
    Simplest baseline: embeddings + dense → one Linear.
    Cross binaries belong on the wide path.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FeatureConfig,
    ):
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
    """Paper-wide: sparse binary crosses + dense → Linear."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: FeatureConfig,
    ):
        super().__init__()
        if not config.cross_feature_configs or not config.dense_configs:
            raise ValueError(
                "WideNN expects cross_feature_configs and dense_configs"
            )
        self.config = config
        self.dense_encoder = DenseFeatureEncoder(config.dense_configs)
        self.cross_encoder = CrossFeatureEncoder(config.cross_feature_configs)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.dense_encoder({n: features[n] for n in self.config.dense_cols})
        crosses = self.cross_encoder(
            {n: features[n] for n in self.config.cross_feature_cols}
        )
        return self.linear(torch.cat([dense, crosses], dim=-1))


class DeepNN(nn.Module):
    """Paper-deep: embeddings + dense → MLP."""

    def __init__(
        self,
        config: FeatureConfig,
        nn_dims: list[int] | None = None,
        dropout_ratio: float = 0.5,
    ):
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
    Paper Wide & Deep: two towers, sum logits.

      logit = wide_linear(wide_feats) + deep_mlp(embed || dense)

    wide_mode controls what the wide Linear sees (deep always gets emb||dense):
      dense_cross -- dense || crosses (paper default)
      crosses     -- crosses only (ablate duplicated dense on wide)
      dense       -- dense only (ablate crosses)
    """

    def __init__(
        self,
        config: FeatureConfig,
        nn_dims: list[int] | None = None,
        dropout_ratio: float = 0.5,
        wide_mode: str = "dense_cross",
    ):
        super().__init__()
        if wide_mode not in ("dense_cross", "crosses", "dense"):
            raise ValueError(
                f"wide_mode must be dense_cross|crosses|dense, got {wide_mode!r}"
            )
        if not config.embedding_configs or not config.dense_configs:
            raise ValueError(
                "WideDeepNN expects embedding_configs and dense_configs (deep path)"
            )
        if wide_mode in ("dense_cross", "crosses") and not config.cross_feature_configs:
            raise ValueError(
                f"wide_mode={wide_mode} requires cross_feature_configs"
            )
        self.config = config
        self.wide_mode = wide_mode
        self.dense_encoder = DenseFeatureEncoder(config.dense_configs)
        self.cross_encoder = CrossFeatureEncoder(config.cross_feature_configs)
        self.embedding_encoder = EmbeddingEncoder(config.embedding_configs)
        if wide_mode == "dense_cross":
            wide_dim = config.dense_dims + config.cross_feature_dims
        elif wide_mode == "crosses":
            wide_dim = config.cross_feature_dims
        else:
            wide_dim = config.dense_dims
        self.wide_linear = nn.Linear(wide_dim, 1)
        if nn_dims is None:
            nn_dims = [config.deep_input_dims, 16, 8, 1]
        if nn_dims[-1] != 1:
            raise ValueError(f"deep nn_dims must end with 1 logit, got {nn_dims}")
        self.deep_mlp = MLP(nn_dims, dropout_ratio)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.dense_encoder({n: features[n] for n in self.config.dense_cols})
        emb = self.embedding_encoder({n: features[n] for n in self.config.embedding_cols})
        wide_parts: list[torch.Tensor] = []
        if self.wide_mode in ("dense_cross", "dense"):
            wide_parts.append(dense)
        if self.wide_mode in ("dense_cross", "crosses"):
            crosses = self.cross_encoder(
                {n: features[n] for n in self.config.cross_feature_cols}
            )
            wide_parts.append(crosses)
        wide_logit = self.wide_linear(torch.cat(wide_parts, dim=-1))
        deep_logit = self.deep_mlp(torch.cat([dense, emb], dim=-1))
        return wide_logit + deep_logit
