"""
Feature encoders — model-side transforms from batch dict → concatenated tensors.

  DenseFeatureEncoder  — numerics (optional log1p)
  EmbeddingEncoder     — categorical ids → nn.Embedding
  CrossFeatureEncoder  — precomputed binary crosses → [B, C] (wide path)

See: Cheng et al. Wide & Deep — wide uses sparse crosses; deep uses embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from feed_ranker.features.config import (
    CrossFeatureConfig,
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
)


class DenseFeatureEncoder(nn.Module):
    """Numeric columns → [B, D] (optional per-column log1p)."""

    def __init__(self, configs: list[DenseFeatureConfig]) -> None:
        super().__init__()
        self.configs = {c.name: c.transform for c in configs}

    def _normalize(self, vals: torch.Tensor, transform: str) -> torch.Tensor:
        vals = vals.to(torch.float32)
        if transform == "log1p_nonnegative":
            vals = torch.log1p(vals.clamp(min=0))
        return vals

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.configs:
            return torch.zeros(0, 0)

        outs: list[torch.Tensor] = []
        for name, transform in self.configs.items():
            if name not in features:
                raise ValueError(f"{name} not found in features keys {list(features.keys())}")
            vals = self._normalize(features[name], transform)
            if vals.ndim == 1:
                vals = vals.unsqueeze(-1)
            outs.append(vals)
        return torch.cat(outs, dim=-1)


class EmbeddingEncoder(nn.Module):
    """Categorical ids → concatenated embeddings [B, sum(E)]."""

    def __init__(self, configs: list[EmbeddingFeatureConfig]) -> None:
        super().__init__()
        for config in configs:
            if config.vocab_size is None:
                raise ValueError(
                    f"EmbeddingFeatureConfig('{config.name}').vocab_size is None; "
                    "call FeatureConfig.apply_vocab(...) before building encoders."
                )
        self.embeddings = nn.ModuleDict(
            {
                c.name: nn.Embedding(
                    num_embeddings=c.vocab_size,
                    embedding_dim=c.embedding_dim,
                )
                for c in configs
            }
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.embeddings:
            return torch.zeros(0, 0)

        outs: list[torch.Tensor] = []
        for name, emb in self.embeddings.items():
            if name not in features:
                raise ValueError(f"{name} not found in features keys {list(features.keys())}")
            outs.append(emb(features[name].long()))
        return torch.cat(outs, dim=-1)


class CrossFeatureEncoder(nn.Module):
    """Precomputed equality-cross binaries → [B, C] for the wide path."""

    def __init__(self, configs: list[CrossFeatureConfig]) -> None:
        super().__init__()
        self.names = [c.name for c in configs]

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.names:
            return torch.zeros(0, 0)

        outs: list[torch.Tensor] = []
        for name in self.names:
            if name not in features:
                raise ValueError(f"{name} not found in features keys {list(features.keys())}")
            vals = features[name].to(torch.float32)
            if vals.ndim == 1:
                vals = vals.unsqueeze(-1)
            outs.append(vals)
        return torch.cat(outs, dim=-1)


class CategoricalMainEffects(nn.Module):
    """One coefficient per category, equivalent to a one-hot linear term.

    Cheng et al. (2016), https://arxiv.org/abs/1606.07792.
    Index zero is the training-vocabulary OOV category, with a fixed zero effect.
    """

    def __init__(self, configs: list[EmbeddingFeatureConfig]) -> None:
        super().__init__()
        if not configs or any(c.vocab_size is None for c in configs):
            raise ValueError("Main effects require scalar categoricals with fitted vocabularies")
        self.coefficients = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocab_size, 1, padding_idx=0) for c in configs}
        )
        for coefficients in self.coefficients.values():
            nn.init.zeros_(coefficients.weight)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = [coefficients(features[name]) for name, coefficients in self.coefficients.items()]
        return torch.stack(terms).sum(dim=0)
