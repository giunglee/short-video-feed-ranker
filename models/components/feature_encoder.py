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

from data.config import (
    CrossFeatureConfig,
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
)


class DenseFeatureEncoder(nn.Module):
    """Numeric columns → [B, D] (optional per-column log1p)."""

    def __init__(self, configs: list[DenseFeatureConfig]):
        super().__init__()
        self.configs = {c.name: c.normalize for c in configs}

    def _normalize(self, vals: torch.Tensor, normalize: bool) -> torch.Tensor:
        vals = vals.to(torch.float32)
        if normalize:
            vals = torch.log1p(vals.clamp(min=0))
        return vals

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.configs:
            return torch.zeros(0, 0)

        outs: list[torch.Tensor] = []
        for name, do_norm in self.configs.items():
            if name not in features:
                raise ValueError(
                    f"{name} not found in features keys {list(features.keys())}"
                )
            vals = self._normalize(features[name], do_norm)
            if vals.ndim == 1:
                vals = vals.unsqueeze(-1)
            outs.append(vals)
        return torch.cat(outs, dim=-1)


class EmbeddingEncoder(nn.Module):
    """Categorical ids → concatenated embeddings [B, sum(E)]."""

    def __init__(self, configs: list[EmbeddingFeatureConfig]):
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
                raise ValueError(
                    f"{name} not found in features keys {list(features.keys())}"
                )
            outs.append(emb(features[name].long()))
        return torch.cat(outs, dim=-1)


class CrossFeatureEncoder(nn.Module):
    """Precomputed equality-cross binaries → [B, C] for the wide path."""

    def __init__(self, configs: list[CrossFeatureConfig]):
        super().__init__()
        self.names = [c.name for c in configs]

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.names:
            return torch.zeros(0, 0)

        outs: list[torch.Tensor] = []
        for name in self.names:
            if name not in features:
                raise ValueError(
                    f"{name} not found in features keys {list(features.keys())}"
                )
            vals = features[name].to(torch.float32)
            if vals.ndim == 1:
                vals = vals.unsqueeze(-1)
            outs.append(vals)
        return torch.cat(outs, dim=-1)
