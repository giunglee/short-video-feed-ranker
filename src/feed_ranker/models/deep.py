"""Named-output Deep ranker using materialized numeric inputs.

Cheng et al., Wide & Deep Learning (2016), https://arxiv.org/abs/1606.07792.
Only the Deep branch is active. Preprocessing and supervision are external.
"""

from dataclasses import dataclass

import torch
from torch import nn

from feed_ranker.data.prepared import Batch, FieldSpec
from feed_ranker.models.components.mlp import MLP


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[str, ...]
    hidden: tuple[int, ...] = (
        64,
        32,
    )  # reasonable default, not from any specific production config
    embedding_dim: int = 8  # reasonable default, not from any specific production config
    dropout: float = 0.0
    output: str = "long_view"

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "hidden", tuple(self.hidden))
        if (
            not self.features
            or len(set(self.features)) != len(self.features)
            or any(not n for n in self.features)
        ):
            raise ValueError("Require ordered unique feature names")
        if not self.hidden or any(type(n) is not int or n < 1 for n in self.hidden):
            raise ValueError("Require positive hidden widths")
        if (
            type(self.embedding_dim) is not int
            or self.embedding_dim < 1
            or not 0 <= self.dropout < 1
            or not self.output
        ):
            raise ValueError("Invalid embedding width, dropout or output name")


class DeepModel(nn.Module):
    """Read an explicit feature allowlist; return one named [B] logit tensor."""

    def __init__(self, config: ModelConfig, schema: dict[str, FieldSpec]) -> None:
        super().__init__()
        if any(n not in schema or schema[n].role != "feature" for n in config.features):
            raise ValueError("Model inputs must be declared features, never targets/metadata")
        self.config = config
        self.schema = {n: schema[n] for n in config.features}
        self.embeddings = nn.ModuleDict(
            {
                n: nn.Embedding(s.vocab_size, config.embedding_dim, padding_idx=0)
                for n, s in self.schema.items()
                if s.vocab_size is not None
            }
        )
        width = sum(
            config.embedding_dim if s.vocab_size is not None else 1 for s in self.schema.values()
        )
        self.mlp = MLP([width, *config.hidden, 1], config.dropout)

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        features = batch.get("features", {})
        values = []
        length = None
        for name, spec in self.schema.items():
            if name not in features:
                raise ValueError(f"Missing model feature: {name}")
            v = features[name]
            dtype = torch.int64 if spec.dtype == "int64" else torch.float32
            if v.ndim != 1 or v.dtype != dtype or (length is not None and len(v) != length):
                raise ValueError(f"{name}: expected aligned {spec.dtype} [B]")
            length = len(v)
            if not torch.isfinite(v).all():
                raise ValueError(f"{name}: nonfinite input")
            if name in self.embeddings:
                if (v < 0).any() or (v >= spec.vocab_size).any():
                    raise ValueError(f"{name}: code outside fitted vocabulary")
                values.append(self.embeddings[name](v))
            else:
                values.append(v.to(torch.float32).unsqueeze(1))
        return {self.config.output: self.mlp(torch.cat(values, dim=1))[:, 0]}


def build_model(config: ModelConfig, schema: dict[str, FieldSpec]) -> DeepModel:
    return DeepModel(config, schema)
