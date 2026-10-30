"""Deep ranking model: input layout, layers and forward computation.

Cheng et al., Wide & Deep Learning (2016), https://arxiv.org/abs/1606.07792.
Only the Deep branch is active. Inputs have already been transformed by ETL.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
from torch import nn

from feed_ranker.models.components.mlp import MLP


@dataclass(frozen=True)
class DeepConfig:
    """Concrete inputs: scalar float32 dense values and int64 categorical codes.

    Sparse keys are sorted before concatenation; dense order is explicit.
    This deterministic layout survives JSON key reordering. Vocabulary sizes
    include OOV row zero. No storage schema or provenance belongs here.
    """

    dense: tuple[str, ...] = ()
    sparse: Mapping[str, int] = field(default_factory=dict)
    hidden: tuple[int, ...] = (
        64,
        32,
    )  # reasonable default, not from any specific production config
    embedding_dim: int = 8  # reasonable default, not from any specific production config
    dropout: float = 0.0
    output: str = "long_view"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dense", tuple(self.dense))
        object.__setattr__(self, "sparse", MappingProxyType(dict(self.sparse)))
        object.__setattr__(self, "hidden", tuple(self.hidden))
        names = self.feature_names
        if not names or len(set(names)) != len(names) or any(not n for n in names):
            raise ValueError("Require unique nonempty dense/sparse feature names")
        if any(type(size) is not int or size < 1 for size in self.sparse.values()):
            raise ValueError("Sparse inputs require positive vocabulary sizes")
        if not self.hidden or any(type(n) is not int or n < 1 for n in self.hidden):
            raise ValueError("Require positive hidden widths")
        if (
            type(self.embedding_dim) is not int
            or self.embedding_dim < 1
            or not 0 <= self.dropout < 1
            or not self.output
        ):
            raise ValueError("Invalid embedding width, dropout or output name")

    def to_dict(self) -> dict[str, object]:
        """Serialize the immutable input definition with the architecture settings."""
        return {
            "dense": self.dense,
            "sparse": dict(self.sparse),
            "hidden": self.hidden,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "output": self.output,
        }

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*sorted(self.sparse), *self.dense)


class DeepModel(nn.Module):
    """Own architecture and parameters; forward accepts features only.

    State: immutable construction config and embedding/MLP parameters.
    Dataset paths, raw IDs, transforms and supervision are absent.
    """

    def __init__(self, config: DeepConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(config.sparse[name], config.embedding_dim, padding_idx=0)
                for name in sorted(config.sparse)
            }
        )
        width = len(config.sparse) * config.embedding_dim + len(config.dense)
        self.mlp = MLP([width, *config.hidden, 1], config.dropout)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = []
        length = None
        for name in self.config.feature_names:
            if name not in features:
                raise ValueError(f"Missing model feature: {name}")
            value = features[name]
            dtype = torch.int64 if name in self.embeddings else torch.float32
            if (
                value.ndim != 1
                or value.dtype != dtype
                or (length is not None and len(value) != length)
            ):
                raise ValueError(f"{name}: expected aligned {dtype} [B]")
            length = len(value)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite input")
            if name in self.embeddings:
                if (value < 0).any() or (value >= self.config.sparse[name]).any():
                    raise ValueError(f"{name}: code outside fitted vocabulary")
                values.append(self.embeddings[name](value))
            else:
                values.append(value.unsqueeze(1))
        return {self.config.output: self.mlp(torch.cat(values, dim=1))[:, 0]}
