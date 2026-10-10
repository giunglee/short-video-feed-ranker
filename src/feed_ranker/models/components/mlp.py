"""Feed-forward layers used by the Wide & Deep deep path (Cheng et al., 2016)."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Linear/ReLU/dropout hidden layers and a linear output; Cheng et al. (2016)."""

    def __init__(self, dims: list[int], dropout_ratio: float = 0.5) -> None:
        super().__init__()
        if len(dims) < 2 or any(width <= 0 for width in dims):
            raise ValueError("dims must contain at least two positive widths")
        layers: list[nn.Module] = []
        for input_width, output_width in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [nn.Linear(input_width, output_width), nn.ReLU(), nn.Dropout(dropout_ratio)]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
