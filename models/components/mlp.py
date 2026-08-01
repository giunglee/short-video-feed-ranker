"""
models/components/mlp.py

Reusable MLP (Linear + activation + dropout). No single paper — shared
building block, reused by every model that needs a plain feed-forward tower.
"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims: list[int] =[16, 8, 1], dropout_ratio=0.5):
        super().__init__()
        
        if len(dims) < 2:
            raise ValueError(f"Error: NN should have at least 1 input and 1 output. Expected: >=2, Actual: {len(dims)}")

        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_ratio),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers.forward(input)

