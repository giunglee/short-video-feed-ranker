"""
training/optimizer.py

Optimizer factory (AdamW). Scheduler optional / deferred to trainer.
No paper — infrastructure.

NOTE (future): RecSys runs often use parameter groups —
  embedding weights → little/no weight decay,
  biases / LayerNorm → no weight decay,
  dense Linear weights → standard weight decay.
  Keep v1 as a single AdamW over model.parameters(); add grouping when
  embeddings are large or overfitting shows up (DIN / SASRec era).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OptimizerConfig:
    """Study defaults for public-data experiments — not from any production config."""

    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> tuple[torch.optim.Optimizer, None]:
    """Build AdamW over all trainable parameters. Returns (optimizer, scheduler=None)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embedding" in name or "bias" in name or param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if not decay_params and not no_decay_params:
        raise ValueError("model has no trainable parameters")

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps
    )
    lr_scheduler = None 

    return optimizer, lr_scheduler


if __name__ == "__main__":
    config = OptimizerConfig()
    model = nn.Linear(20, 30)
    optimizer, _ = build_optimizer(model, config)

    x = torch.randn(8, 20)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    print(
        f"OK: AdamW lr={config.lr} weight_decay={config.weight_decay} "
        f"n_params={sum(p.numel() for p in model.parameters())}"
    )
