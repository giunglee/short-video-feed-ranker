"""AdamW with fixed learning rate; embeddings, biases, and scalar/vector parameters
are exempt from weight decay. Matrix weights receive the configured decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import nn


@dataclass
class OptimizerConfig:
    # reasonable default, not from any specific production config
    lr: float = 1e-3
    weight_decay: float = 0.0

    def validate(self) -> None:
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    config.validate()
    groups = _parameter_groups(model, config.weight_decay)
    return torch.optim.AdamW(groups, lr=config.lr)


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    embedding_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = (
            id(parameter) in embedding_parameters or parameter.ndim <= 1 or name.endswith(".bias")
        )
        if exempt:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("No trainable parameters")
    return groups


def optimizer_contract(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Record the supplied optimizer's type, settings and ordered parameter names."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        if any(id(parameter) not in names for parameter in group["params"]):
            raise ValueError("Optimizer parameters must belong to the supplied model")
        groups.append(
            {
                **{key: value for key, value in group.items() if key != "params"},
                "params": [names[id(parameter)] for parameter in group["params"]],
            }
        )
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": groups,
    }
