"""Plain checkpoint dictionaries and validated CPU epoch-boundary continuation."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

VERSION = "training-checkpoint-v1"


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Serialize a complete dictionary before atomically replacing the file."""
    temporary = path.with_suffix(".tmp")
    torch.save({"version": VERSION, **checkpoint}, temporary)
    temporary.replace(path)


def load_model_weights(model: nn.Module, checkpoint_path: Path) -> None:
    """Explicitly select saved weights; optimizer/RNG state is left untouched."""
    checkpoint = _read_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=True)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != VERSION:
        raise ValueError("Unsupported checkpoint version; use its original training code")
    return checkpoint


def resume_training(
    source: Path,
    destination: Path,
    model: nn.Module,
    optimizer: Optimizer,
    training: dict[str, Any],
    run_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Restore last.pt and carry forward best.pt plus complete epoch history.

    Paths are checkpoint directories. Only the epoch cap may change. Validate
    both checkpoints before mutating live model/optimizer/RNG state.
    """
    if training["device"] != "cpu":
        raise ValueError("Exact continuation currently supports CPU RNG only")
    last = _read_checkpoint(source / "last.pt")
    best = _read_checkpoint(source / "best.pt")
    current = {key: value for key, value in training.items() if key != "epochs"}
    for name, checkpoint in (("Last", last), ("Best", best)):
        previous = {k: v for k, v in checkpoint["training"].items() if k != "epochs"}
        if previous != current or checkpoint["run_identity"] != run_identity:
            raise ValueError(f"{name} checkpoint training/resume contract mismatch")
    history = json.loads((source.parent / "history.json").read_text())
    if [row["epoch"] for row in history] != list(range(last["epoch"] + 1)):
        raise ValueError("History must contain every epoch through the last checkpoint")
    if training["epochs"] <= len(history):
        raise ValueError("Continuation requires a larger epoch cap")
    if not history or last["val"] != history[-1]["val"]:
        raise ValueError("Last checkpoint disagrees with validation history")
    scores = [row["val"][training["selection_metric"]] for row in history]
    if any(score is None or not math.isfinite(score) for score in scores):
        raise ValueError("History contains an undefined selection metric")
    selected = min(scores) if training["selection_mode"] == "min" else max(scores)
    if best["epoch"] != scores.index(selected) or best["val"] != history[best["epoch"]]["val"]:
        raise ValueError("Best checkpoint disagrees with selection history")
    model.load_state_dict(last["model"], strict=True)
    optimizer.load_state_dict(last["optimizer"])
    torch.set_rng_state(last["torch_rng"])
    shutil.copy2(source / "best.pt", destination / "best.pt")
    return history
