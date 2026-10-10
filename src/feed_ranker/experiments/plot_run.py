"""Saved training loss and validation metrics; dashed lines mark the selected epoch.

Loss and evaluation metrics have separate axes because their units can differ.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def plot_run(directory: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((directory / "history.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if not history:
        raise ValueError("Cannot plot an empty training history")
    config = result["training"]
    metrics = config.get("metrics", ["mae", "xauc", "user_xauc"])
    loss_key = "train_loss" if "train_loss" in history[0] else "train_online_mae"
    curves = {
        f"Training loss ({config.get('loss', 'mae')}, online)": [row[loss_key] for row in history],
        **{f"Validation {name}": [row["val"][name] for row in history] for name in metrics},
        "Mean gradient norm": [row["gradient_norm_mean"] for row in history],
    }
    epochs = [row["epoch"] for row in history]
    rows = math.ceil(len(curves) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(10, 3 * rows), squeeze=False)
    try:
        for axis, (title, values) in zip(axes.flat, curves.items()):
            axis.plot(epochs, values)
            axis.axvline(result["best_epoch"], linestyle="--", color="gray")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
        for axis in list(axes.flat)[len(curves) :]:
            axis.set_visible(False)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_run(args.run, args.output)


if __name__ == "__main__":
    main()
