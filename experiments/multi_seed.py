"""
Multi-seed variance check for one Wide & Deep config.

A single seed's val MAE/XAUC gap between two configs can just be that
seed's noise (weight init, dropout masks, train-batch shuffling all derive
from torch.manual_seed/np.random.seed -- see wide_deep_kuairec.parse_args).
This runs the *same* config across several seeds and reports mean +/- std
per metric, so a "winner" from the ablation sweep can be checked before
treating it as the answer rather than a plausible one.

Run from repo root, e.g.:
  python -m experiments.multi_seed --model wide_deep_nn --nn-dims 256,64,8,1 \
      --dropout 0 --emb-mode cats --seeds 42,43,44,45,46 \
      --output artifacts/wide_deep_m1/wd_wide_path/wd_cats_drop0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.wide_deep_kuairec import run_experiment

# What to aggregate across seeds. best_val_mae/val_xauc are what selection
# should use; test_* is logged for monitoring only (see run_experiment).
METRICS = ["best_val_mae", "val_xauc", "test_mae", "test_xauc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed variance check for one Wide & Deep config")
    p.add_argument(
        "--model",
        required=True,
        choices=["linear_reg", "wide_nn", "deep_nn", "wide_deep_nn"],
    )
    p.add_argument("--nn-dims", type=str, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument(
        "--emb-mode",
        choices=["all", "ids", "cats"],
        default="all",
        help="deep_nn / wide_deep_nn embedding tables (see wide_deep_kuairec).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seeds. 5 is a reasonable minimum to get a "
        "usable std estimate without the sweep taking too long; go higher "
        "if a decision hinges on a close call.",
    )
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hidden_dims = [int(d) for d in args.nn_dims.split(",")] if args.nn_dims else None
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = []
    for seed in seeds:
        # Same global-RNG contract as wide_deep_kuairec.__main__: seed
        # everything once per run, before model construction and dataset
        # shuffling both draw from it.
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = run_experiment(
            args.model,
            hidden_dims=hidden_dims,
            dropout_ratio=args.dropout,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            steps=args.steps,
            compute_xauc=True,
            seed=seed,
            emb_mode=args.emb_mode,
        )
        print(
            f"seed={seed:>4}  best_val_mae={result['best_val_mae']:.4f}  "
            f"val_xauc={result['val_xauc']:.4f}  test_mae={result['test_mae']:.4f}  "
            f"test_xauc={result['test_xauc']:.4f}"
        )
        runs.append(result)

    summary: dict = {
        "model": args.model,
        "nn_dims": hidden_dims,
        "dropout_ratio": args.dropout,
        "weight_decay": args.weight_decay,
        "emb_mode": args.emb_mode if args.model in ("deep_nn", "wide_deep_nn") else None,
        "seeds": seeds,
        "n_seeds": len(seeds),
    }
    for metric in METRICS:
        vals = np.array([r[metric] for r in runs], dtype=float)
        summary[f"{metric}_mean"] = float(vals.mean())
        # Sample std (ddof=1) -- undefined for n=1, reported as 0.0 rather
        # than NaN so downstream JSON consumers don't need a special case.
        summary[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary["runs"] = runs

    print("\n--- summary (mean +/- std across seeds) ---")
    for metric in METRICS:
        print(f"{metric:>14}: {summary[f'{metric}_mean']:.4f} +/- {summary[f'{metric}_std']:.4f}")

    print(f"\nRESULT_JSON: {json.dumps(summary, sort_keys=True)}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
