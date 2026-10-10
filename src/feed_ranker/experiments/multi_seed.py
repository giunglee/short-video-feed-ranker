"""Validation comparisons use matched training seeds and one prepared data split.

The first model is the baseline. Deltas are model minus baseline for the same seed;
undefined metrics and failed runs must not silently disappear from the comparison.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from feed_ranker.artifacts import write_json
from feed_ranker.experiments.wide_deep_kuairec import (
    MODELS,
    add_train_args,
    config_from_args,
    run_experiment,
)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run every requested model/seed combination; propagate any failed run."""
    _validate_comparison(args.models, args.seeds)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "comparison.json",
        {
            **vars(args),
            "data": str(args.data),
            "output": str(args.output),
        },
    )
    runs = []
    for model in args.models:
        for seed in args.seeds:
            run = run_experiment(
                directory=args.data,
                output=args.output / f"{model}-{seed}",
                model_name=model,
                config=config_from_args(args, seed),
                hidden=args.hidden,
                dropout=args.dropout,
                use_ids=not args.no_ids,
            )
            runs.append(run)

    result = summarize(runs, args.models, args.seeds, tuple(args.metrics))
    result["exploratory"] = args.steps is not None
    result["per_seed"] = [
        {key: run[key] for key in ("model", "seed", "best_epoch", "val")} for run in runs
    ]
    write_json(args.output / "summary.json", result)
    return result


def summarize(
    runs: list[dict[str, Any]],
    models: list[str],
    seeds: list[int],
    metrics: tuple[str, ...] = ("mae", "xauc", "user_xauc"),
) -> dict[str, Any]:
    _validate_comparison(models, seeds)
    lookup = {(run["model"], run["seed"]): run for run in runs}
    expected = {(model, seed) for model in models for seed in seeds}
    if len(lookup) != len(runs) or set(lookup) != expected:
        raise ValueError(
            "Need exactly one run per model and seed; never drop a failed run silently"
        )

    baseline = models[0]
    summaries = {}
    paired_deltas = {}
    for model in models:
        summaries[model] = {}
        if model != baseline:
            paired_deltas[model] = {}
        for metric in metrics:
            values = [lookup[model, seed]["val"][metric] for seed in seeds]
            summaries[model][metric] = describe(values)
            if model != baseline:
                reference = [lookup[baseline, seed]["val"][metric] for seed in seeds]
                deltas = _paired_differences(values, reference)
                paired_deltas[model][metric] = {"values": deltas, **describe(deltas)}
    return {
        "seeds": seeds,
        "baseline": baseline,
        "models": summaries,
        "paired_deltas": paired_deltas,
    }


def describe(values: list[float | None]) -> dict[str, float | int | None]:
    """Sample standard deviation requires two runs; any undefined input stays undefined."""
    if not values or any(value is None for value in values):
        return {"mean": None, "std": None, "n": len(values)}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def _paired_differences(
    values: list[float | None], baseline: list[float | None]
) -> list[float | None]:
    differences = []
    for value, reference in zip(values, baseline):
        difference = None if value is None or reference is None else value - reference
        differences.append(difference)
    return differences


def _validate_comparison(models: list[str], seeds: list[int]) -> None:
    if not models or not seeds:
        raise ValueError("A comparison requires at least one model and one seed")
    if len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError("Models and seeds must be unique")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["deep_nn", "wide_deep_nn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    try:
        _validate_comparison(args.models, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    result = run_comparison(_parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
