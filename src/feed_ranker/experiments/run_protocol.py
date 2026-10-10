"""Run a frozen big-to-small comparison, resuming at completed-run boundaries.

Incomplete runs are archived and restarted from their seed. Test evaluation begins
only after every training run finishes; test results never change this recipe.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from feed_ranker.artifacts import collect_run_provenance, write_json
from feed_ranker.data.kuairec_preprocessor import KuaiRecPreprocessor
from feed_ranker.experiments.eda_kuairec import audit_matrices
from feed_ranker.experiments.multi_seed import summarize
from feed_ranker.experiments.plot_run import plot_run
from feed_ranker.experiments.wide_deep_kuairec import evaluate_checkpoint, run_experiment
from feed_ranker.training import TrainConfig

MODELS = ["deep_nn", "wide_deep_nn"]
SEEDS = [42, 43, 44, 45, 46]


def run_protocol(raw: Path, output: Path, config: TrainConfig, threads: int = 2) -> None:
    """Persist progress and fail loudly; invoking the same recipe resumes the job."""
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=True)
    recipe = {
        "raw": str(raw.resolve()),
        "training": asdict(config),
        "threads": threads,
        "models": MODELS,
        "seeds": SEEDS,
        "hidden": [128, 32],
        "dropout": 0.0,
        "features": {
            "categorical": ["user_id", "video_id"],
            "dense": ["video_duration"],
            "crosses": {},
        },
        "split_seed": 2026,
        "val_ratio": 0.1,
        "source_sha256": collect_run_provenance()["source_sha256"],
    }
    # JSON round-trip normalizes tuples before comparing a saved recipe.
    recipe = json.loads(json.dumps(recipe))
    recipe_path = output / "recipe.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise ValueError(
                "Recipe/source changed: restore the original or use a new output directory"
            )
    else:
        write_json(recipe_path, recipe)
    try:
        _prepare(raw, output, recipe)
        runs = _train(output, config, recipe)
        tests = _evaluate(output, runs)
        _report(output, runs, tests, config.metrics)
        _status(output, "complete")
    except Exception:
        _status(output, "failed", error=traceback.format_exc())
        raise


def _prepare(raw: Path, output: Path, recipe: dict[str, Any]) -> None:
    audit = output / "eda.json"
    if not audit.exists():
        _status(output, "eda")
        audit_matrices(raw, audit)
    prepared = output / "prepared"
    if not prepared.exists():
        _status(output, "preprocessing")
        KuaiRecPreprocessor(
            raw, prepared, recipe["features"], recipe["split_seed"], recipe["val_ratio"]
        ).run()


def _train(output: Path, config: TrainConfig, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        for model in MODELS:
            directory = output / f"{model}-{seed}"
            result = directory / "result.json"
            if result.exists() and "model" in json.loads(result.read_text()):
                run = json.loads(result.read_text())
            else:
                if directory.exists():
                    directory.rename(output / f"{directory.name}.interrupted-{time.time_ns()}")
                _status(output, "training", model=model, seed=seed)
                training = asdict(config)
                training.pop("optimizer")
                training["seed"] = seed
                run = run_experiment(
                    output / "prepared",
                    directory,
                    model,
                    TrainConfig(**training, optimizer=config.optimizer),
                    recipe["hidden"],
                    recipe["dropout"],
                )
            runs.append(run)
            plot_run(directory, directory / "curves.png")
    write_json(output / "validation_summary.json", summarize(runs, MODELS, SEEDS, config.metrics))
    return runs


def _evaluate(output: Path, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests = []
    for run in runs:
        directory = output / f"{run['model']}-{run['seed']}"
        path = directory / "test.json"
        _status(output, "test_evaluation", model=run["model"], seed=run["seed"])
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = evaluate_checkpoint(output / "prepared", directory / "best.pt", path)
        tests.append(result)
    return tests


def _report(
    output: Path, runs: list[dict[str, Any]], tests: list[dict[str, Any]], metrics: tuple[str, ...]
) -> None:
    validation = summarize(runs, MODELS, SEEDS, metrics)
    test = summarize([{**run, "val": run["test"]} for run in tests], MODELS, SEEDS, metrics)
    write_json(output / "summary.json", {"validation": validation, "test": test, "per_seed": tests})
    lines = [
        "# Fixed Deep versus Wide & Deep comparison",
        "",
        "Big-only pair-grouped 90/10 train/validation; small matrix final evaluation.",
        "Seeds 42–46; one fixed recipe; checkpoint selected by validation only.",
        "No feature crosses: this tests the additive dense wide branch.",
        "Prior small-matrix experiments and descriptive label EDA mean this is not a pristine test.",
        "",
    ]
    for split, report in (("Validation", validation), ("Test", test)):
        lines.extend([f"## {split}", ""])
        for model, scores in report["models"].items():
            lines.append(
                f"- {model}: "
                + "; ".join(
                    f"{name} = {stats['mean']} (sample std {stats['std']})"
                    for name, stats in scores.items()
                )
            )
        lines.extend(["", "W&D minus Deep, paired by training seed:"])
        for name, stats in report["paired_deltas"]["wide_deep_nn"].items():
            lines.append(
                f"- {name}: mean {stats['mean']}; sample std {stats['std']}; deltas {stats['values']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Review before drawing conclusions",
            "",
            "Inspect each run's curves.png, history.json, selected epoch and neighboring metrics.",
            "Training-seed variation is not an evaluation confidence interval. No causal or temporal claim follows.",
            "Inspect EDA duration cohorts and test user coverage; model-error cohort analysis remains a follow-up.",
            "These are measurements; the keep/revert decision and generalized insight remain to be reviewed.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _status(output: Path, stage: str, **details: Any) -> None:
    status = {"stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details}
    write_json(output / "status.json", status)
    print(json.dumps(status), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/kuairec"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another worker is already running in this output directory"
            ) from error
        run_protocol(args.raw_dir, args.output, TrainConfig(), args.threads)


if __name__ == "__main__":
    main()
