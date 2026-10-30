"""Recheck completed regression runs; never train or select a new checkpoint.

Requires local current and historical run artifacts, not only the public report.
Prepared-data and controlled weight-replay diagnostics are separate recorded
evidence; this command does not repeat those experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.kuairand import CHANNELS as SOURCE_KINDS
from experiments.kuairand import summarize
from feed_ranker.data.dataset import EventDataset, make_loader
from feed_ranker.data.parquet import ParquetReader
from feed_ranker.evaluation.funnel import evaluate_pipeline, open_benchmark
from feed_ranker.evaluation.predictions import (
    PredictionEvaluator,
    Predictions,
    evaluate_predictions,
)
from feed_ranker.persistence import collect_run_provenance, write_json
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.rankers import load_ranker
from feed_ranker.recommendation.retrievers import load_retriever

SEEDS = (43, 44, 45, 46)
RECIPES = (
    "reference",
    "add_author",
    "basic_duration",
    "anonymous_user",
    "combined",
    "author_duration",
)
CHANNELS = ("popularity", "cooccurrence", "mf", "rr_all", "rrf_all")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(current: Path, historical: Path, ablation: Path) -> dict[str, Any]:
    """Validate all 24 fits, prediction metrics and 120 candidate exports."""
    rows, old_rows, curves = [], [], []
    for seed in SEEDS:
        runs = read_json(current / f"run{seed}/summary.json")["runs"]
        require(
            len(runs) == len(RECIPES) and {r["recipe"] for r in runs} == set(RECIPES),
            f"seed {seed}: require exactly the six declared recipes",
        )
        require(all(r["seed"] == seed for r in runs), f"seed {seed}: mismatched run seed")
        require(
            len({r["candidate_cache_sha256"] for r in runs}) == 1,
            f"seed {seed}: recipes must share frozen candidates",
        )
        for row in runs:
            recipe = row["recipe"]
            label = f"seed {seed}/{recipe}"
            path = current / f"run{seed}/seed{seed}/{recipe}"
            old_root = ablation if recipe == "author_duration" else historical
            old_path = old_root / f"run{seed}/seed{seed}/{recipe}"
            history = read_json(path / "history.json")
            require([h["epoch"] for h in history] == list(range(30)), f"{label}: incomplete fit")
            require(all(h["train_rows"] == 4962447 for h in history), f"{label}: changed rows")
            require(
                all(
                    np.isfinite(h[k]) for h in history for k in ("train_loss", "gradient_norm_mean")
                ),
                f"{label}: nonfinite training diagnostics",
            )
            auc = [h["val"]["long_view/roc_auc"] for h in history]
            require(all(np.isfinite(v) for v in auc), f"{label}: nonfinite selection metric")
            require(row["best_epoch"] == auc.index(max(auc)), f"{label}: changed selection rule")
            require(
                row["validation"] == history[row["best_epoch"]]["val"],
                f"{label}: selected-checkpoint metrics disagree",
            )
            pred = pd.read_parquet(path / "predictions.parquet")
            require(
                len(pred) == 839905 and pred.event_id.is_unique, f"{label}: invalid predictions"
            )
            records = Predictions(
                pred.label.to_numpy(),
                pred.logit.to_numpy(),
                pred.user_id.to_numpy(),
                pred.warm_item.to_numpy(),
                np.ones(len(pred), dtype=bool),
            )
            require(
                evaluate_predictions({"long_view": records}) == row["validation"],
                f"{label}: saved predictions disagree with reported metrics",
            )
            old = read_json(old_path / "review.json")
            validation = {}
            for cohort, metrics in old["cohorts"].items():
                prefix = "long_view/" + ("" if cohort == "all" else cohort + "/")
                validation.update({prefix + k: v for k, v in metrics.items()})
            old_rows.append(
                {
                    "seed": seed,
                    "recipe": recipe,
                    "validation": validation,
                    "pipeline": old["funnel"],
                    "best_epoch": old["ranker"]["best_epoch"],
                }
            )
            for channel in CHANNELS:
                with (
                    np.load(old_path / f"{channel}_candidates.npz") as before,
                    np.load(path / channel / "recommendation_candidates.npz") as after,
                ):
                    for key in ("users", "items"):
                        require(
                            np.array_equal(before[key], after[key]),
                            f"{label}/{channel}: candidate {key} changed",
                        )
                for key in (
                    "recall",
                    "oracle_ndcg",
                    "all_target_recall",
                    "returned",
                    "queries",
                    "pool_shortfall_queries",
                    "method_shortfall_queries",
                ):
                    require(
                        row["pipeline"][channel][key] == old["funnel"][channel][key],
                        f"{label}/{channel}: retrieval metric {key} changed",
                    )
            rows.append(row)
            curves.append(
                {
                    "seed": seed,
                    "recipe": recipe,
                    "epochs": [
                        {
                            "epoch": h["epoch"],
                            "auc": h["val"]["long_view/roc_auc"],
                            "train_loss": h["train_loss"],
                            "gradient_norm_mean": h["gradient_norm_mean"],
                        }
                        for h in history
                    ],
                }
            )
    config = read_json(current / "study_config.json")
    return {
        "runs": rows,
        "historical_runs": old_rows,
        "curves": curves,
        "summary": summarize(rows, config),
        "historical_summary": summarize(old_rows, config),
        "verification": {
            "complete_fits": len(rows),
            "epochs_per_fit": 30,
            "prediction_exports": len(rows),
            "candidate_exports_exact": len(rows) * len(CHANNELS),
            "all_retrieval_metrics_exact": True,
        },
    }


def replay_scores(current: Path, prepared: Path, output: Path) -> dict[str, Any]:
    """Re-score the saved six-arm regression through current loading and inference.

    Uses fresh materialized validation rows and rebuilds all recall caches from
    fitted scorers. Every prediction, trace, per-user metric and paired summary
    must match the reference. No fitting, selection or historical mutation.
    """
    data = ParquetReader(prepared)
    recorded = read_json(current / f"run{SEEDS[0]}/data_manifest.json")
    for key in (
        "source_sha256",
        "snapshot_sha256",
        "selected_users",
        "train_end",
        "val_end",
        "targets",
    ):
        require(data.manifest[key] == recorded[key], f"Replay data contract changed: {key}")
    benchmark = open_benchmark(prepared)
    validation = data.read_split("validation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", collect_run_provenance())
    write_json(output / "data_manifest.json", data.manifest)
    rows = []
    for seed in SEEDS:
        source = current / f"run{seed}/seed{seed}"
        seed_dir = output / f"seed{seed}"
        seed_dir.mkdir()
        recall_dir = source / "recall"
        if (recall_dir / "models").is_dir():
            recall_dir = recall_dir / "models"
        frozen = []
        for name, kind in SOURCE_KINDS.items():
            cached = seed_dir / f"{name}.parquet"
            fitted = load_retriever(recall_dir / kind)
            frozen.append(benchmark.cache(fitted, cached))
            pd.testing.assert_frame_equal(
                pd.read_parquet(cached),
                pd.read_parquet(source / "candidates" / f"{name}.parquet"),
                check_exact=True,
                obj=f"seed={seed} source={name}",
            )
        print(f"seed={seed}: all three source rankings/scores/support match", flush=True)
        policies = {r.name: r for r in frozen}
        policies.update(
            rr_all=MergedRetriever(frozen, "round_robin", source_depth=50),
            rrf_all=MergedRetriever(frozen, "rrf", source_depth=50),
        )
        previous = read_json(current / f"run{seed}/summary.json")["runs"]
        for row in previous:
            recipe = row["recipe"]
            path = source / recipe
            ranker = load_ranker(path / "model")
            instances = EventDataset(validation, ranker.model.config.feature_names, ("long_view",))
            records = PredictionEvaluator({"long_view": "long_view"}).collect(
                ranker.model, make_loader(instances, 8192)
            )
            expected = pd.read_parquet(path / "predictions.parquet")
            record = records["long_view"]
            for name, actual, before in (
                ("event_id", validation.event_id, expected.event_id),
                ("user_id", record.users, expected.user_id),
                ("label", record.labels, expected.label),
                ("warm_item", record.warm, expected.warm_item),
                ("logit", record.logits, expected.logit),
            ):
                np.testing.assert_array_equal(actual, before, err_msg=f"{seed}/{recipe}/{name}")
            metrics = evaluate_predictions(records)
            require(metrics == row["validation"], f"{seed}/{recipe}: event metrics changed")
            pipeline = {}
            for name, retriever in policies.items():
                destination = seed_dir / recipe / name
                pipeline[name] = evaluate_pipeline(
                    benchmark, ranker, retriever, destination, budget=50
                )["metrics"]
                require(
                    pipeline[name] == row["pipeline"][name],
                    f"{seed}/{recipe}/{name}: funnel metrics changed",
                )
                for suffix in ("requests", "trace"):
                    pd.testing.assert_frame_equal(
                        pd.read_parquet(destination / f"recommendation_{suffix}.parquet"),
                        pd.read_parquet(path / name / f"recommendation_{suffix}.parquet"),
                        check_exact=True,
                        obj=f"{seed}/{recipe}/{name}/{suffix}",
                    )
                pd.testing.assert_frame_equal(
                    pd.read_parquet(destination / "per_user.parquet"),
                    pd.read_parquet(path / name / "per_user.parquet"),
                    check_exact=True,
                    obj=f"{seed}/{recipe}/{name}/per_user",
                )
                with (
                    np.load(destination / "recommendation_candidates.npz") as actual,
                    np.load(path / name / "recommendation_candidates.npz") as before,
                ):
                    for key in ("users", "items"):
                        np.testing.assert_array_equal(
                            actual[key], before[key], err_msg=f"{seed}/{recipe}/{name}/{key}"
                        )
            rows.append({**row, "validation": metrics, "pipeline": pipeline})
            write_json(output / "runs.json", rows)
            print(
                f"seed={seed} recipe={recipe}: {len(expected)} logits and five pipelines match exactly",
                flush=True,
            )
        del frozen, policies, ranker, instances, records
    config = read_json(current / "study_config.json")
    summary = summarize(rows, config)
    expected_rows = [
        row for seed in SEEDS for row in read_json(current / f"run{seed}/summary.json")["runs"]
    ]
    require(summary == summarize(expected_rows, config), "Paired summaries changed")
    result = {
        "data_identity": data.identity,
        "runs": rows,
        "summary": summary,
        "verification": {
            "model_prediction_exports_exact": len(rows),
            "validation_rows_per_model": len(validation),
            "source_caches_exact": len(SEEDS) * len(SOURCE_KINDS),
            "candidate_exports_exact": len(rows) * len(CHANNELS),
            "ranking_traces_and_per_user_metrics_exact": True,
            "paired_summaries_exact": True,
            "new_training": False,
        },
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current", type=Path, required=True, help="Contains run43..46 and study_config.json"
    )
    parser.add_argument(
        "--historical", type=Path, required=True, help="Original five-arm run43..46 root"
    )
    parser.add_argument(
        "--ablation", type=Path, required=True, help="Original author_duration run43..46 root"
    )
    parser.add_argument("--out", type=Path, required=True, help="New output JSON; must not exist")
    parser.add_argument(
        "--prepared", type=Path, help="Fresh prepared-v4 data for current-code scoring replay"
    )
    parser.add_argument(
        "--replay-out", type=Path, help="New directory for current-code replay evidence"
    )
    args = parser.parse_args()
    if (args.prepared is None) != (args.replay_out is None):
        parser.error("--prepared and --replay-out must be supplied together")
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.current, args.historical, args.ablation)
    if args.prepared is not None:
        replay = replay_scores(args.current, args.prepared, args.replay_out)
        result["current_code_replay"] = replay["verification"]
    with args.out.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["verification"]))


if __name__ == "__main__":
    main()
