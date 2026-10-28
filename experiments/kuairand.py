"""Matched KuaiRand comparisons composed from the shared package interfaces.

Historical results remain evidence from their pinned source. This runner does not
promote a model or score the final window; a real run needs a declared budget.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.artifacts import load_artifact
from feed_ranker.data.instances import EventDataset, make_loader
from feed_ranker.data.prepared import PreparedDataset
from feed_ranker.evaluation.funnel import WindowBenchmark, evaluate_pipeline, open_benchmark
from feed_ranker.evaluation.metrics import binary_metrics
from feed_ranker.evaluation.predictions import PredictionEvaluator, evaluate_predictions
from feed_ranker.persistence import collect_run_provenance, sha256, snapshot_sources, write_json
from feed_ranker.recommendation.merged import MergedRetriever
from feed_ranker.recommendation.rankers import load_ranker
from feed_ranker.recommendation.replay import ReplayRetriever, ScoreTable
from feed_ranker.recommendation.retrievers import load_retriever
from feed_ranker.training.jobs import fit_recall_job, job_config, train_job

CHANNELS = {"popularity": "popularity", "cooccurrence": "cosine", "mf": "mf"}


def frozen_recall(
    data: PreparedDataset,
    benchmark: WindowBenchmark,
    directory: Path,
    seed: int,
    epochs: int,
    source: Path | None,
) -> tuple[list[ReplayRetriever], str]:
    """Fit once or copy verified current artifacts/caches; never retrain on reuse."""
    contract = {
        "data_manifest_sha256": data.identity,
        "seed": seed,
        "mf_epochs": epochs,
        "target": "long_view",
    }
    if source is None:
        fit_recall_job(data.path, directory / "recall", epochs=epochs, seed=seed)
        cache_dir = directory / "candidates"
        cache_dir.mkdir()
        fitted = [load_retriever(directory / "recall" / kind) for kind in CHANNELS.values()]
        for retriever in fitted:
            benchmark.cache(retriever, cache_dir / f"{retriever.name}.parquet")
        manifest = {
            **contract,
            "artifacts": {r.name: r.artifact_id for r in fitted},
            "files": {
                str(p.relative_to(directory)): sha256(p)
                for subdir in ("recall", "candidates")
                for p in sorted((directory / subdir).rglob("*"))
                if p.is_file()
            },
        }
        write_json(directory / "recall_manifest.json", manifest)
    else:
        source = source / f"seed{seed}"
        manifest = json.loads((source / "recall_manifest.json").read_text())
        if any(manifest.get(k) != v for k, v in contract.items()):
            raise ValueError("Recall data, seed or MF budget differs")
        actual = {
            str(p.relative_to(source)): sha256(p)
            for subdir in ("recall", "candidates")
            for p in sorted((source / subdir).rglob("*"))
            if p.is_file()
        }
        if actual != manifest["files"]:
            raise ValueError("Frozen recall files changed or are incomplete")
        for name in CHANNELS:
            artifact = load_artifact(source / "recall" / CHANNELS[name])
            if artifact.identity != manifest["artifacts"][name]:
                raise ValueError("Recall artifact identity mismatch")
            provenance = artifact.provenance
            if any(provenance.get(k) != v for k, v in contract.items()):
                raise ValueError("Recall artifact provenance mismatch")
        directory.mkdir(parents=True, exist_ok=False)
        for subdir in ("recall", "candidates"):
            shutil.copytree(source / subdir, directory / subdir)
        shutil.copy2(source / "recall_manifest.json", directory / "recall_manifest.json")
        write_json(directory / "recall_reuse.json", {"source": str(source.resolve())})
    frozen = [
        ReplayRetriever(
            name,
            ScoreTable(
                pd.read_parquet(directory / "candidates" / f"{name}.parquet"),
                manifest["artifacts"][name],
                allow_unsupported=True,
            ),
        )
        for name in CHANNELS
    ]
    return frozen, sha256(directory / "recall_manifest.json")


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Seed-aligned estimates and paired differences, keeping replication separate."""
    names = [r["name"] for r in config["recipes"]]
    comparisons = [(name, names[0]) for name in names]
    comparisons.extend(tuple(pair) for pair in config.get("comparisons", []))
    groups = {"primary": config.get("primary_seeds", config["seeds"])}
    replication = [s for s in config["seeds"] if s not in groups["primary"]]
    if replication:
        groups["replication"] = replication
    indexed = {(r["recipe"], r["seed"]): r for r in rows}
    result = []
    for group, seeds in groups.items():
        for name, baseline in dict.fromkeys(comparisons):
            entries = [indexed[name, seed] for seed in seeds]
            references = [indexed[baseline, seed] for seed in seeds]
            values = {
                metric: (
                    [r["validation"][metric] for r in entries],
                    [r["validation"][metric] for r in references],
                )
                for metric in entries[0]["validation"]
                if metric.endswith(("roc_auc", "macro_user_auc", "logloss", "brier"))
            }
            for channel in (*CHANNELS, "rr_all", "rrf_all"):
                for metric in ("ndcg", "recall", "oracle_ndcg", "all_target_recall"):
                    values[f"{channel}/{metric}"] = (
                        [r["pipeline"][channel][metric] for r in entries],
                        [r["pipeline"][channel][metric] for r in references],
                    )
            for metric, (observed, reference) in values.items():
                paired = [
                    None if a is None or b is None else a - b
                    for a, b in zip(observed, reference, strict=True)
                ]
                entry = {
                    "group": group,
                    "recipe": name,
                    "reference": baseline,
                    "metric": metric,
                    "seeds": seeds,
                    "values": observed,
                    "paired_differences": paired,
                }
                for prefix, items in (("", observed), ("paired_", paired)):
                    valid = [v for v in items if v is not None]
                    entry.update(
                        {
                            prefix + "mean": float(np.mean(valid)) if valid else None,
                            prefix + "sample_sd": float(np.std(valid, ddof=1))
                            if len(valid) > 1
                            else None,
                            prefix + "defined_seeds": len(valid),
                        }
                    )
                result.append(entry)
    return result


def run_study(
    config: dict[str, Any],
    directory: Path,
    *,
    recall_from: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    seeds, recipes = config["seeds"], config["recipes"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("Require distinct valid seeds")
    primary = config.get("primary_seeds", seeds)
    if (
        not primary
        or len(set(primary)) != len(primary)
        or any(type(s) is not int for s in primary)
        or not set(primary) <= set(seeds)
    ):
        raise ValueError("Primary seeds must be a nonempty distinct subset")
    names = [r["name"] for r in recipes]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not n.replace("_", "").isalnum() or n in ("recall", "candidates") for n in names)
    ):
        raise ValueError("Require unique safe recipe names")
    if any(len(pair) != 2 or not set(pair) <= set(names) for pair in config.get("comparisons", [])):
        raise ValueError("Comparisons must name two configured recipes")
    if any(
        type(config.get(k, 50)) is not int or config.get(k, 50) < 1 for k in ("mf_epochs", "budget")
    ):
        raise ValueError("Require positive integer MF epochs and candidate budget")
    if config["tasks"] != [{"output": "long_view", "target": "long_view"}]:
        raise ValueError("This study requires unweighted long_view BCE supervision")
    if resume_from is not None:
        if recall_from is not None and recall_from.resolve() != resume_from.resolve():
            raise ValueError("Continuation must reuse recall from the same study")
        recall_from = resume_from
        previous = json.loads((resume_from / "config.json").read_text())
        before, after = json.loads(json.dumps(previous)), json.loads(json.dumps(config))
        old_epochs, new_epochs = before["training"].pop("epochs"), after["training"].pop("epochs")
        if before != after or new_epochs <= old_epochs:
            raise ValueError("Study continuation may only increase the epoch cap")
    if config["training"].get("device", "cpu") != "cpu":
        raise ValueError("This study requires CPU for exact exported-scorer checks")
    data_path = Path(config["data"])
    data = PreparedDataset(data_path)
    data.validate()
    # Validate every recipe before fitting recall or creating output directories.
    jobs = {}
    for recipe in recipes:
        job = job_config(
            {
                "data": str(data_path),
                "model": recipe["model"],
                "tasks": config["tasks"],
                "training": {**config["training"], "seed": seeds[0]},
                "loading": config.get("loading", {}),
            }
        )
        job.training.optimizer.validate()
        jobs[recipe["name"]] = job
    benchmark = open_benchmark(data_path)
    validation = data.read_split("validation")
    prior = float(np.clip(data.read_split("train", ["long_view"]).long_view.mean(), 1e-7, 1 - 1e-7))
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    write_json(directory / "data_manifest.json", data.manifest)
    write_json(directory / "provenance.json", collect_run_provenance())
    snapshot_sources(directory / "source")
    rows = []
    for seed in seeds:
        seed_dir = directory / f"seed{seed}"
        frozen, cache_id = frozen_recall(
            data, benchmark, seed_dir, seed, config["mf_epochs"], recall_from
        )
        policies = {r.name: r for r in frozen}
        policies.update(
            rr_all=MergedRetriever(frozen, "round_robin", source_depth=50),
            rrf_all=MergedRetriever(frozen, "rrf", source_depth=50),
        )
        for recipe in recipes:
            template = jobs[recipe["name"]]
            job = replace(template, training=replace(template.training, seed=seed))
            run_dir = seed_dir / recipe["name"]
            result = train_job(
                job,
                run_dir,
                resume_from=(
                    None if resume_from is None else resume_from / f"seed{seed}" / recipe["name"]
                ),
            )
            ranker = load_ranker(run_dir / "model")
            instances = EventDataset(
                validation, ranker.fitted.feature_state, job.model.feature_names, ("long_view",)
            )
            records = PredictionEvaluator({"long_view": "long_view"}).collect(
                ranker.fitted.model, make_loader(instances, job.eval_batch_size)
            )
            metrics = evaluate_predictions(records)
            if metrics != result["val"]:
                raise ValueError("Reloaded scorer metrics disagree with the selected checkpoint")
            pd.DataFrame(
                {
                    "event_id": validation.event_id,
                    "user_id": records["long_view"].users,
                    "label": records["long_view"].labels,
                    "logit": records["long_view"].logits,
                    "warm_item": records["long_view"].warm,
                }
            ).to_parquet(run_dir / "predictions.parquet", index=False)
            pipeline = {
                name: evaluate_pipeline(
                    benchmark, ranker, retriever, run_dir / name, budget=config.get("budget", 50)
                )["metrics"]
                for name, retriever in policies.items()
            }
            rows.append(
                {
                    "seed": seed,
                    "recipe": recipe["name"],
                    "validation": metrics,
                    "pipeline": pipeline,
                    "candidate_cache_sha256": cache_id,
                    "best_epoch": result["best_epoch"],
                    "model_artifact": ranker.artifact_id,
                }
            )
            write_json(directory / "runs.json", rows)
    result = {
        "data_manifest_sha256": data.identity,
        "coverage": benchmark.coverage,
        "prior_baseline": binary_metrics(
            validation.long_view.to_numpy(), np.full(len(validation), np.log(prior / (1 - prior)))
        ),
        "runs": rows,
        "summary": summarize(rows, config),
        "interpretation": "Reused validation, not a fresh holdout. Primary/replication seeds are explicit. No automatic promotion or final evaluation.",
    }
    write_json(directory / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recall-from", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run_study(
        json.loads(args.config.read_text()),
        args.out,
        recall_from=args.recall_from,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
