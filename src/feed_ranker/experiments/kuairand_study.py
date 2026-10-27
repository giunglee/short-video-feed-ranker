"""Strict temporal KuaiRand baseline study on the shared recommendation pipeline.

Prepare once, then screen seven independent width/feature recipes against frozen
popularity/cooccurrence/MF candidates. Validation is exploratory and exposure-conditional.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.kuairand import (
    TRAIN_END,
    VAL_END,
    EventDataset,
    histories,
    load_prepared,
    prepare,
)
from feed_ranker.data.metadata import MetadataEncoding
from feed_ranker.evaluation.binary import (
    binary_metrics,
    predict_binary,
    user_auc_metrics,
)
from feed_ranker.evaluation.kuairand import WindowBenchmark
from feed_ranker.features.kuairand import FEATURE_SETS, KuaiRandFeatures
from feed_ranker.inference.kuairand import SparseCooccurrenceRetriever, load_ranker
from feed_ranker.inference.merged import MergedRetriever
from feed_ranker.inference.rankers import ModelRanker
from feed_ranker.inference.replay import ReplayRetriever, ScoreTable
from feed_ranker.inference.retrievers import MFRetriever, PopularityRetriever
from feed_ranker.models.ranking_factory import ModelSpec
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.traces import RecommendationTrace
from feed_ranker.training.kuairand import fit_legacy_deep, fit_mf
from feed_ranker.training.optimizer import OptimizerConfig
from feed_ranker.training.trainer import TrainConfig


def read_recipes(path: Path, selected: list[str] | None) -> list[dict[str, Any]]:
    config = json.loads(path.read_text())
    if config["version"] not in ("kuairand-strict-screen-v1", "kuairand-side-study-v1"):
        raise ValueError("Unsupported study recipe version")
    recipes = config["recipes"]
    names = [r["name"] for r in recipes]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Recipe names must be unique and nonempty")
    for r in recipes:
        if not r["name"].replace("_", "").isalnum() or r["features"] not in FEATURE_SETS:
            raise ValueError("Invalid recipe name or feature contract")
        ModelSpec("deep", tuple(r["hidden"]))
    if selected is not None:
        if not set(selected) <= set(names) or len(set(selected)) != len(selected):
            raise ValueError("Unknown/duplicate selected recipes")
        recipes = [r for r in recipes if r["name"] in selected]
    return recipes


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Defaults preserve the historical logloss screen for existing callers.
    args.selection_metric = getattr(args, "selection_metric", "logloss")
    args.resume_from = getattr(args, "resume_from", None)
    args.recall_from = getattr(args, "recall_from", None)
    args.side_features = getattr(args, "side_features", None)
    args.include_merges = getattr(args, "include_merges", False)
    if args.selection_metric not in ("logloss", "roc_auc"):
        raise ValueError("Selection metric must be logloss or roc_auc")
    if min(args.epochs, args.mf_epochs, args.batch_size, args.threads, args.budget) < 1:
        raise ValueError("Budgets and threads must be positive")
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in args.seeds)
    ):
        raise ValueError("Require distinct training seeds")
    recipes = read_recipes(args.config, args.recipes)
    if args.resume_from is not None:
        previous = {r["name"]: r for r in read_recipes(args.resume_from / "recipes.json", None)}
        if any(previous.get(r["name"]) != r for r in recipes):
            raise ValueError("Continuation requires identical feature/model recipes")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    write_json(args.out / "provenance.json", collect_run_provenance())
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        args.out / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(args.config, args.out / "recipes.json")
    train, val, encoding, manifest = load_prepared(args.data)
    label = manifest["label"]
    manifest_hash = sha256(args.data / "manifest.json")
    metadata, metadata_hash = None, None
    if args.side_features is not None:
        fitted = torch.load(args.side_features, weights_only=True)
        if fitted["data_manifest_sha256"] != manifest_hash:
            raise ValueError("Metadata fitted to different training data")
        metadata = MetadataEncoding.from_state(fitted["encoding"])
        metadata_hash = sha256(args.side_features)
        write_json(args.out / "feature_assumptions.json", fitted["manifest"])
    write_json(args.out / "data_manifest.json", manifest)
    benchmark = WindowBenchmark(
        val,
        encoding.items,
        label,
        manifest["train_end_exclusive_ms"],
        manifest["val_end_exclusive_ms"],
    )
    if args.recall_from is None:
        positive, negative = histories(train, encoding, label)
        cooc = SparseCosineCooccurrence(positive, len(encoding.items))
    prior = float(np.clip(train[label].mean(), 1e-7, 1 - 1e-7))
    prior_metrics = binary_metrics(
        val[label].to_numpy(), np.full(len(val), np.log(prior / (1 - prior)))
    )
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_dir = args.out / f"seed{seed}"
        seed_dir.mkdir()
        if args.recall_from is not None:
            cached = reuse_recall(args.recall_from, seed_dir, seed, manifest_hash, args.mf_epochs)
        else:
            cached = fit_recall(
                seed_dir,
                seed,
                args.mf_epochs,
                positive,
                negative,
                encoding,
                cooc,
                manifest_hash,
                benchmark,
            )
        if args.include_merges:
            sources = list(cached)
            cached.extend(MergedRetriever(sources, policy, 50) for policy in ("rr_all", "rrf_all"))
        for recipe in recipes:
            started = time.monotonic()
            directory = seed_dir / recipe["name"]
            features = KuaiRandFeatures(encoding, recipe["features"], metadata)
            config = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                train_eval_rows=0,
                loss="bce_logits",
                metrics=("logloss", "roc_auc"),
                selection_metric=args.selection_metric,
                selection_mode="max" if args.selection_metric == "roc_auc" else "min",
                optimizer=OptimizerConfig(lr=0.001),
            )
            model, result = fit_legacy_deep(
                train,
                val,
                encoding,
                features,
                label,
                recipe["hidden"],
                config,
                directory,
                manifest_hash,
                metadata_hash,
                resume_from=(
                    args.resume_from / f"seed{seed}" / recipe["name"]
                    if args.resume_from is not None
                    else None
                ),
            )
            val_data = EventDataset(val, encoding, label, recipe["features"], metadata)
            result["exploratory"] = True
            write_json(directory / "result.json", result)
            labels, logits = predict_binary(model, val_data.batches(8192))
            pd.DataFrame({"event_id": val.event_id.to_numpy(), "logit": logits}).to_parquet(
                directory / "predictions.parquet", index=False
            )
            warm = pd.Index(encoding.items).get_indexer(val.video_id) >= 0
            cohorts = {
                name: {
                    **binary_metrics(labels[mask], logits[mask]),
                    **user_auc_metrics(val.user_id.to_numpy()[mask], labels[mask], logits[mask]),
                }
                for name, mask in [
                    ("all", np.ones(len(val), dtype=bool)),
                    ("warm_item", warm),
                    ("cold_item", ~warm),
                ]
                if mask.any()
            }
            artifact = directory / "inference.pt"
            ranker = load_ranker(artifact)
            expected = (
                ModelRanker(model, features, "training-model")
                .score(benchmark.queries[0][0], benchmark.queries[0][1].items)
                .scores
            )
            actual = ranker.score(*[benchmark.queries[0][0], benchmark.queries[0][1].items]).scores
            np.testing.assert_array_equal(expected, actual)
            funnel = {}
            for retriever in cached:
                pipeline = RecommendationPipeline(retriever, ranker, args.budget, 10)
                recommendations = [pipeline.recommend(q, scope) for q, scope in benchmark.queries]
                metrics, per_user = benchmark.evaluate(recommendations)
                per_user.to_parquet(directory / f"{retriever.name}_per_user.parquet", index=False)
                RecommendationTrace.from_results(recommendations).write(directory / retriever.name)
                funnel[retriever.name] = metrics
            row = {
                "recipe": recipe["name"],
                "features": recipe["features"],
                "hidden": recipe["hidden"],
                "seed": seed,
                "ranker": result,
                "cohorts": cohorts,
                "funnel": funnel,
                "seconds": time.monotonic() - started,
                "parameters": sum(p.numel() for p in model.parameters()),
                "inference_sha256": sha256(artifact),
                "checkpoint_candidate_parity": "exact",
            }
            results.append(row)
            write_json(directory / "review.json", row)
            write_json(args.out / "runs.json", results)
            del model, ranker, val_data
            gc.collect()
    aggregates = []
    for recipe in recipes:
        rows = [r for r in results if r["recipe"] == recipe["name"]]
        entry = {"recipe": recipe["name"], "seeds": [r["seed"] for r in rows]}
        for metric in ("logloss", "roc_auc"):
            values = np.array([r["ranker"]["val"][metric] for r in rows])
            paired = []
            for r in rows:
                reference = next(
                    (b for b in results if b["recipe"] == "reference" and b["seed"] == r["seed"]),
                    None,
                )
                if reference:
                    paired.append(r["ranker"]["val"][metric] - reference["ranker"]["val"][metric])
            entry.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    f"paired_{metric}_deltas_vs_reference": paired,
                }
            )
        aggregates.append(entry)
    summary = {
        "scope": f"exploratory temporal {label} screen; observed warm candidates, repeats eligible",
        "data_manifest_sha256": manifest_hash,
        "coverage": benchmark.coverage,
        "train_events": len(train),
        "val_events": len(val),
        "users": len(encoding.users),
        "items": len(encoding.items),
        "prior_baseline": prior_metrics,
        "aggregates": aggregates,
        "runs": results,
        "selection_metric": args.selection_metric,
        "selection": "No automatic promotion; review curves and cohorts before confirmation",
    }
    write_json(args.out / "summary.json", summary)
    return summary


def reuse_recall(
    source: Path,
    destination: Path,
    seed: int,
    manifest_hash: str,
    mf_epochs: int,
) -> list[ReplayRetriever]:
    """Reuse only hash-verified retrieval artifacts from the same data and seed."""
    config = json.loads((source / "config.json").read_text())
    if seed not in config["seeds"] or config["mf_epochs"] != mf_epochs:
        raise ValueError("Recall seed or training budget differs")
    directory = source / f"seed{seed}"
    evidence = json.loads((directory / "retrieval.json").read_text())
    if evidence["data_manifest_sha256"] != manifest_hash:
        raise ValueError("Recall data manifest differs")
    required = {
        "mf.pt",
        "mf_history.json",
        "recall_state.npz",
        "popularity_scores.parquet",
        "cooccurrence_scores.parquet",
        "mf_scores.parquet",
    }
    if set(evidence["files"]) != required:
        raise ValueError("Incomplete recall artifact manifest")
    for name, digest in evidence["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Recall artifact hash mismatch: {name}")
        shutil.copy2(directory / name, destination / name)
    evidence["reused_from"] = str(source.resolve())
    write_json(destination / "retrieval.json", evidence)
    return [
        ReplayRetriever(
            name,
            ScoreTable(
                pd.read_parquet(destination / f"{name}_scores.parquet"),
                evidence["files"][f"{name}_scores.parquet"],
                allow_unsupported=True,
            ),
        )
        for name in ("popularity", "cooccurrence", "mf")
    ]


def fit_recall(
    seed_dir: Path,
    seed: int,
    epochs: int,
    positive: list[np.ndarray],
    negative: list[np.ndarray],
    encoding: Any,
    cooc: SparseCosineCooccurrence,
    manifest_hash: str,
    benchmark: WindowBenchmark,
) -> list[ReplayRetriever]:
    start = time.monotonic()
    mf, supported, _ = fit_mf(positive, negative, encoding, epochs=epochs, seed=seed, out=seed_dir)
    mf_seconds = time.monotonic() - start
    # Fitted graph state is independent of labels at inference and sufficient to rebuild both baselines.
    np.savez(
        seed_dir / "recall_state.npz",
        users=encoding.users,
        items=encoding.items,
        positive_offsets=np.r_[0, np.cumsum([len(p) for p in positive])],
        positive_items=np.concatenate(positive),
        supported=supported,
    )
    retrievers = [
        PopularityRetriever(encoding.items, cooc.counts, manifest_hash + ":popularity"),
        SparseCooccurrenceRetriever(
            encoding.users, encoding.items, cooc, manifest_hash + ":cosine-exact"
        ),
        MFRetriever(mf, encoding.users, encoding.items, supported, sha256(seed_dir / "mf.pt")),
    ]
    cache_start = time.monotonic()
    cached = [benchmark.cache(r, seed_dir / f"{r.name}_scores.parquet") for r in retrievers]
    write_json(
        seed_dir / "retrieval.json",
        {
            "mf_seconds": mf_seconds,
            "cache_seconds": time.monotonic() - cache_start,
            "cooccurrence_storage_bytes": cooc.storage_bytes,
            "dense_item_matrix_bytes_avoided": len(encoding.items) ** 2 * 4,
            "mf_supported_users": int(supported.sum()),
            "mf_checkpoint_policy": "fixed final epoch",
            "data_manifest_sha256": manifest_hash,
            "coverage": benchmark.coverage,
            "files": {p.name: sha256(p) for p in seed_dir.iterdir() if p.is_file()},
        },
    )
    del mf, retrievers
    gc.collect()
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--users", type=int, default=0, help="0 selects all training users")
    prep.add_argument("--cohort-seed", type=int, default=2026)
    prep.add_argument("--train-end", type=int, default=TRAIN_END)
    prep.add_argument("--val-end", type=int, default=VAL_END)
    study = commands.add_parser("run")
    study.add_argument("--data", type=Path, required=True)
    study.add_argument("--out", type=Path, required=True)
    study.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs/kuairand_study.json",
    )
    study.add_argument("--recipes", nargs="+")
    study.add_argument("--selection-metric", choices=("logloss", "roc_auc"), default="logloss")
    study.add_argument("--resume-from", type=Path, help="Prior study; only epoch cap may change")
    study.add_argument("--recall-from", type=Path, help="Reuse verified frozen recall artifacts")
    study.add_argument(
        "--side-features", type=Path, help="Frozen metadata lookup with training-only vocabularies"
    )
    study.add_argument(
        "--include-merges",
        action="store_true",
        help="Also evaluate fixed top50-source RR and RRF60",
    )
    # reasonable defaults, not from any specific production config
    study.add_argument("--seeds", type=int, nargs="+", default=[42])
    study.add_argument("--epochs", type=int, default=5)
    study.add_argument("--mf-epochs", type=int, default=5)
    study.add_argument("--batch-size", type=int, default=4096)
    study.add_argument("--threads", type=int, default=2)
    study.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.raw,
            args.out,
            users=args.users,
            cohort_seed=args.cohort_seed,
            train_end=args.train_end,
            val_end=args.val_end,
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
