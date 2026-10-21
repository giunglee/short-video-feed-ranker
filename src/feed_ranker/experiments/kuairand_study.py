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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.kuairand import (
    TRAIN_END,
    VAL_END,
    EventDataset,
    histories,
    load_prepared,
    prepare,
)
from feed_ranker.evaluation.binary import BinaryEvaluator, binary_metrics, predict_binary
from feed_ranker.evaluation.kuairand import WindowBenchmark
from feed_ranker.features.kuairand import FEATURE_SETS, KuaiRandFeatures
from feed_ranker.inference.kuairand import SparseCooccurrenceRetriever, load_ranker
from feed_ranker.inference.rankers import ModelRanker
from feed_ranker.inference.retrievers import MFRetriever, PopularityRetriever
from feed_ranker.models.ranking_factory import ModelSpec, build_model
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.traces import RecommendationTrace
from feed_ranker.training.kuairand import export_ranker, fit_mf
from feed_ranker.training.optimizer import OptimizerConfig
from feed_ranker.training.trainer import TrainConfig, Trainer


def read_recipes(path: Path, selected: list[str] | None) -> list[dict[str, Any]]:
    config = json.loads(path.read_text())
    if config["version"] != "kuairand-strict-screen-v1":
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
    if min(args.epochs, args.mf_epochs, args.batch_size, args.threads, args.budget) < 1:
        raise ValueError("Budgets and threads must be positive")
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in args.seeds)
    ):
        raise ValueError("Require distinct training seeds")
    recipes = read_recipes(args.config, args.recipes)
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
    write_json(args.out / "data_manifest.json", manifest)
    positive, negative = histories(train, encoding, label)
    benchmark = WindowBenchmark(
        val,
        encoding.items,
        label,
        manifest["train_end_exclusive_ms"],
        manifest["val_end_exclusive_ms"],
    )
    cooc = SparseCosineCooccurrence(positive, len(encoding.items))
    prior = float(np.clip(train[label].mean(), 1e-7, 1 - 1e-7))
    prior_metrics = binary_metrics(
        val[label].to_numpy(), np.full(len(val), np.log(prior / (1 - prior)))
    )
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_dir = args.out / f"seed{seed}"
        seed_dir.mkdir()
        start = time.monotonic()
        mf, supported, _ = fit_mf(
            positive, negative, encoding, epochs=args.mf_epochs, seed=seed, out=seed_dir
        )
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
        for recipe in recipes:
            started = time.monotonic()
            directory = seed_dir / recipe["name"]
            features = KuaiRandFeatures(encoding, recipe["features"])
            train_data, val_data = [
                EventDataset(f, encoding, label, recipe["features"]) for f in [train, val]
            ]
            torch.manual_seed(seed)
            model = build_model(features.config, ModelSpec("deep", tuple(recipe["hidden"])))
            config = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                train_eval_rows=0,
                loss="bce_logits",
                metrics=("logloss",),
                selection_metric="logloss",
                optimizer=OptimizerConfig(lr=0.001),
            )
            evaluator = BinaryEvaluator(lambda data=val_data: data.batches(8192))
            trainer = Trainer(
                model,
                lambda epoch, data=train_data, run_seed=seed: data.batches(
                    args.batch_size, run_seed + epoch
                ),
                config,
                loss_fn=nn.BCEWithLogitsLoss(),
                evaluator=evaluator,
            )
            print(f"recipe={recipe['name']} seed={seed}", flush=True)
            result = trainer.fit(
                directory,
                {
                    "data_manifest_sha256": manifest_hash,
                    "model_name": "DeepNN",
                    "hidden": recipe["hidden"],
                    "dropout": 0.0,
                    "use_ids": recipe["features"] not in ("no_user", "no_item"),
                    "feature_set": recipe["features"],
                    "feature_config": asdict(features.config),
                },
            )
            result["exploratory"] = True
            write_json(directory / "result.json", result)
            labels, logits = predict_binary(model, val_data.batches(8192))
            pd.DataFrame({"event_id": val.event_id.to_numpy(), "logit": logits}).to_parquet(
                directory / "predictions.parquet", index=False
            )
            warm = pd.Index(encoding.items).get_indexer(val.video_id) >= 0
            cohorts = {
                name: binary_metrics(labels[mask], logits[mask])
                for name, mask in [("warm_item", warm), ("cold_item", ~warm)]
                if mask.any()
            }
            artifact = directory / "inference.pt"
            export_ranker(
                artifact, model, encoding, recipe["features"], recipe["hidden"], manifest_hash
            )
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
            del trainer, model, ranker, train_data, val_data
            gc.collect()
    # Seeds remain explicit; singleton screens have no sample-SD estimate.
    aggregates = []
    for recipe in recipes:
        rows = [r for r in results if r["recipe"] == recipe["name"]]
        values = np.array([r["ranker"]["val"]["logloss"] for r in rows])
        paired = []
        for r in rows:
            reference = next(
                (b for b in results if b["recipe"] == "reference" and b["seed"] == r["seed"]), None
            )
            if reference:
                paired.append(r["ranker"]["val"]["logloss"] - reference["ranker"]["val"]["logloss"])
        aggregates.append(
            {
                "recipe": recipe["name"],
                "seeds": [r["seed"] for r in rows],
                "logloss_mean": float(values.mean()),
                "logloss_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "paired_logloss_deltas_vs_reference": paired,
            }
        )
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
        "selection": "No automatic promotion; review curves and cohorts before confirmation",
    }
    write_json(args.out / "summary.json", summary)
    return summary


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
