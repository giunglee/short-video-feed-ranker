"""Evaluate prepared inference artifacts on the fixed KuaiRec validation benchmark.

Replay caches scores from those artifacts for repeated budgets. Model execution
scores selected candidates at request time. Both use the same pipeline/evaluator.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.recall import RecallData
from feed_ranker.evaluation.channels import candidate_recall
from feed_ranker.evaluation.funnel import FunnelEvaluator
from feed_ranker.evaluation.kuairec import ObservedBenchmark, verify_split_membership
from feed_ranker.evaluation.retrieval import evaluate as evaluate_binary
from feed_ranker.evaluation.retrieval import score_pool
from feed_ranker.inference.artifacts import ArtifactManifest, load_ranker, load_retriever
from feed_ranker.inference.channels import ChannelRetriever
from feed_ranker.inference.replay import ReplayRanker, ReplayRetriever, ScoreTable
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.traces import RecommendationTrace


def run(
    data_path: Path,
    ranking_data: Path,
    ranker_path: Path,
    retriever_paths: list[Path],
    output: Path,
    threads: int = 1,
    execution: str = "replay",
    merge_policy: str | None = None,
    anchor: str = "mf",
    anchor_fraction: float = 0.8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if threads < 1 or execution not in ("replay", "model") or not retriever_paths:
        raise ValueError("Require positive threads, replay/model execution and retrieval artifacts")
    torch.set_num_threads(threads)
    loaded = load_ranker(ranker_path)
    provenance = loaded.manifest.provenance
    if provenance.data_manifest_sha256 != sha256(ranking_data / "meta.json"):
        raise ValueError("Ranker training data differs from benchmark")
    data = RecallData(data_path)
    hashes = verify_split_membership(data, ranking_data)
    for path in (
        ranking_data / "meta.json",
        data_path / "meta.json",
        *(data_path / n for n in data.meta["files"]),
    ):
        hashes[str(path)] = sha256(path)
    retrievers = []
    methods = set()
    artifact_ids = {"ranker": loaded.ranker.artifact_id}
    for path in retriever_paths:
        manifest = ArtifactManifest.read(path)
        if manifest.provenance.data_manifest_sha256 != sha256(data_path / "meta.json"):
            raise ValueError("Retriever data differs from benchmark")
        if manifest.kind == "mf" and manifest.provenance.seed != provenance.seed:
            raise ValueError("Pair each ranker with the matching MF seed")
        retriever = load_retriever(path)
        if retriever.name in methods:
            raise ValueError("Provide distinct retrieval methods")
        methods.add(retriever.name)
        artifact_ids[retriever.name] = retriever.artifact_id
        retrievers.append(retriever)
    for directory in (ranker_path, *retriever_paths):
        for name in ("manifest.json", "state.pt"):
            hashes[str(directory / name)] = sha256(directory / name)

    # Validate composition before writing output or scoring the benchmark.
    composition = (
        ChannelRetriever(retrievers, merge_policy, anchor=anchor, anchor_fraction=anchor_fraction)
        if merge_policy is not None
        else None
    )

    queries = ObservedBenchmark(data)
    evaluator = FunnelEvaluator(queries.pool)
    users = data.users[data.pools.users]
    # Explicit offline upper reference/cache. The request pipeline ranks retrieved IDs only.
    scores = loaded.ranker.predict(loaded.features.build(users, queries.items))
    replay_ranker = ReplayRanker(ScoreTable(queries.score_frame(scores), loaded.ranker.artifact_id))
    active_ranker = loaded.ranker if execution == "model" else replay_ranker
    output.mkdir(parents=True)
    write_json(output / "provenance.json", collect_run_provenance())
    shutil.copytree(
        Path(__file__).parents[1],
        output / "source/feed_ranker",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    np.save(output / "pool_predictions.npy", scores)
    pd.DataFrame({"user_id": users, "video_id": queries.items, "score": scores}).to_parquet(
        output / "rank_predictions.parquet", index=False
    )
    reports, active_retrievers = {}, []
    for retriever in retrievers:
        method = retriever.name
        recall_scores = score_pool(retriever, queries)
        np.save(output / f"{method}_pool_scores.npy", recall_scores)
        active_retriever = (
            retriever
            if execution == "model"
            else ReplayRetriever(
                method,
                ScoreTable(
                    queries.score_frame(recall_scores),
                    retriever.artifact_id,
                    allow_unsupported=True,
                ),
            )
        )
        binary, _, _ = evaluate_binary(data, recall_scores)
        reports[method] = {"binary": binary, "budgets": {}}
        active_retrievers.append(active_retriever)
    if composition is not None:
        merged = ChannelRetriever(
            active_retrievers, merge_policy, anchor=anchor, anchor_fraction=anchor_fraction
        )
        active_retrievers.append(merged)
        artifact_ids[merged.name] = merged.artifact_id
        reports[merged.name] = {"composition": merged.config, "budgets": {}}
    for active_retriever in active_retrievers:
        method = active_retriever.name
        for m in (20, 50, 100):
            pipeline = RecommendationPipeline(
                active_retriever, active_ranker, budget=m, k=queries.pool.k
            )
            results = [pipeline.recommend(request, scope) for request, scope in queries]
            report = evaluator.evaluate(results)
            recall = candidate_recall(queries.pool, data.pools.relevant, results)
            per_user = report.per_user.merge(recall, on="user_id", validate="one_to_one")
            report.metrics["binary_recall"] = (
                float(recall.binary_recall.mean()) if recall.positives.gt(0).any() else None
            )
            report.metrics["users_zero_positive"] = int(recall.positives.eq(0).sum())
            report.metrics["fraction_prediction_gt2"] = float((scores > 2).mean())
            per_user.insert(
                len(per_user.columns) - 2,
                "positive_history",
                [len(v) for v in data.training.positive],
            )
            per_user.to_parquet(output / f"{method}_m{m}_per_user.parquet", index=False)
            reports[method]["budgets"][str(m)] = report.metrics
            RecommendationTrace.from_results(results).write(output / f"{method}_m{m}")
            if m == 50:
                report.duration.to_parquet(output / f"{method}_duration.parquet", index=False)
    full, _ = queries.pool.evaluate(scores)
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input changed during evaluation: {path}")
    result = {
        "seed": provenance.seed,
        "epoch": provenance.epoch,
        "model_spec": asdict(loaded.spec),
        "training_provenance": asdict(provenance),
        "checkpoint_sha256": provenance.source_sha256,
        "input_hashes": hashes,
        "split_pair_membership": "exact train and validation",
        "test_evaluated": False,
        "execution": execution,
        "inference_artifacts": artifact_ids,
        "feature_policy": "Training lookups; training-median duration and derived band when configured; snapshot timing unverified",
        "methods": reports,
        "full_pool": full,
    }
    write_json(output / "summary.json", result)
    write_json(
        output / "rank_manifest.json",
        {
            "prediction_sha256": sha256(output / "rank_predictions.parquet"),
            "recall_data_manifest_sha256": sha256(data_path / "meta.json"),
            "checkpoint_sha256": provenance.source_sha256,
            "ranker_artifact": loaded.ranker.artifact_id,
            "checkpoint_selection": provenance.selection_metric,
            "scope": "Validation-only observed-pool benchmark",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "ranking-data", "ranker", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--retrievers", type=Path, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execution", choices=("replay", "model"), default="replay")
    parser.add_argument("--merge-policy", choices=("round_robin", "rrf", "anchored"))
    parser.add_argument("--anchor", default="mf")
    parser.add_argument("--anchor-fraction", type=float, default=0.8)
    args = parser.parse_args()
    run(
        args.data,
        args.ranking_data,
        args.ranker,
        args.retrievers,
        args.output,
        args.threads,
        args.execution,
        args.merge_policy,
        args.anchor,
        args.anchor_fraction,
    )


if __name__ == "__main__":
    main()
