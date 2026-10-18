"""Configure the single-retriever pipeline and export a frozen KuaiRec benchmark.

Replay and model execution use the same pipeline. Full-pool prediction is an
explicit offline reference; recommendation itself ranks only retrieved items.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.recall import RecallData
from feed_ranker.recommendation.artifacts import load_ranker, load_retriever
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.replay import ReplayRanker, ReplayRetriever, ScoreTable
from feed_ranker.training.export import export_checkpoint, export_retriever
from feed_ranker.training.funnel import (
    FrozenDeepReference,
    ObservedBenchmark,
    verify_split_membership,
)
from feed_ranker.training.recall_evaluation import evaluate as evaluate_binary


def run(
    data_path: Path,
    ranking_data: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    reference_trace: Path,
    runs: list[Path],
    output: Path,
    threads: int = 1,
    execution: str = "replay",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if threads < 1 or execution not in ("replay", "model"):
        raise ValueError("Require positive threads and replay/model execution")
    torch.set_num_threads(threads)
    reference = FrozenDeepReference.read(checkpoint, checkpoint_sha256)
    output.mkdir(parents=True)
    ranker_path = output / "inference/ranker"
    export_checkpoint(checkpoint, ranking_data, data_path / "item_duration.parquet", ranker_path)
    loaded = load_ranker(ranker_path)
    ranker = loaded.ranker
    data = RecallData(data_path)
    hashes = verify_split_membership(data, ranking_data)
    hashes.update(
        {
            str(p): sha256(p)
            for p in (
                checkpoint,
                reference_trace,
                ranking_data / "meta.json",
                data_path / "meta.json",
                *(data_path / name for name in data.meta["files"]),
            )
        }
    )
    parity = reference.verify_predictions(loaded, ranking_data, reference_trace)
    queries = ObservedBenchmark(data)
    pool = queries.pool
    # Offline full-pool reference and reusable keyed replay cache, never request inputs.
    users = data.users[data.pools.users]
    scores = ranker.predict(loaded.features.build(users, queries.items))
    replay_ranker = ReplayRanker(ScoreTable(queries.score_frame(scores), checkpoint_sha256))
    active_ranker = ranker if execution == "model" else replay_ranker
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
    reports = {}
    for directory in runs:
        config = json.loads((directory / "config.json").read_text())
        method = config["method"]
        if method not in ("popularity", "cooccurrence", "mf") or method in reports:
            raise ValueError("Provide distinct reference retrieval methods")
        if config["data_manifest_sha256"] != sha256(data_path / "meta.json"):
            raise ValueError("Retriever data mismatch")
        if method == "mf" and config["seed"] != reference.seed:
            raise ValueError("Pair each ranker with the matching MF seed")
        for name in ("config.json", "scores.npy"):
            hashes[str(directory / name)] = sha256(directory / name)
        recall_scores = np.load(directory / "scores.npy", allow_pickle=False)
        retriever = ReplayRetriever(
            method,
            ScoreTable(
                queries.score_frame(recall_scores),
                hashes[str(directory / "scores.npy")],
                allow_unsupported=True,
            ),
        )
        if execution == "model":
            retriever_path = output / "inference" / method
            export_retriever(data, directory, retriever_path)
            retriever = load_retriever(retriever_path)
            if method == "mf":
                hashes[str(directory / "best.pt")] = sha256(directory / "best.pt")
        binary, _, _ = evaluate_binary(data, recall_scores)
        reports[method] = {"binary": binary, "budgets": {}}
        for m in (20, 50, 100):
            pipeline = RecommendationPipeline(retriever, active_ranker, budget=m, k=pool.k)
            results = [pipeline.recommend(request, scope) for request, scope in queries]
            candidates, finals = queries.result_positions(results)
            expected = pool.candidates(recall_scores, m)
            for actual, expected_items in zip(candidates, expected, strict=True):
                np.testing.assert_array_equal(
                    actual, expected_items, err_msg="Frozen retrieval parity"
                )
            metrics, frame = pool.evaluate_ordered(candidates, finals)
            metrics["fraction_prediction_gt2"] = float((scores > 2).mean())
            frame["positive_history"] = [len(v) for v in data.training.positive]
            frame["pool_shortfall"] = [r.pool_shortfall for r in results]
            frame["method_shortfall"] = [r.method_shortfall for r in results]
            frame.to_parquet(output / f"{method}_m{m}_per_user.parquet", index=False)
            reports[method]["budgets"][str(m)] = metrics
            packed = np.full((len(results), m), -1, dtype=np.int64)
            traces = []
            for u, result in enumerate(results):
                packed[u, : len(result.retrieved.items)] = result.retrieved.items
                traces.append(
                    pd.DataFrame(
                        {
                            "query_id": result.request.query_id,
                            "video_id": result.retrieved.items,
                            "source_rank": np.arange(1, len(result.retrieved.items) + 1),
                            "retrieval_score": result.retrieved.scores,
                            "ranking_score": result.ranked.scores,
                            "final_rank": pd.Series(
                                np.arange(1, len(result.items) + 1), index=result.items
                            )
                            .reindex(result.retrieved.items)
                            .fillna(0)
                            .to_numpy(dtype=int),
                        }
                    )
                )
            np.savez_compressed(
                output / f"{method}_m{m}_candidates.npz", users=data.users, items=packed
            )
            pd.concat(traces, ignore_index=True).to_parquet(
                output / f"{method}_m{m}_trace.parquet", index=False
            )
            if m == 50:
                pool.duration_ordered(finals).to_parquet(
                    output / f"{method}_duration.parquet", index=False
                )
    full, _ = pool.evaluate(scores)
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input changed during evaluation: {path}")
    result = {
        "seed": reference.seed,
        "epoch": reference.epoch,
        "recipe": asdict(reference.recipe),
        "checkpoint_sha256": checkpoint_sha256,
        "input_hashes": hashes,
        **parity,
        "split_pair_membership": "exact train and validation",
        "test_evaluated": False,
        "execution": execution,
        "inference_artifacts": {
            str(path.parent.relative_to(output)): sha256(path)
            for path in sorted((output / "inference").glob("*/manifest.json"))
        },
        "feature_policy": "Training profiles; training-median duration and derived band; snapshot timing unverified",
        "methods": reports,
        "full_pool": full,
    }
    write_json(output / "summary.json", result)
    write_json(
        output / "rank_manifest.json",
        {
            "prediction_sha256": sha256(output / "rank_predictions.parquet"),
            "recall_data_manifest_sha256": sha256(data_path / "meta.json"),
            "checkpoint_sha256": checkpoint_sha256,
            "ranker_name": f"Frozen PR6 capped Deep seed{result['seed']}",
            "feature_policy": result["feature_policy"],
            "checkpoint_selection": "validation user-XAUC; earliest maximum",
            "scope": "Frozen architecture integration; feature exploration separate",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "ranking-data", "checkpoint", "reference-trace", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execution", choices=("replay", "model"), default="replay")
    args = parser.parse_args()
    run(
        args.data,
        args.ranking_data,
        args.checkpoint,
        args.checkpoint_sha256,
        args.reference_trace,
        args.runs,
        args.output,
        args.threads,
        args.execution,
    )


if __name__ == "__main__":
    main()
