"""Configure the single-retriever pipeline and export a frozen KuaiRec benchmark.

Replay and model execution use the same pipeline. Full-pool prediction is an
explicit offline reference; recommendation itself ranks only retrieved items.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.funnel import (
    ObservedQueries,
    load_retriever,
    make_pool,
    result_positions,
    verify_split_membership,
)
from feed_ranker.data.recall import RecallData
from feed_ranker.data.recommendation import encode_candidates
from feed_ranker.recommendation.pipeline import RecommendationPipeline
from feed_ranker.recommendation.ranking import ModelRanker, load_deep_ranker, predict
from feed_ranker.recommendation.replay import ReplayRanker, ReplayRetriever, ScoreTable
from feed_ranker.training.recall_evaluation import evaluate as evaluate_binary


def verify_predictions(
    ranker: ModelRanker, ranking_data: Path, reference_trace: Path
) -> dict[str, int | float]:
    """Independent prepared-feature reconstruction and frozen prediction parity."""
    check = next(
        pq.ParquetFile(ranking_data / "val.parquet").iter_batches(batch_size=8192)
    ).to_pandas()
    raw = pd.DataFrame(
        {
            "user_id": check.raw_user_id,
            "video_id": check.raw_video_id,
            "video_duration": check.video_duration,
        }
    )
    direct = encode_candidates(raw, ranker.features.metadata, ranker.features.profiles)
    for name, values in direct.items():
        np.testing.assert_array_equal(
            values, check[name].to_numpy(), err_msg=f"Feature parity: {name}"
        )
    predictions = predict(ranker.model, direct)
    trace = next(pq.ParquetFile(reference_trace).iter_batches(batch_size=len(check))).to_pandas()
    np.testing.assert_array_equal(trace.raw_user_id, check.raw_user_id)
    np.testing.assert_array_equal(trace.raw_video_id, check.raw_video_id)
    np.testing.assert_allclose(predictions, trace.prediction, atol=1e-6, rtol=1e-6)
    return {
        "feature_parity_rows": len(check),
        "prediction_parity_max_abs": float(np.abs(predictions - trace.prediction.to_numpy()).max()),
    }


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
    ranker, saved = load_deep_ranker(
        checkpoint, checkpoint_sha256, ranking_data, data_path / "item_duration.parquet"
    )
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
    parity = verify_predictions(ranker, ranking_data, reference_trace)
    queries, pool = ObservedQueries(data), make_pool(data)
    # Offline full-pool reference and reusable keyed replay cache, never request inputs.
    users = data.users[data.pools.users]
    scores = predict(ranker.model, ranker.features.build(users, queries.items))
    replay_ranker = ReplayRanker(ScoreTable(queries.score_frame(scores), checkpoint_sha256))
    active_ranker = ranker if execution == "model" else replay_ranker
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
    reports = {}
    for directory in runs:
        config = json.loads((directory / "config.json").read_text())
        method = config["method"]
        if method not in ("popularity", "cooccurrence", "mf") or method in reports:
            raise ValueError("Provide distinct reference retrieval methods")
        if config["data_manifest_sha256"] != sha256(data_path / "meta.json"):
            raise ValueError("Retriever data mismatch")
        if method == "mf" and config["seed"] != saved["training"]["seed"]:
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
            retriever = load_retriever(data, directory)
            if method == "mf":
                hashes[str(directory / "best.pt")] = sha256(directory / "best.pt")
        binary, _, _ = evaluate_binary(data, recall_scores)
        reports[method] = {"binary": binary, "budgets": {}}
        for m in (20, 50, 100):
            pipeline = RecommendationPipeline(retriever, active_ranker, budget=m, k=pool.k)
            results = [pipeline.recommend(request, scope) for request, scope in queries]
            candidates, finals = result_positions(pool, results)
            expected = pool.candidates(recall_scores, m)
            for actual, reference in zip(candidates, expected, strict=True):
                np.testing.assert_array_equal(actual, reference, err_msg="Frozen retrieval parity")
            metrics, frame = pool.evaluate_ordered(candidates, finals)
            metrics["fraction_prediction_gt2"] = float((scores > 2).mean())
            frame["positive_history"] = [len(v) for v in data.positive]
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
        "seed": saved["training"]["seed"],
        "epoch": saved["epoch"],
        "recipe": saved["metadata"]["recipe"],
        "checkpoint_sha256": checkpoint_sha256,
        "input_hashes": hashes,
        **parity,
        "split_pair_membership": "exact train and validation",
        "test_evaluated": False,
        "execution": execution,
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
