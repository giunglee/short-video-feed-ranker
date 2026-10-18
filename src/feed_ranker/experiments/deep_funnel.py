"""Evaluate frozen capped Deep checkpoints on unchanged observed recall pools.

This is inference only. Profile codes come from prepared training rows; duration
and its band use training-only video medians. Snapshot timing remains unverified.
No feature study, checkpoint selection, retriever fitting or test access occurs.
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
from torch import nn

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.kuairec_preprocessor import duration_band
from feed_ranker.data.recall import RecallData
from feed_ranker.experiments.ranking_architecture import Recipe, build_architecture, make_dataset
from feed_ranker.experiments.ranking_target import make_pool
from feed_ranker.experiments.recall_rank import candidate_features
from feed_ranker.training.recall_evaluation import evaluate as evaluate_binary

PROFILES = ["age_range", "follow_user_num_range"]


def pair_codes(frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """Collision-free packing for the nonnegative uint32 KuaiRec identifiers."""
    values = []
    for name in ("user_id", "video_id"):
        v = frame[prefix + name].to_numpy()
        if v.dtype.kind not in "iu" or (v < 0).any() or (v >= 2**32).any():
            raise ValueError("Pair IDs must be uint32-compatible integers")
        values.append(v.astype(np.uint64))
    return (values[0] << np.uint64(32)) | values[1]


def training_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Require a single encoded profile per training user; never aggregate conflicts."""
    unique = frame[["raw_user_id", *PROFILES]].drop_duplicates()
    if unique.raw_user_id.duplicated().any() or unique[PROFILES].isna().any().any():
        raise ValueError("Conflicting or missing training profile codes")
    for name in PROFILES:
        if unique[name].dtype.kind not in "iu" or unique[name].lt(0).any():
            raise ValueError("Profile codes must be nonnegative integers")
    return unique.set_index("raw_user_id")


def encode_candidates(
    raw: pd.DataFrame,
    metadata: dict[str, Any],
    profiles: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build only declared model inputs; held-out labels and event context are ignored."""
    duration = raw.video_duration.to_numpy(dtype=np.float32)
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Require positive finite candidate durations")
    result = {"video_duration": duration}
    for name, values in {
        "user_id": raw.user_id,
        "video_id": raw.video_id,
        "duration_band": duration_band(raw.video_duration),
    }.items():
        mapping = {str(v): i + 1 for i, v in enumerate(metadata["vocab"][name])}
        result[name] = values.astype("string").map(mapping).fillna(0).to_numpy(dtype=np.int64)
    aligned = profiles.reindex(raw.user_id.to_numpy())
    for name in PROFILES:
        result[name] = aligned[name].fillna(0).to_numpy(dtype=np.int64)
        if (result[name] >= len(metadata["vocab"][name]) + 1).any():
            raise ValueError("Training profile code exceeds frozen vocabulary")
    return result


@torch.no_grad()
def predict(model: nn.Module, features: dict[str, np.ndarray]) -> np.ndarray:
    model.eval()
    n = len(features["video_duration"])
    outputs = []
    for start in range(0, n, 8192):
        batch = {
            name: torch.from_numpy(v[start : start + 8192].copy()) for name, v in features.items()
        }
        value = model(batch).reshape(-1).cpu().numpy()
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite Deep predictions")
        outputs.append(value)
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def run(
    data_path: Path,
    ranking_data: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    reference_trace: Path,
    runs: list[Path],
    output: Path,
    threads: int = 1,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if threads < 1:
        raise ValueError("Threads must be positive")
    torch.set_num_threads(threads)
    if sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint differs from the frozen finalist")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    meta = saved["metadata"]
    recipe = Recipe(**meta["recipe"])
    if recipe.architecture != "deep" or recipe.feature_set != "profiles" or meta["target_cap"] != 2:
        raise ValueError("Require frozen capped Deep/profile recipe")
    if saved["training"]["selection_metric"] != "user_xauc":
        raise ValueError("Preserve the standalone user-XAUC checkpoint selector")
    if sha256(ranking_data / "meta.json") != meta["data_manifest_sha256"]:
        raise ValueError("Frozen ranking data manifest mismatch")
    dataset = make_dataset(ranking_data, recipe)
    hashes = {
        str(p): sha256(p)
        for p in (checkpoint, reference_trace, ranking_data / "meta.json", data_path / "meta.json")
    }
    recall_meta = json.loads((data_path / "meta.json").read_text())
    if dataset.meta["sources"]["big_matrix.csv"] != recall_meta["raw_big_sha256"]:
        raise ValueError("Ranking and recall use different source data")
    # Exact pair membership, not just matching split seeds or row counts.
    profile_parts = []
    for split in ("train", "val"):
        path = ranking_data / f"{split}.parquet"
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != dataset.meta["splits"][split]["sha256"]:
            raise ValueError("Prepared ranking split changed")
        codes = []
        columns = (
            ["raw_user_id", "raw_video_id", *PROFILES]
            if split == "train"
            else ["raw_user_id", "raw_video_id"]
        )
        for batch in pq.ParquetFile(path).iter_batches(batch_size=131072, columns=columns):
            frame = batch.to_pandas()
            codes.append(pair_codes(frame, "raw_"))
            if split == "train":
                profile_parts.append(frame[["raw_user_id", *PROFILES]].drop_duplicates())
        recall_pairs = pd.read_parquet(
            data_path / f"{split}.parquet", columns=["user_id", "video_id"]
        )
        if not np.array_equal(
            np.unique(np.concatenate(codes)), np.unique(pair_codes(recall_pairs))
        ):
            raise ValueError(f"Ranking/recall {split} pair membership differs")
        del codes, recall_pairs
    profiles = training_profiles(pd.concat(profile_parts, ignore_index=True))
    del profile_parts
    data = RecallData(data_path)
    for name in data.meta["files"]:
        hashes[str(data_path / name)] = sha256(data_path / name)
    model = build_architecture(dataset.config.feature, recipe, saved["training"]["seed"])
    model.load_state_dict(saved["model"])
    # Independent validation-feature reconstruction + saved prediction parity.
    check = next(
        pq.ParquetFile(ranking_data / "val.parquet").iter_batches(batch_size=8192)
    ).to_pandas()
    raw_check = check.rename(
        columns={"raw_user_id": "user_id_check", "raw_video_id": "video_id_check"}
    )
    raw_check = pd.DataFrame(
        {
            "user_id": raw_check.user_id_check,
            "video_id": raw_check.video_id_check,
            "video_duration": raw_check.video_duration,
        }
    )
    direct = encode_candidates(raw_check, dataset.meta, profiles)
    for name, values in direct.items():
        np.testing.assert_array_equal(
            values, check[name].to_numpy(), err_msg=f"Feature parity: {name}"
        )
    parity_predictions = predict(model, direct)
    trace = next(pq.ParquetFile(reference_trace).iter_batches(batch_size=len(check))).to_pandas()
    np.testing.assert_array_equal(trace.raw_user_id, check.raw_user_id)
    np.testing.assert_array_equal(trace.raw_video_id, check.raw_video_id)
    np.testing.assert_allclose(parity_predictions, trace.prediction, atol=1e-6, rtol=1e-6)
    raw = candidate_features(data)
    features = encode_candidates(raw, dataset.meta, profiles)
    scores = predict(model, features)
    pool = make_pool(data)
    output.mkdir(parents=True)
    write_json(output / "provenance.json", collect_run_provenance())
    shutil.copytree(
        Path(__file__).parents[1],
        output / "source/feed_ranker",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    np.save(output / "pool_predictions.npy", scores)
    raw[["user_id", "video_id"]].assign(score=scores).to_parquet(
        output / "rank_predictions.parquet", index=False
    )
    reports = {}
    seen_methods = set()
    for directory in runs:
        config = json.loads((directory / "config.json").read_text())
        method = config["method"]
        if method not in {"popularity", "cooccurrence", "mf"} or method in seen_methods:
            raise ValueError("Provide distinct reference retrieval methods")
        seen_methods.add(method)
        if config["data_manifest_sha256"] != sha256(data_path / "meta.json"):
            raise ValueError("Retriever data mismatch")
        if method == "mf" and config["seed"] != saved["training"]["seed"]:
            raise ValueError("Pair each ranker with the matching MF seed")
        for name in ("config.json", "scores.npy"):
            hashes[str(directory / name)] = sha256(directory / name)
        recall_scores = np.load(directory / "scores.npy", allow_pickle=False)
        binary, _, _ = evaluate_binary(data, recall_scores)
        reports[method] = {"binary": binary, "budgets": {}}
        for m in (20, 50, 100):
            candidates = pool.candidates(recall_scores, m)
            metrics, frame = pool.evaluate(scores, candidates)
            assert (frame.cap2_ndcg10 <= frame.candidate_oracle_ndcg10 + 1e-12).all()
            frame["positive_history"] = [len(v) for v in data.positive]
            frame.to_parquet(output / f"{method}_m{m}_per_user.parquet", index=False)
            reports[method]["budgets"][str(m)] = metrics
            packed = np.full((len(data.users), m), -1, dtype=np.int64)
            for u, positions in enumerate(candidates):
                packed[u, : len(positions)] = pool.items[positions]
            np.savez_compressed(
                output / f"{method}_m{m}_candidates.npz", users=data.users, items=packed
            )
            if m == 50:
                pool.duration_report(scores, candidates).to_parquet(
                    output / f"{method}_duration.parquet", index=False
                )
    full, _ = pool.evaluate(scores)
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input changed during evaluation: {path}")
    result = {
        "seed": saved["training"]["seed"],
        "epoch": saved["epoch"],
        "recipe": meta["recipe"],
        "checkpoint_sha256": checkpoint_sha256,
        "input_hashes": hashes,
        "feature_parity_rows": len(check),
        "prediction_parity_max_abs": float(
            np.abs(parity_predictions - trace.prediction.to_numpy()).max()
        ),
        "split_pair_membership": "exact train and validation",
        "test_evaluated": False,
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
            "ranker_name": f"Frozen PR6 capped Deep seed{saved['training']['seed']}",
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
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--threads", type=int, default=1)
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
    )


if __name__ == "__main__":
    main()
