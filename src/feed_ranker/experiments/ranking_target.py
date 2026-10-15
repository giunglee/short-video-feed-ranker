"""Seed-42 raw versus cap-2 target pilot on frozen Recall validation pools.

Run from a frozen source snapshot. Event-level MAE training stays unchanged except
for target capping; all checkpoints use first-event full-pool linear cap-2 NDCG10.
No small-matrix access. This pilot precedes any five-seed confirmation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.recall import RecallData
from feed_ranker.experiments.recall_rank import candidate_features
from feed_ranker.experiments.wide_deep_kuairec import build_model, verify_data
from feed_ranker.training.evaluation import RegressionEvaluator, predict_regression
from feed_ranker.training.funnel_evaluation import GainPool, TargetMAE
from feed_ranker.training.optimizer import OptimizerConfig
from feed_ranker.training.recall_evaluation import evaluate as evaluate_binary
from feed_ranker.training.trainer import TrainConfig, Trainer

METHODS = ("popularity", "cooccurrence", "mf")
VERSION = "first-event-linear-cap2-full-pool-ndcg10-v1"


class FunnelEvaluator:
    """Raw-label event diagnostics plus a separate serving-feature selection pool."""

    def __init__(
        self,
        regression: RegressionEvaluator,
        pool: GainPool,
        batches: list[dict[str, torch.Tensor]],
    ) -> None:
        self.regression = regression
        self.loaders = regression.loaders
        self.pool = pool
        self.batches = batches

    @torch.no_grad()
    def predict_pool(self, model: nn.Module) -> np.ndarray:
        model.eval()
        return np.concatenate(
            [
                predict_regression(model, batch, len(batch["video_duration"]), "cpu").numpy()
                for batch in self.batches
            ]
        )

    def evaluate(self, model: nn.Module, split: str) -> dict[str, Any]:
        result = self.regression.evaluate(model, split)
        if split == "val":
            scores = self.predict_pool(model)
            funnel, _ = self.pool.evaluate(scores)
            result.update(funnel)
        return result


def make_pool(data: RecallData) -> GainPool:
    val = pd.read_parquet(data.directory / "val.parquet").set_index(["user_id", "video_id"])
    keys = pd.MultiIndex.from_arrays([data.users[data.pools.users], data.items[data.pools.items]])
    aligned = val.reindex(keys)
    if not np.array_equal(aligned.outcome.eq(1).to_numpy(), data.pools.relevant):
        raise ValueError("First-event labels do not match frozen evaluation pool")
    return GainPool(
        data.users,
        data.items[data.pools.items],
        data.pools.offsets,
        aligned.watch_ratio.to_numpy(),
        aligned.video_duration.to_numpy(),
    )


def encode_pool(
    data: RecallData, meta: dict[str, Any], batch_size: int
) -> list[dict[str, torch.Tensor]]:
    raw = candidate_features(data)
    values = {"video_duration": raw.video_duration.to_numpy(dtype=np.float32)}
    for name in ("user_id", "video_id"):
        mapping = {str(value): i + 1 for i, value in enumerate(meta["vocab"][name])}
        values[name] = raw[name].astype("string").map(mapping).fillna(0).to_numpy(dtype=np.int64)
    return [
        {
            name: torch.from_numpy(value[start : start + batch_size].copy())
            for name, value in values.items()
        }
        for start in range(0, len(raw), batch_size)
    ]


def event_audit(directory: Path) -> dict[str, Any]:
    result = {}
    for split in ("train", "val"):
        frame = pd.read_parquet(
            directory / f"{split}.parquet", columns=["watch_ratio", "video_duration"]
        )
        ratio = frame.watch_ratio.to_numpy(dtype=float)
        if not np.isfinite(ratio).all() or (ratio < 0).any():
            raise ValueError("Invalid raw event labels")
        result[split] = {
            "rows": len(ratio),
            "fraction_gt2": float((ratio > 2).mean()),
            "raw_mean": float(ratio.mean()),
            "cap2_mean": float(np.minimum(ratio, 2).mean()),
            "ratio_mass_gt2": float(ratio[ratio > 2].sum() / ratio.sum()),
            "max": float(ratio.max()),
            "duration_gt2": {},
        }
        for group, mask in {
            "le10s": frame.video_duration.le(10000),
            "gt60s": frame.video_duration.gt(60000),
        }.items():
            result[split]["duration_gt2"][group] = float((ratio[mask] > 2).mean())
    return result


def verify_inputs(ranking: Path, recall: Path, reference: Path) -> dict[str, str]:
    saved = torch.load(reference / "best.pt", map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    if sha256(ranking / "meta.json") != meta["data_manifest_sha256"]:
        raise ValueError("Historical checkpoint data mismatch")
    for source, expected in meta["source_sha256"].items():
        if sha256(Path(source)) != expected:
            raise ValueError(f"Historical source changed: {source}")
    recall_meta = json.loads((recall / "meta.json").read_text())
    if recall_meta["ranking_manifest_sha256"] != meta["data_manifest_sha256"]:
        raise ValueError("Recall and ranking use different split origins")
    hashes = {
        str(ranking / "meta.json"): sha256(ranking / "meta.json"),
        str(recall / "meta.json"): sha256(recall / "meta.json"),
        str(reference / "best.pt"): sha256(reference / "best.pt"),
    }
    for split in ("train", "val"):
        path = ranking / f"{split}.parquet"
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != recall_meta["ranking_split_sha256"][split]:
            raise ValueError("Ranking split changed since Recall preparation")
    for name, expected in recall_meta["files"].items():
        path = recall / name
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != expected:
            raise ValueError(f"Recall input changed: {name}")
    # Guard the historical model, batching and optimizer implementations used at runtime.
    source_root = Path(__file__).resolve().parents[1]
    frozen_root = reference.parent / "source/feed_ranker"
    for relative in [
        "data/kuairec_dataset.py",
        "data/config.py",
        "data/schema.py",
        "models/wide_deep.py",
        "models/components/feature_encoder.py",
        "models/components/mlp.py",
        "training/optimizer.py",
        "training/evaluation.py",
    ]:
        if sha256(source_root / relative) != sha256(frozen_root / relative):
            raise ValueError(f"Runtime differs from historical training implementation: {relative}")
    return hashes


def export_funnel(
    output: Path,
    model: nn.Module,
    evaluator: FunnelEvaluator,
    data: RecallData,
    scores: dict[str, np.ndarray],
    candidates: dict[str, dict[int, list[np.ndarray]]],
) -> dict[str, Any]:
    predicted = evaluator.predict_pool(model)
    np.save(output / "pool_predictions.npy", predicted)
    summary = {}
    full, users = evaluator.pool.evaluate(predicted)
    users.to_csv(output / "full_pool_per_user.csv", index=False)
    summary["full_pool"] = full
    for method in METHODS:
        summary[method] = {}
        binary, users, _ = evaluate_binary(data, scores[method], predicted)
        users.to_csv(output / f"{method}_binary_per_user.csv", index=False)
        summary[method]["binary"] = binary
        for budget in (20, 50, 100):
            metrics, users = evaluator.pool.evaluate(predicted, candidates[method][budget])
            users.to_csv(output / f"{method}_m{budget}_per_user.csv", index=False)
            summary[method][str(budget)] = metrics
        duration = evaluator.pool.duration_report(predicted, candidates[method][50])
        duration.groupby("duration").agg(
            users=("user_id", "size"),
            mean_selected_slots=("selected_slots", "mean"),
            mean_utility_per_slot10=("utility_per_slot10", "mean"),
            eligible_users=("duration_ndcg10", "count"),
            duration_ndcg10=("duration_ndcg10", "mean"),
        ).to_csv(output / f"{method}_m50_duration.csv")
    write_json(output / "funnel_summary.json", summary)
    return summary


def write_review(output: Path, results: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [
        "# Raw versus capped-target ranking pilot",
        "",
        "Seed 42, validation only. Both arms retrained for 120 epochs and selected by maximum",
        "full-pool first-event linear cap-2 NDCG@10; earliest exact tie. Test was not read.",
        "",
        "Training preserves deduplicated event rows, event weighting, MAE, IDs/duration,",
        "AdamW lr=.001, batch1024, hidden[128,32], dropout0. Only training target capping differs.",
        "Candidate inputs use training-only median item duration; labels use held-out first events.",
        "",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for arm, result in results.items():
        history = json.loads((output / arm / "history.json").read_text())
        for ax, metric in zip(axes, ("cap2_ndcg10", "mae", "fraction_prediction_gt2"), strict=True):
            ax.plot(
                [r["epoch"] + 1 for r in history], [r["val"][metric] for r in history], label=arm
            )
            ax.set(xlabel="Epoch", ylabel=metric)
            ax.legend()
        lines += [
            f"## {arm}",
            "",
            f"- Selected epoch: {result['training']['best_epoch'] + 1}.",
            f"- Full-pool cap-2 NDCG10: {result['funnel']['full_pool']['cap2_ndcg10']:.8f}.",
            f"- Raw-event validation MAE: {result['training']['val']['mae']:.8f}.",
            f"- Training/evaluation wall time: {result['fit_seconds']:.1f} seconds.",
        ]
        for method in METHODS:
            m = result["funnel"][method]["50"]
            lines.append(
                f"- {method}, M50: cap-2 NDCG10 {m['cap2_ndcg10']:.8f}; candidate oracle {m['candidate_oracle_ndcg10']:.8f}."
            )
        lines += [""]
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    delta = (
        results["capped"]["funnel"]["full_pool"]["cap2_ndcg10"]
        - results["raw"]["funnel"]["full_pool"]["cap2_ndcg10"]
    )
    lines += [
        "## Paired seed-42 differences (capped minus raw)",
        "",
        f"- Full-pool cap-2 NDCG10: {delta:+.8f}.",
    ]
    for method in METHODS:
        for budget in (20, 50, 100):
            a = results["raw"]["funnel"][method][str(budget)]
            b = results["capped"]["funnel"][method][str(budget)]
            lines.append(f"- {method}, M{budget}: {b['cap2_ndcg10'] - a['cap2_ndcg10']:+.8f}.")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- This is one exploratory paired seed, not five-seed confirmation or population uncertainty.",
        "- Capping MAE targets changes gradients only where predictions cross the cap for tail labels; it remains a median objective, not direct utility/NDCG optimization.",
        "- Selection now optimizes full-pool ranking under capped gain. Historical MAE-selected checkpoints are descriptive references, not the new control.",
        "- Inspect selected epochs, neighbors, duration cohorts and all fixed candidate budgets before choosing the next experiment.",
        "- Frozen MF remains selected under its original binary Recall50 rule. This isolates the ranker target, not joint funnel tuning.",
        "- Observed-pool results are exposure-conditional; random pair holdout does not establish future behavior or online utility.",
        "- Human review precedes matched-seed confirmation, architecture changes or test evaluation.",
    ]
    (output / "review_summary.md").write_text("\n".join(lines) + "\n")


def run(output: Path, ranking: Path, recall_root: Path, reference: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    if (output / "protocol.json").exists():
        raise FileExistsError("Use a fresh comparison directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"stage": "preflight", "test_evaluated": False})
    hashes = verify_inputs(ranking, recall_root / "data_v1", reference)
    historical = json.loads((reference / "config.json").read_text())
    if str(torch.__version__) != historical["runtime"]["torch_version"]:
        raise ValueError("Torch runtime differs from reference")
    data = RecallData(recall_root / "data_v1")
    pool = make_pool(data)
    config_values = dict(historical["training"])
    config_values.update(
        metrics=("mae", "xauc", "user_xauc", "cap2_ndcg10"),
        selection_metric="cap2_ndcg10",
        selection_mode="max",
    )
    config_values["optimizer"] = OptimizerConfig(**config_values["optimizer"])
    config = TrainConfig(**config_values)
    model, dataset = build_model(ranking, "wide_deep_nn", [128, 32], 0.0, True)
    verify_data(dataset, ("train", "val"))
    if dataset.meta["spec"] != {
        "categorical": ["user_id", "video_id"],
        "dense": ["video_duration"],
        "crosses": {},
    }:
        raise ValueError("Expected IDs/duration feature recipe")
    batches = encode_pool(data, dataset.meta, config.eval_batch_size)
    regression = RegressionEvaluator(
        {
            "val": dataset.loader("val", config.eval_batch_size),
            "train_diagnostic": dataset.fixed_train_loader(
                config.train_eval_rows, config.train_eval_seed, config.eval_batch_size
            ),
        },
        ("mae", "xauc", "user_xauc"),
        "cpu",
        config.metric_seed,
        config.xauc_pairs,
        config.user_pairs,
    )
    evaluator = FunnelEvaluator(regression, pool, batches)
    audit = event_audit(ranking)
    write_json(output / "event_label_audit.json", audit)
    scores = {}
    candidates = {}
    for method in METHODS:
        directory = recall_root / method
        cfg = json.loads((directory / "config.json").read_text())
        if cfg["data_manifest_sha256"] != sha256(data.directory / "meta.json"):
            raise ValueError("Retriever pool mismatch")
        for name in ("scores.npy", "candidates.npy", "config.json"):
            hashes[str(directory / name)] = sha256(directory / name)
        scores[method] = np.load(directory / "scores.npy")
        candidates[method] = {m: pool.candidates(scores[method], m) for m in (20, 50, 100)}
        expected = np.load(directory / "candidates.npy")
        actual = np.full_like(expected, -1)
        for u, positions in enumerate(candidates[method][100]):
            actual[u, : len(positions)] = data.pools.items[positions]
        np.testing.assert_array_equal(actual, expected)
    saved = torch.load(reference / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model"])
    previous = evaluator.predict_pool(model)
    np.testing.assert_array_equal(previous, np.load(recall_root / "integration/rank_scores.npy"))
    hashes[str(recall_root / "integration/rank_scores.npy")] = sha256(
        recall_root / "integration/rank_scores.npy"
    )
    prior, _ = pool.evaluate(previous)
    provenance = collect_run_provenance()
    protocol = {
        "version": VERSION,
        "training": asdict(config),
        "targets": {"raw": None, "capped": 2.0},
        "features": dataset.meta["spec"],
        "hidden": [128, 32],
        "dropout": 0,
        "headline_M": 50,
        "reported_M": [20, 50, 100],
        "K": 10,
        "gain_cap": 2,
        "tie_seed": 2026,
        "input_hashes": hashes,
        "provenance": provenance,
        "selection": "maximum full-pool macro cap2 NDCG10, earliest exact tie",
        "historical_checkpoint_full_pool": prior,
        "test_evaluated": False,
        "verification": "Exact historical prediction parity; saved candidate identity; source/data hashes",
        "optimizer_class": "torch.optim.AdamW",
        "seed_scope": "ranker seed42; retrieval artifacts fixed",
    }
    write_json(output / "protocol.json", protocol)
    results = {}
    for arm, cap in (("raw", None), ("capped", 2.0)):
        write_json(
            output / "status.json",
            {
                "stage": "training",
                "arm": arm,
                "completed_arms": len(results),
                "test_evaluated": False,
            },
        )
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        model, _ = build_model(ranking, "wide_deep_nn", [128, 32], 0, True)
        trainer = Trainer(
            model,
            lambda epoch: dataset.loader("train", config.batch_size, config.seed + epoch),
            config,
            loss_fn=TargetMAE(cap),
            evaluator=evaluator,
        )
        metadata = {
            "model_name": "wide_deep_nn",
            "hidden": [128, 32],
            "dropout": 0.0,
            "use_ids": True,
            "target_cap": cap,
            "metric_version": VERSION,
            "data_manifest_sha256": sha256(ranking / "meta.json"),
            "comparison_protocol_sha256": sha256(output / "protocol.json"),
            **provenance,
        }
        fit_started = time.monotonic()
        training = trainer.fit(output / arm, metadata)
        training["steps_truncated"] = training.pop("exploratory")
        training["exploratory"] = True
        training["experiment_scope"] = "single-seed target-capping pilot"
        write_json(output / arm / "result.json", training)
        fit_seconds = time.monotonic() - fit_started
        funnel = export_funnel(output / arm, model, evaluator, data, scores, candidates)
        results[arm] = {"training": training, "funnel": funnel, "fit_seconds": fit_seconds}
        write_json(output / "results.json", results)
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Input changed during training: {filename}")
    for filename, expected in provenance["source_sha256"].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Training source changed: {filename}")
    write_review(output, results)
    write_json(
        output / "status.json",
        {
            "stage": "complete",
            "completed_arms": 2,
            "wall_seconds": time.monotonic() - started,
            "test_evaluated": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ranking-data", type=Path, default=Path("artifacts/training_diagnostics/baseline_data")
    )
    parser.add_argument("--recall-root", type=Path, default=Path("artifacts/recall_run"))
    parser.add_argument(
        "--reference", type=Path, default=Path("artifacts/ablations_round1/baseline_event-42")
    )
    args = parser.parse_args()
    existing_protocol = (args.output / "protocol.json").exists()
    try:
        run(args.output, args.ranking_data, args.recall_root, args.reference)
    except Exception as error:
        if args.output.exists() and not existing_protocol:
            write_json(args.output / "status.json", {"stage": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
