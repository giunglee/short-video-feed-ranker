"""Validation-only Deep feature ablations: screen six arms, confirm at most two.

Reuse frozen architecture-study baselines after compatibility checks. No test
loader is constructed. Feature decisions use retraining; permutation diagnostics
inspect frozen models and never select checkpoints or confirmation candidates.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.config import DataConfig, LabelConfig
from feed_ranker.data.feature_ablation import BASE_FEATURES, PROFILE_ADDITIONS, prepare_feature_data
from feed_ranker.data.kuairec_dataset import KuaiRecDataset
from feed_ranker.experiments.diagnostics import export_validation_trace
from feed_ranker.experiments.ranking_architecture import (
    METRICS,
    evaluator,
)
from feed_ranker.experiments.ranking_recipe import Recipe, build_architecture
from feed_ranker.features.config import (
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
    FeatureConfig,
)
from feed_ranker.models.ranking_factory import ModelSpec
from feed_ranker.training.losses import TargetMAE
from feed_ranker.training.optimizer import OptimizerConfig
from feed_ranker.training.trainer import TrainConfig, Trainer

ARMS = {
    "baseline": BASE_FEATURES,
    "no_age": tuple(n for n in BASE_FEATURES if n != "age_range"),
    "no_follow": tuple(n for n in BASE_FEATURES if n != "follow_user_num_range"),
    "no_profiles": tuple(
        n for n in BASE_FEATURES if n not in ("age_range", "follow_user_num_range")
    ),
    "no_duration_band": tuple(n for n in BASE_FEATURES if n != "duration_band"),
    "add_categories": (*BASE_FEATURES, "item_categories"),
    "add_activity_profiles": (*BASE_FEATURES, *PROFILE_ADDITIONS),
}
RECIPE = Recipe("deep", hidden=(64, 32))


def make_dataset(data: Path, arm: str) -> KuaiRecDataset:
    metadata = json.loads((data / "meta.json").read_text())
    features = FeatureConfig(
        dense_configs=[DenseFeatureConfig("video_duration", "log1p_nonnegative")],
        embedding_configs=[
            EmbeddingFeatureConfig(
                name,
                len(metadata["vocab"][name]) + 1,
                8,
                "sum" if name == "item_categories" else None,
            )
            for name in ARMS[arm]
        ],
    )
    return KuaiRecDataset(DataConfig(features, LabelConfig("watch_ratio")), data)


def select_candidates(records: list[dict[str, Any]]) -> list[str]:
    """At most two strictly positive seed-42 gains; declared arm order breaks ties."""
    scores = {r["arm"]: r["user_xauc"] for r in records if r["seed"] == 42}
    if set(scores) != set(ARMS) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("Screening requires finite scores for the baseline and all six arms")
    ordered = sorted((a for a in ARMS if a != "baseline"), key=lambda a: -scores[a])
    return [a for a in ordered if scores[a] > scores["baseline"]][:2]


def fit_arm(
    data: Path,
    output: Path,
    arm: str,
    seed: int,
    provenance: dict[str, Any],
    *,
    epochs: int = 120,
    steps: int | None = None,
) -> dict[str, Any]:
    dataset = make_dataset(data, arm)
    model = build_architecture(dataset.config.feature, RECIPE, seed)
    config = TrainConfig(
        epochs=epochs,
        seed=seed,
        batch_size=1024,
        eval_batch_size=8192,
        steps=steps,
        selection_metric="user_xauc",
        selection_mode="max",
        optimizer=OptimizerConfig(lr=0.001, weight_decay=0.0),
    )
    metadata = {
        **provenance,
        "arm": arm,
        "features": list(ARMS[arm]),
        "recipe": asdict(RECIPE),
        "model_spec": asdict(ModelSpec("deep", RECIPE.hidden)),
        "feature_config": asdict(dataset.config.feature),
        "target_cap": 2.0,
        "model_name": "deep",
        "hidden": [64, 32],
        "dropout": 0.0,
        "use_ids": True,
        "data_manifest_sha256": sha256(data / "meta.json"),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    trainer = Trainer(
        model,
        lambda epoch: dataset.loader("train", 1024, seed + epoch),
        config,
        loss_fn=TargetMAE(2),
        evaluator=evaluator(dataset, ("val",), train_diagnostic=True),
    )
    started = time.monotonic()
    result = trainer.fit(output, metadata)
    result.update(arm=arm, run=str(output), fit_seconds=time.monotonic() - started)
    write_json(output / "result.json", result)
    return result


def score_arm(data: Path, run: Path, arm: str, output: Path) -> dict[str, Any]:
    dataset = make_dataset(data, arm)
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if sha256(data / "meta.json") != saved["metadata"]["data_manifest_sha256"]:
        raise ValueError("Checkpoint/data manifest mismatch")
    model = build_architecture(dataset.config.feature, RECIPE, saved["training"]["seed"])
    model.load_state_dict(saved["model"])
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "arm": arm,
        "seed": saved["training"]["seed"],
        "epoch": saved["epoch"] + 1,
        "checkpoint_sha256": sha256(run / "best.pt"),
        **evaluator(dataset, ("val",), detailed=True).evaluate(model, "val"),
    }
    write_json(output / "metrics.json", result)
    export_validation_trace(
        model,
        dataset,
        run / "best.pt",
        output / "trace",
        feature_allowlist=["video_duration"],
        label_cap=2,
    )
    return result


def reuse_baselines(study: Path, base: Path) -> tuple[list[dict[str, Any]], dict[int, Path]]:
    """Verify frozen contracts, stored predictions and current detailed validation scores.

    Parity on stored predictions checks loading compatibility, not training dynamics.
    The recorded study used the original source; new runs record their own provenance.
    """
    frozen = json.loads((study / "finalists.json").read_text())
    selected = [r for r in frozen["runs"] if r["recipe"]["architecture"] == "deep"]
    if sorted(r["seed"] for r in selected) != list(range(42, 47)):
        raise ValueError("Expected five frozen Deep baselines")
    dataset = make_dataset(base, "baseline")
    check = next(iter(dataset.loader("val", 8192)))
    records, paths = [], {}
    for entry in selected:
        run = Path(entry["run"])
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        expected = TrainConfig(
            epochs=120, seed=entry["seed"], selection_metric="user_xauc", selection_mode="max"
        )
        if saved["training"] != asdict(expected) or Recipe(**entry["recipe"]) != RECIPE:
            raise ValueError("Baseline training recipe changed")
        if (
            saved["metadata"].get("target_cap") != 2
            or Recipe(**saved["metadata"]["recipe"]) != RECIPE
        ):
            raise ValueError("Baseline checkpoint objective or model recipe changed")
        if saved["metadata"]["data_manifest_sha256"] != sha256(base / "meta.json"):
            raise ValueError("Baseline data changed")
        metrics = json.loads(
            (study / "evaluation/val" / f"deep_{entry['seed']}" / "metrics.json").read_text()
        )
        if any(
            metrics.get(key) != value
            for key, value in {
                "split": "val",
                "target_cap": 2,
                "global_pairs": 1_000_000,
                "pairs_per_user": 4096,
                "seed": entry["seed"],
            }.items()
        ):
            raise ValueError("Baseline evaluation protocol differs from the feature study")
        actual = sha256(run / "best.pt")
        if actual != entry["checkpoint_sha256"] or actual != metrics["checkpoint_sha256"]:
            raise ValueError("Baseline checkpoint changed")
        model = build_architecture(dataset.config.feature, RECIPE, entry["seed"])
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        import pyarrow.parquet as pq

        trace = next(
            pq.ParquetFile(
                study / f"evaluation/val/deep_{entry['seed']}/trace/validation_trace.parquet"
            ).iter_batches(batch_size=len(check["label"]))
        ).to_pandas()
        np.testing.assert_array_equal(check["raw_user_id"].numpy(), trace.raw_user_id)
        np.testing.assert_array_equal(check["raw_video_id"].numpy(), trace.raw_video_id)
        with torch.no_grad():
            predictions = model(check["feature"]).reshape(-1).numpy()
        np.testing.assert_allclose(predictions, trace.prediction, rtol=1e-6, atol=1e-6)
        current = evaluator(dataset, ("val",), detailed=True).evaluate(model, "val")
        for name in METRICS:
            if not np.isclose(current[name], metrics["metrics"][name], rtol=0, atol=1e-12):
                raise ValueError(f"Baseline metric implementation changed: {name}")
        records.append(
            {
                "arm": "baseline",
                "seed": entry["seed"],
                "epoch": metrics["epoch"] + 1,
                "checkpoint_sha256": actual,
                **metrics["metrics"],
            }
        )
        paths[entry["seed"]] = run
    return records, paths


def summarize(output: Path, records: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_seed.csv", index=False)
    frame.groupby("arm")[list(METRICS)].agg(["mean", "std", "count"]).to_csv(output / "means.csv")
    baseline = frame[frame.arm.eq("baseline")]
    paired = frame[~frame.arm.eq("baseline")].merge(
        baseline, on="seed", suffixes=("", "_baseline"), validate="many_to_one"
    )
    for metric in METRICS:
        paired[metric] -= paired[f"{metric}_baseline"]
    paired[["arm", "seed", *METRICS]].to_csv(output / "paired_deltas.csv", index=False)
    paired.groupby("arm")[list(METRICS)].agg(["mean", "std", "count"]).to_csv(
        output / "paired_summary.csv"
    )
    paired[paired.seed.ne(42)].groupby("arm")[list(METRICS)].agg(["mean", "std", "count"]).to_csv(
        output / "paired_summary_seeds43_46.csv"
    )
    cohorts = []
    for path in sorted((output / "evaluation").glob("*/*/trace/cohort_metrics.csv")):
        arm, seed = path.parts[-4:-2]
        cohorts.append(pd.read_csv(path).assign(arm=arm, seed=int(seed)))
    if cohorts:
        cohort = pd.concat(cohorts, ignore_index=True)
        cohort.to_csv(output / "cohort_per_seed.csv", index=False)
        cohort.groupby(["arm", "dimension", "cohort"])[list(METRICS)].agg(
            ["mean", "std", "count"]
        ).to_csv(output / "cohort_means.csv")
        delta = cohort[~cohort.arm.eq("baseline")].merge(
            cohort[cohort.arm.eq("baseline")],
            on=["seed", "dimension", "cohort"],
            suffixes=("", "_baseline"),
            validate="many_to_one",
        )
        for metric in METRICS:
            delta[metric] -= delta[f"{metric}_baseline"]
        delta.groupby(["arm", "dimension", "cohort"])[list(METRICS)].agg(
            ["mean", "std", "count"]
        ).to_csv(output / "cohort_paired_summary.csv")
    lines = [
        "# Deep feature study — big validation only",
        "",
        "- Capped MAE training; maximum validation user-XAUC checkpoint selection; 120 epochs.",
        "- Final scores use 1M global / 4096 pairs per user; same fixed pairs across arms.",
        "- Single-seed rows are exploratory. Five-seed results include screening seed42; inspect43–46 separately.",
        "- Training-seed SD is not dataset uncertainty. Profile/category availability is unverified.",
        "- No small-matrix access, no architecture or optimizer retuning.",
        "",
    ]
    for arm, group in frame.groupby("arm", sort=False):
        lines.append(
            f"- {arm} (n={len(group)}): " + "; ".join(f"{m} {group[m].mean():.6f}" for m in METRICS)
        )
        if arm != "baseline":
            d = paired[paired.arm.eq(arm)].user_xauc
            lines.append(
                f"  Paired user-XAUC {d.mean():+.6f}; positive in {int(d.gt(0).sum())}/{len(d)} seeds."
            )
    lines += [
        "",
        "## Review",
        "",
        "- per_seed.csv; paired_summary.csv; paired_summary_seeds43_46.csv.",
        "- cohort_means.csv; cohort_paired_summary.csv; each run's full history and trace.",
        "- Read MAE/global-XAUC and cohort regressions alongside the primary score; no automatic deployment.",
    ]
    (output / "review_summary.md").write_text("\n".join(lines) + "\n")
    plot_curves(output, records)


def plot_curves(output: Path, records: list[dict[str, Any]]) -> None:
    """Compare validation ranking and same-checkpoint gaps on shared axis scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reused = json.loads((output / "baseline_reuse.json").read_text())["runs"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    colors = dict(zip(ARMS, plt.get_cmap("tab10").colors))
    for record in records:
        arm, seed = record["arm"], record["seed"]
        run = Path(reused[str(seed)]) if arm == "baseline" else output / "runs" / f"{arm}_{seed}"
        history = json.loads((run / "history.json").read_text())
        epochs = [r["epoch"] + 1 for r in history]
        label = arm if seed == 42 else None
        style = {"color": colors[arm], "alpha": 0.85 if seed == 42 else 0.3, "label": label}
        axes[0].plot(epochs, [r["val"]["user_xauc"] for r in history], **style)
        axes[1].plot(epochs, [r["val"]["mae"] - r["train_eval"]["mae"] for r in history], **style)
    axes[0].set_ylabel("Validation capped user-XAUC (256 pairs/user)")
    axes[0].legend(ncol=2)
    axes[1].set(xlabel="Epoch", ylabel="Validation minus fixed-train capped MAE")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)


class FeatureStudy:
    """Own the frozen protocol, run ledger, and bounded confirmation decision."""

    def __init__(self, base: Path, raw: Path, baseline_study: Path, output: Path) -> None:
        self.base, self.raw, self.baseline_study, self.output = base, raw, baseline_study, output
        self.started = time.monotonic()
        self.records: list[dict[str, Any]] = []

    def status(self, stage: str, **details: Any) -> None:
        write_json(
            self.output / "status.json",
            {
                "stage": stage,
                "elapsed_seconds": time.monotonic() - self.started,
                "completed_new_runs": sum(r["arm"] != "baseline" for r in self.records),
                "test_accessed": False,
                **details,
            },
        )

    def run(self) -> None:
        import shutil

        if (self.output / "protocol.json").exists():
            raise FileExistsError("Preserve the existing study; use a fresh output directory")
        self.output.mkdir(parents=True, exist_ok=True)
        provenance = collect_run_provenance()
        protocol = {
            "question": "Which feature groups improve capped user-XAUC for fixed Deep[64,32]?",
            "arms": ARMS,
            "recipe": asdict(RECIPE),
            "epochs": 120,
            "seeds": list(range(42, 47)),
            "split_seed": 2026,
            "metric_seed": 0,
            "target_cap": 2,
            "weighting": "event",
            "selection": "maximum validation user-XAUC, earliest exact tie;256 pairs/user",
            "confirmation": "At most two largest strictly positive seed42 gains using4096 pairs/user; arm order breaks exact ties",
            "budget": "six screenings plus at most eight confirmation runs; no combinations or extensions",
            "guardrails": "Report MAE, global XAUC and cohorts; no post-hoc thresholds or automatic promotion",
            "test": "No test reads; historical small-matrix exposure remains disclosed",
            "baseline_study": str(self.baseline_study),
            "base_manifest_sha256": sha256(self.base / "meta.json"),
            "feature_importance": "Separate post-training analysis; does not choose candidates",
            "provenance": provenance,
        }
        write_json(self.output / "protocol.json", protocol)
        self.status("baseline_verification")
        self.records, baseline_paths = reuse_baselines(self.baseline_study, self.base)
        write_json(
            self.output / "baseline_reuse.json",
            {"records": self.records, "runs": {str(s): str(p) for s, p in baseline_paths.items()}},
        )
        for seed in baseline_paths:
            target = self.output / "evaluation/baseline" / str(seed) / "trace"
            target.mkdir(parents=True)
            shutil.copy2(
                self.baseline_study / f"evaluation/val/deep_{seed}/trace/cohort_metrics.csv",
                target / "cohort_metrics.csv",
            )
        self.status("preparing_data")
        data = self.output / "data"
        metadata = prepare_feature_data(self.base, self.raw, data)
        # Existing input encodings must remain identical, not merely have the same cardinality.
        original = json.loads((self.base / "meta.json").read_text())
        for name in BASE_FEATURES:
            if metadata["vocab"][name] != original["vocab"][name]:
                raise ValueError(f"Baseline encoding changed: {name}")
        provenance["feature_study_protocol_sha256"] = sha256(self.output / "protocol.json")

        def execute(arm: str, seed: int) -> None:
            self.status("training", arm=arm, seed=seed, epochs=120)
            run = self.output / "runs" / f"{arm}_{seed}"
            fit_arm(data, run, arm, seed, provenance)
            self.status("validation_scoring", arm=arm, seed=seed)
            self.records.append(
                score_arm(data, run, arm, self.output / "evaluation" / arm / str(seed))
            )
            summarize(self.output, self.records)
            gc.collect()

        for arm in ARMS:
            if arm != "baseline":
                execute(arm, 42)
        candidates = select_candidates(self.records)
        write_json(
            self.output / "confirmation_freeze.json",
            {"arms": candidates, "screening": self.records, "rule": protocol["confirmation"]},
        )
        for seed in range(43, 47):
            for arm in candidates:
                execute(arm, seed)
        for split in ("train", "val"):
            if sha256(data / f"{split}.parquet") != metadata["splits"][split]["sha256"]:
                raise ValueError("Prepared data changed during study")
        for name, digest in provenance["source_sha256"].items():
            if sha256(Path(name)) != digest:
                raise ValueError("Frozen source changed during study")
        self.status("complete", confirmed_arms=candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "raw", "baseline-study", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    study = FeatureStudy(
        args.base.resolve(),
        args.raw.resolve(),
        args.baseline_study.resolve(),
        args.output.resolve(),
    )
    existing = (study.output / "protocol.json").exists()
    try:
        study.run()
    except Exception as error:
        if study.output.exists() and not existing:
            study.status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
