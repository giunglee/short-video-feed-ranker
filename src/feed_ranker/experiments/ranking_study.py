"""Bounded, sequential ranking study: explore → freeze → confirm → test.

The predeclared search is deliberately small. A particular architecture ordering
is not a stopping condition. Small-matrix bytes are read only after all final
recipes and checkpoint hashes have been frozen from big validation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.experiments.ranking_architecture import (
    ARCHITECTURES,
    METRICS,
    Recipe,
    fit_recipe,
    score_checkpoint,
)


def initial_recipes() -> list[Recipe]:
    """Three screening trials per family; no validation-dependent grid expansion."""
    result = []
    for architecture in ARCHITECTURES:
        if architecture in ("linear", "wide"):
            result += [Recipe(architecture, lr=lr) for lr in (0.001, 0.003, 0.01)]
        else:
            result += [Recipe(architecture, hidden=h) for h in ((32,), (64, 32), (128, 32))]
    return result


def best_run(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest capped user-XAUC; an exact trial tie retains the first trial."""
    if not results or any(not np.isfinite(r["val"]["user_xauc"]) for r in results):
        raise ValueError("Selection requires finite validation user-XAUC")
    return max(results, key=lambda r: r["val"]["user_xauc"])


def needs_longer_budget(run: Path) -> bool:
    history = json.loads((run / "history.json").read_text())
    scores = np.array([row["val"]["user_xauc"] for row in history])
    if len(scores) < 20:
        return False
    return bool(
        np.argmax(scores) >= len(scores) - 10
        and scores[-10:].mean() - scores[-20:-10].mean() > 0.0002
    )


def verify_hashes(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def assert_finalists_unchanged(finalists: dict[str, Any]) -> None:
    for run in finalists["runs"]:
        if sha256(Path(run["run"]) / "best.pt") != run["checkpoint_sha256"]:
            raise ValueError("A frozen finalist checkpoint changed before final evaluation")


class RankingStudy:
    """Own the bounded study's run ledger, data identity and phase transitions."""

    def __init__(self, data: Path, output: Path, *, smoke: bool = False) -> None:
        self.data = data.resolve()
        self.output = output.resolve()
        self.smoke = smoke
        self.started = time.monotonic()
        self.results: list[dict[str, Any]] = []
        self.seeds = [42] if smoke else list(range(42, 47))
        self.epochs = 2 if smoke else 120
        self.provenance: dict[str, Any] = {}
        self.hashes: dict[str, str] = {}

    def status(self, stage: str, **details: Any) -> None:
        write_json(
            self.output / "status.json",
            {
                "stage": stage,
                "completed_training_runs": len(self.results),
                "elapsed_seconds": time.monotonic() - self.started,
                "smoke": self.smoke,
                **details,
            },
        )

    def prepare(self) -> None:
        if (self.output / "protocol.json").exists():
            raise FileExistsError("Study already started; preserve its protocol and results")
        self.output.mkdir(parents=True, exist_ok=True)
        metadata = json.loads((self.data / "meta.json").read_text())
        if metadata.get("preprocessing_version") != "signature-dedup-v2":
            raise ValueError("Expected signature-dedup-v2 prepared data")
        self.hashes[str(self.data / "meta.json")] = sha256(self.data / "meta.json")
        for split in ("train", "val"):
            path = self.data / f"{split}.parquet"
            actual = sha256(path)
            if actual != metadata["splits"][split]["sha256"]:
                raise ValueError(f"{split} manifest hash mismatch")
            self.hashes[str(path)] = actual
        self.provenance = collect_run_provenance()
        protocol = {
            "question": "Do explicit interactions and nonlinear representation improve capped within-user ranking?",
            "label_cap": 2.0,
            "loss": "event-weighted MAE on min(ratio, 2)",
            "primary": "capped user_xauc",
            "secondary": "capped xauc and capped mae",
            "checkpoint": "maximum validation user_xauc, earliest exact epoch tie",
            "trial_selection": "maximum validation user_xauc, earliest exact trial tie",
            "initial_recipes": [asdict(r) for r in initial_recipes()],
            "refinements": [
                "One matrix-weight-decay=1e-4 follow-up per family's best screening trial; existing optimizer exempts embeddings and biases",
                "Age-only and follow-only crosses for selected Wide and W&D",
                "Choose common cross family by mean validation user_xauc of Wide and W&D",
                "One IDs/duration-only W&D diagnostic; not eligible to replace matched-feature finalists",
            ],
            "budget": {
                "screening_trials": 21,
                "epochs": self.epochs,
                "extension": "If any finalist peaks in final10 and last10 mean exceeds preceding10 by >.0002, continue all4 to180 once",
                "confirmation": "Selected recipe per architecture, matched seeds42-46; reuse selected seed42",
                "maximum_full_training_runs": 37,
                "maximum_additional_60_epoch_continuations": 4,
            },
            "seeds": self.seeds,
            "metric_seed": 0,
            "split_seed": metadata["split_seed"],
            "metric_sampling": {
                "selection_global": 100000,
                "selection_per_user": 256,
                "final_global": 1000000,
                "final_per_user": 4096,
            },
            "feature_policy": "Same IDs, log-duration, age, follow-count and duration-band information; scalar wide main effects plus supported cross families",
            "availability": "Profile snapshot timing unverified; conditional modeling comparison, not serving-ready claim",
            "oov": "Fixed zero categorical coefficients and embedding vectors",
            "input_hashes": self.hashes,
            "provenance": self.provenance,
            "test": "Read only after finalist recipe/checkpoint freeze; no test-driven retuning",
            "prior_exposure": "Small-matrix labels were used in earlier EDA/training/evaluation; changed settings or deleted files do not reset exposure",
            "recall": "Outside study; no candidate artifacts or Recall results used",
            "seed_disclosure": "42 is used for screening; 43-46 have prior project history; include 43-46 sensitivity",
            "completion": "Analysis, matched-seed/cohort comparisons and frozen test report, regardless of architecture order",
            "smoke": self.smoke,
        }
        write_json(self.output / "protocol.json", protocol)
        self.provenance["study_protocol_sha256"] = sha256(self.output / "protocol.json")

    def fit(
        self, name: str, recipe: Recipe, seed: int = 42, resume: Path | None = None
    ) -> dict[str, Any]:
        self.status("training", active_run=name, seed=seed, epochs=self.epochs)
        result = fit_recipe(
            self.data,
            self.output / name,
            recipe,
            seed,
            self.epochs,
            self.provenance,
            resume_from=resume,
            steps=2 if self.smoke else None,
        )
        self.results.append(result)
        write_json(self.output / "completed_runs.json", self.results)
        return result

    def explore(self) -> dict[str, dict[str, Any]]:
        families: dict[str, list[dict[str, Any]]] = {a: [] for a in ARCHITECTURES}
        for index, recipe in enumerate(initial_recipes()):
            families[recipe.architecture].append(self.fit(f"screen_{index:02d}", recipe))
        selected = {}
        for architecture, runs in families.items():
            original = best_run(runs)
            recipe = replace(Recipe(**original["recipe"]), weight_decay=0.0001)
            runs.append(self.fit(f"regularized_{architecture}", recipe))
            selected[architecture] = best_run(runs)

        cross_runs = {"both": {a: selected[a] for a in ("wide", "wide_deep")}}
        for family in ("age", "follow"):
            cross_runs[family] = {}
            for architecture in ("wide", "wide_deep"):
                recipe = replace(Recipe(**selected[architecture]["recipe"]), cross_family=family)
                cross_runs[family][architecture] = self.fit(
                    f"cross_{family}_{architecture}", recipe
                )
        chosen = max(
            cross_runs,
            key=lambda family: np.mean(
                [r["val"]["user_xauc"] for r in cross_runs[family].values()]
            ),
        )
        selected.update(cross_runs[chosen])
        diagnostic = replace(
            Recipe(**selected["wide_deep"]["recipe"]),
            feature_set="ids_duration",
            cross_family="none",
        )
        self.fit("ids_duration_reference", diagnostic)
        write_json(
            self.output / "exploration_decisions.json",
            {
                "common_cross_family": chosen,
                "cross_scores": {
                    f: {a: r["val"] for a, r in rs.items()} for f, rs in cross_runs.items()
                },
                "selected": selected,
                "limitation": "Sequential bounded tuning, not an exhaustive factorial search; architecture-specific knobs have different budgets",
            },
        )
        return selected

    def confirm(self, selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.smoke and any(needs_longer_budget(Path(r["run"])) for r in selected.values()):
            self.epochs = 180
            selected = {
                a: self.fit(f"extended_{a}", Recipe(**r["recipe"]), resume=Path(r["run"]))
                for a, r in selected.items()
            }
        write_json(
            self.output / "recipe_freeze.json",
            {
                "epochs": self.epochs,
                "selected": selected,
                "reason": "All choices made from big validation before matched-seed confirmation",
            },
        )
        runs = list(selected.values())
        for seed in self.seeds[1:]:
            for architecture in ARCHITECTURES:
                runs.append(
                    self.fit(
                        f"confirm_{architecture}_{seed}",
                        Recipe(**selected[architecture]["recipe"]),
                        seed,
                    )
                )
        return runs

    def freeze_and_evaluate(self, runs: list[dict[str, Any]]) -> None:
        verify_hashes(self.hashes)
        verify_hashes(self.provenance["source_sha256"])
        finalists = {
            "frozen_before_test": True,
            "recipe_freeze_sha256": sha256(self.output / "recipe_freeze.json"),
            "runs": [{**r, "checkpoint_sha256": sha256(Path(r["run"]) / "best.pt")} for r in runs],
        }
        write_json(self.output / "finalists.json", finalists)
        records = []
        for split in ("val", "test"):
            self.status("evaluation", split=split, test_evaluated=split == "test")
            assert_finalists_unchanged(finalists)
            # This is the first access to small-matrix file bytes in the study.
            if split == "test":
                meta = json.loads((self.data / "meta.json").read_text())
                path = self.data / "test.parquet"
                expected = meta["splits"]["test"]["sha256"]
                if sha256(path) != expected:
                    raise ValueError("Test data differs from prepared manifest")
                self.hashes[str(path)] = expected
            for result in runs:
                run = Path(result["run"])
                name = f"{result['recipe']['architecture']}_{result['seed']}"
                score = score_checkpoint(
                    self.data,
                    run,
                    self.output / "evaluation" / split / name,
                    split,
                    export_trace=not self.smoke,
                )
                records.append(
                    {
                        "split": split,
                        "architecture": result["recipe"]["architecture"],
                        "seed": result["seed"],
                        "epoch": result["best_epoch"] + 1,
                        "parameters": result["parameters"],
                        **score["metrics"],
                    }
                )
            pd.DataFrame(records).to_csv(self.output / "per_seed.csv", index=False)
        verify_hashes(self.hashes)
        verify_hashes(self.provenance["source_sha256"])
        assert_finalists_unchanged(finalists)
        summarize(self.output, runs, records)
        self.status("complete", test_evaluated=True, finalist_runs=len(runs))

    def run(self) -> None:
        self.prepare()
        self.status("preflight", test_evaluated=False)
        selected = self.explore()
        verify_hashes(self.hashes)
        verify_hashes(self.provenance["source_sha256"])
        self.freeze_and_evaluate(self.confirm(selected))


def summarize(output: Path, runs: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    """Aggregate matched seeds; keep sampling sensitivity separate from selection."""
    frame = pd.DataFrame(records)
    summary = frame.groupby(["split", "architecture"])[list(METRICS)].agg(["mean", "std"])
    summary.to_csv(output / "means.csv")
    deltas = []
    for variant, baseline in (
        ("wide", "linear"),
        ("deep", "wide"),
        ("wide_deep", "deep"),
        ("wide_deep", "wide"),
        ("deep", "linear"),
    ):
        merged = frame[frame.architecture.eq(variant)].merge(
            frame[frame.architecture.eq(baseline)],
            on=["split", "seed"],
            suffixes=("_variant", "_baseline"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            deltas.append(
                {
                    "split": row["split"],
                    "seed": row.seed,
                    "contrast": f"{variant} - {baseline}",
                    **{m: row[f"{m}_variant"] - row[f"{m}_baseline"] for m in METRICS},
                }
            )
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(output / "paired_deltas.csv", index=False)
    delta_frame.groupby(["split", "contrast"])[list(METRICS)].agg(["mean", "std"]).to_csv(
        output / "paired_summary.csv"
    )
    delta_frame[delta_frame.seed.ne(42)].groupby(["split", "contrast"])[list(METRICS)].agg(
        ["mean", "std"]
    ).to_csv(output / "paired_summary_seeds43_46.csv")
    cohorts = []
    for path in sorted((output / "evaluation").glob("*/*/trace/cohort_metrics.csv")):
        split, name = path.parts[-4:-2]
        architecture, seed = name.rsplit("_", 1)
        cohorts.append(
            pd.read_csv(path).assign(split=split, architecture=architecture, seed=int(seed))
        )
    if cohorts:
        cohort = pd.concat(cohorts, ignore_index=True)
        cohort.to_csv(output / "cohort_per_seed.csv", index=False)
        cohort.groupby(["split", "architecture", "dimension", "cohort"])[list(METRICS)].agg(
            ["mean", "std"]
        ).to_csv(output / "cohort_means.csv")
        cohort_deltas = []
        for variant, baseline in (("wide", "linear"), ("deep", "wide"), ("wide_deep", "deep")):
            joined = cohort[cohort.architecture.eq(variant)].merge(
                cohort[cohort.architecture.eq(baseline)],
                on=["split", "seed", "dimension", "cohort"],
                suffixes=("_v", "_b"),
                validate="one_to_one",
            )
            values = joined[["split", "seed", "dimension", "cohort"]].copy()
            values["contrast"] = f"{variant} - {baseline}"
            for metric in METRICS:
                values[metric] = joined[f"{metric}_v"] - joined[f"{metric}_b"]
            cohort_deltas.append(values)
        pd.concat(cohort_deltas).groupby(["split", "contrast", "dimension", "cohort"])[
            list(METRICS)
        ].agg(["mean", "std"]).to_csv(output / "cohort_paired_summary.csv")
    plot_curves(output, runs)
    lines = [
        "# Standalone capped-label ranking: architecture review",
        "",
        "- Question: do explicit crosses and nonlinear representations improve within-user ranking?",
        "- Target: min(watch_ratio, 2). All event rows retained; MAE training, no prediction clipping.",
        "- Checkpoint: maximum capped validation user-XAUC, earliest tie; all metrics share it.",
        "- Same feature information across four families; Wide has categorical main effects and crosses.",
        "- Profile snapshot timing is unverified. This is a conditional modeling result.",
        "- Test was accessed only after recipes and all checkpoint hashes were frozen.",
        "- Earlier small-matrix EDA/training/evaluation exposure remains disclosed.",
        "- Seed42 participated in tuning; see seeds43–46 sensitivity. All seeds have prior project history.",
        "- Final metrics use 1M global / 4096 pairs per user; selection used 100K / 256.",
        "- Recall and funnel evaluation are outside this experiment.",
        "",
        "## Results",
        "",
    ]
    for (split, architecture), group in frame.groupby(["split", "architecture"]):
        metrics = "; ".join(f"{m} {group[m].mean():.6f} ± {group[m].std():.6f}" for m in METRICS)
        lines.append(f"- {split} / {architecture}: {metrics} (sample SD).")
    lines += ["", "## Matched-seed evidence", ""]
    for (split, contrast), group in delta_frame.groupby(["split", "contrast"]):
        values = group.user_xauc
        lines.append(
            f"- {split}, {contrast}: capped user-XAUC difference "
            f"{values.mean():+.6f} ± {values.std():.6f}; "
            f"positive in {int(values.gt(0).sum())}/{len(values)} seeds."
        )
    lines += ["", "## Training diagnostics", ""]
    for architecture in ARCHITECTURES:
        selected_runs = [r for r in runs if r["recipe"]["architecture"] == architecture]
        gaps, epochs, late = [], [], 0
        for result in selected_runs:
            history = json.loads((Path(result["run"]) / "history.json").read_text())
            row = history[result["best_epoch"]]
            gaps.append(row["val"]["mae"] - row["train_eval"]["mae"])
            epochs.append(result["best_epoch"] + 1)
            late += int(result["best_epoch"] >= len(history) - 10)
        lines.append(
            f"- {architecture}: selected epochs {epochs}; "
            f"mean validation minus fixed-train capped MAE {np.mean(gaps):+.6f}; "
            f"{late}/{len(selected_runs)} selections in the final10 epochs."
        )
    if cohorts:
        lines += ["", "## Cohort inspection: joint minus Deep", ""]
        eligible = cohort[cohort.rows.ge(1000) & cohort.users.ge(100)]
        means = eligible.groupby(["split", "architecture", "dimension", "cohort"])[
            "user_xauc"
        ].mean()
        for split in ("val", "test"):
            joint = means.loc[(split, "wide_deep")]
            deep = means.loc[(split, "deep")]
            differences = (joint - deep).dropna().sort_values()
            for (dimension, name), value in pd.concat(
                [differences.head(2), differences.tail(2)]
            ).items():
                lines.append(
                    f"- {split}, {dimension} / {name}: mean user-XAUC difference {value:+.6f}."
                )
        lines.append(
            "- These are descriptive cohort extremes with ≥1,000 rows and ≥100 users, not multiplicity-adjusted discoveries."
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- Higher XAUC is better; lower MAE is better. Different metric orderings are possible.",
        "- Paired differences measure changes within the same seed, not differences against seed42 alone.",
        "- Training-seed SD does not measure dataset uncertainty or remove repeated-validation selection bias.",
        "- The tuning search was bounded and sequential, not exhaustive or equal-FLOP optimization.",
        "- A model trained to minimize capped MAE is not directly optimizing pairwise ranking or top-K utility.",
        "- Do not tune again based on test results. User interpretation and publication remain review steps.",
        "",
        "## Evidence",
        "",
        "- protocol.json; exploration_decisions.json; recipe_freeze.json; finalists.json.",
        "- per_seed.csv; means.csv; paired_deltas.csv; paired_summary.csv; paired_summary_seeds43_46.csv.",
        "- cohort_means.csv; cohort_paired_summary.csv; individual traces under evaluation/.",
        "- learning_curves.png; each run's history.json, config.json, best.pt and last.pt.",
    ]
    (output / "review_summary.md").write_text("\n".join(lines) + "\n")


def plot_curves(output: Path, runs: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, architecture in zip(axes.flat, ARCHITECTURES):
        for run in runs:
            if run["recipe"]["architecture"] != architecture:
                continue
            history = json.loads((Path(run["run"]) / "history.json").read_text())
            ax.plot(
                [r["epoch"] + 1 for r in history],
                [r["val"]["user_xauc"] for r in history],
                label=str(run["seed"]),
                alpha=0.7,
            )
        ax.set(title=architecture, xlabel="Epoch", ylabel="Validation capped user-XAUC")
        ax.legend(title="Seed")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--smoke", action="store_true", help="Synthetic fixtures only; two steps/epoch"
    )
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    study = RankingStudy(args.data, args.output, smoke=args.smoke)
    already_started = (study.output / "protocol.json").exists()
    try:
        study.run()
    except Exception as error:
        if study.output.exists() and not already_started:
            study.status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
