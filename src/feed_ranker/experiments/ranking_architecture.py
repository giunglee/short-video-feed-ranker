"""Standalone architecture comparisons on capped KuaiRec event labels.

Linear and Wide use categorical coefficients, not a learned embedding followed by
a linear head. Deep and joint models share the same embedding/MLP implementation.
No retriever, candidate pool, first-event reduction, or funnel metric is involved.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.kuairec_dataset import KuaiRecDataset
from feed_ranker.experiments.diagnostics import export_validation_trace
from feed_ranker.models.ranking import ModelSpec
from feed_ranker.training.evaluation import RegressionEvaluator
from feed_ranker.training.losses import TargetMAE
from feed_ranker.training.optimizer import OptimizerConfig
from feed_ranker.training.ranking import Recipe, build_architecture, make_dataset
from feed_ranker.training.trainer import TrainConfig, Trainer

METRICS = ("mae", "xauc", "user_xauc")


def evaluator(
    dataset: KuaiRecDataset,
    splits: tuple[str, ...],
    *,
    detailed: bool = False,
    train_diagnostic: bool = False,
) -> RegressionEvaluator:
    loaders = {split: dataset.loader(split, 8192) for split in splits}
    if train_diagnostic:
        loaders["train_diagnostic"] = dataset.fixed_train_loader(100_000, 2026, 8192)
    return RegressionEvaluator(
        loaders,
        METRICS,
        "cpu",
        0,
        1_000_000 if detailed else 100_000,
        4096 if detailed else 256,
        label_cap=2.0,
    )


def fit_recipe(
    data: Path,
    output: Path,
    recipe: Recipe,
    seed: int,
    epochs: int,
    provenance: dict[str, Any],
    *,
    resume_from: Path | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    """Read train/validation only; every epoch is a full pass unless smoke testing."""
    started = time.monotonic()
    dataset = make_dataset(data, recipe)
    model = build_architecture(dataset.config.feature, recipe, seed)
    config = TrainConfig(
        epochs=epochs,
        seed=seed,
        batch_size=1024,
        eval_batch_size=8192,
        train_eval_rows=100_000,
        steps=steps,
        selection_metric="user_xauc",
        selection_mode="max",
        optimizer=OptimizerConfig(lr=recipe.lr, weight_decay=recipe.weight_decay),
    )
    metadata = {
        "recipe": asdict(recipe),
        "model_spec": asdict(ModelSpec(recipe.architecture, recipe.hidden)),
        "feature_config": asdict(dataset.config.feature),
        "target_cap": 2.0,
        "metric_labels": "min(raw_watch_ratio, 2); predictions are not clipped",
        "model_name": recipe.architecture,
        "hidden": list(recipe.hidden),
        "dropout": 0.0,
        "use_ids": True,
        "data_manifest_sha256": sha256(data / "meta.json"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "cross_columns": dataset.config.feature.cross_feature_cols,
        **provenance,
    }
    if resume_from is not None:
        saved = torch.load(resume_from / "last.pt", weights_only=True)
        if Recipe(**saved["metadata"]["recipe"]) != recipe:
            raise ValueError("Continuation cannot change the architecture recipe")
    trainer = Trainer(
        model,
        lambda epoch: dataset.loader("train", config.batch_size, seed + epoch),
        config,
        loss_fn=TargetMAE(2.0),
        evaluator=evaluator(dataset, ("val",), train_diagnostic=True),
    )
    result = trainer.fit(output, metadata, resume_from=resume_from)
    result.update(
        recipe=asdict(recipe),
        parameters=metadata["parameters"],
        fit_seconds=time.monotonic() - started,
        epochs=epochs,
        run=str(output),
        exploratory=True,
        steps_truncated=steps is not None,
    )
    write_json(output / "result.json", result)
    return result


def score_checkpoint(
    data: Path,
    run: Path,
    output: Path,
    split: str,
    *,
    export_trace: bool = True,
) -> dict[str, Any]:
    """Score an already selected checkpoint; never change its selected epoch."""
    if split not in ("val", "test") or output.exists():
        raise ValueError("Use a fresh scoring directory and val/test split")
    saved = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    if sha256(data / "meta.json") != saved["metadata"]["data_manifest_sha256"]:
        raise ValueError("Checkpoint/data manifest mismatch")
    recipe = Recipe(**saved["metadata"]["recipe"])
    dataset = make_dataset(data, recipe)
    model = build_architecture(dataset.config.feature, recipe, saved["training"]["seed"])
    model.load_state_dict(saved["model"])
    output.mkdir(parents=True)
    result = {
        "split": split,
        "target_cap": 2.0,
        "checkpoint_sha256": sha256(run / "best.pt"),
        "seed": saved["training"]["seed"],
        "epoch": saved["epoch"],
        "recipe": asdict(recipe),
        "metrics": evaluator(dataset, (split,), detailed=True).evaluate(model, split),
        "global_pairs": 1_000_000,
        "pairs_per_user": 4096,
    }
    if export_trace:
        export_validation_trace(
            model,
            dataset,
            run / "best.pt",
            output / "trace",
            feature_allowlist=["video_duration"],
            split=split,
            label_cap=2.0,
        )
    write_json(output / "metrics.json", result)
    return result
