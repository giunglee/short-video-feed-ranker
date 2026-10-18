"""Standalone architecture comparisons on capped KuaiRec event labels.

Linear and Wide use categorical coefficients, not a learned embedding followed by
a linear head. Deep and joint models share the same embedding/MLP implementation.
No retriever, candidate pool, first-event reduction, or funnel metric is involved.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.config import (
    CrossFeatureConfig,
    DataConfig,
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
    FeatureConfig,
    LabelConfig,
)
from feed_ranker.data.kuairec_dataset import KuaiRecDataset
from feed_ranker.experiments.diagnostics import export_validation_trace
from feed_ranker.models.components.feature_encoder import EmbeddingEncoder
from feed_ranker.models.wide_deep import DeepNN, WideDeepNN, WideNN
from feed_ranker.training.evaluation import RegressionEvaluator
from feed_ranker.training.losses import TargetMAE
from feed_ranker.training.optimizer import OptimizerConfig
from feed_ranker.training.trainer import TrainConfig, Trainer

ARCHITECTURES = ("linear", "wide", "deep", "wide_deep")
METRICS = ("mae", "xauc", "user_xauc")


@dataclass(frozen=True)
class Recipe:
    architecture: str
    hidden: tuple[int, ...] = ()
    lr: float = 0.001
    weight_decay: float = 0.0
    feature_set: str = "profiles"
    cross_family: str = "both"
    embedding_dim: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(self.hidden))
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Unknown architecture")
        if self.feature_set not in ("profiles", "ids_duration"):
            raise ValueError("Unknown feature set")
        if self.cross_family not in ("both", "age", "follow", "none"):
            raise ValueError("Unknown cross family")
        if self.architecture in ("deep", "wide_deep") and (
            not self.hidden or any(type(d) is not int or d <= 0 for d in self.hidden)
        ):
            raise ValueError("Deep models require positive hidden widths")
        if self.architecture in ("linear", "wide") and self.hidden:
            raise ValueError("Linear models have no hidden layers")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0:
            raise ValueError("Embedding width must be a positive integer")


def feature_config(metadata: dict[str, Any], recipe: Recipe) -> FeatureConfig:
    names = ["user_id", "video_id"]
    if recipe.feature_set == "profiles":
        names += ["age_range", "follow_user_num_range", "duration_band"]
    missing = set(names) - set(metadata["vocab"])
    if missing:
        raise ValueError(f"Prepared data lacks categorical features: {sorted(missing)}")
    crosses = []
    if recipe.architecture in ("wide", "wide_deep") and recipe.feature_set == "profiles":
        for name, conditions in metadata["spec"].get("crosses", {}).items():
            # Main effects are supplied by categorical coefficients, not duplicated
            # by the older one-hot columns stored in this prepared extract.
            if set(conditions) == {"age_range", "duration_band"}:
                family = "age"
            elif set(conditions) == {"follow_user_num_range", "duration_band"}:
                family = "follow"
            else:
                continue
            if recipe.cross_family in ("both", family):
                crosses.append(CrossFeatureConfig(name))
        if recipe.cross_family != "none" and not crosses:
            raise ValueError("Requested interaction family has no prepared columns")
    return FeatureConfig(
        dense_configs=[DenseFeatureConfig("video_duration", "log1p_nonnegative")],
        embedding_configs=[
            EmbeddingFeatureConfig(name, len(metadata["vocab"][name]) + 1, recipe.embedding_dim)
            for name in names
        ],
        cross_feature_configs=crosses,
    )


def build_architecture(features: FeatureConfig, recipe: Recipe, seed: int) -> nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dims = [features.deep_input_dims, *recipe.hidden, 1]
    if recipe.architecture in ("linear", "wide"):
        model = WideNN(features.wide_input_dims, 1, features, categorical_main_effects=True)
    elif recipe.architecture == "deep":
        model = DeepNN(features, dims, 0.0)
    else:
        model = WideDeepNN(features, dims, 0.0, categorical_main_effects=True)
    # Unseen categorical IDs have no estimated identity effect. Freeze their zero
    # vectors consistently in all architectures, instead of random OOV predictions.
    for module in model.modules():
        if isinstance(module, EmbeddingEncoder):
            for embedding in module.embeddings.values():
                embedding.padding_idx = 0
                with torch.no_grad():
                    embedding.weight[0].zero_()
    return model


def make_dataset(directory: Path, recipe: Recipe) -> KuaiRecDataset:
    import json

    metadata = json.loads((directory / "meta.json").read_text())
    return KuaiRecDataset(
        DataConfig(feature_config(metadata, recipe), LabelConfig("watch_ratio")), directory
    )


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
