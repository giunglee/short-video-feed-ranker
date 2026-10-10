"""Big-matrix train/validation runs and explicitly invoked small-matrix evaluation.

Feature selection and checkpoint selection use validation only. Every run records
its data manifest, source fingerprints, configuration, and training seed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from feed_ranker.artifacts import collect_run_provenance, sha256, write_json
from feed_ranker.data.config import (
    DataConfig,
    LabelConfig,
    resolve_feature_config,
)
from feed_ranker.data.kuairec_dataset import KuaiRecDataset
from feed_ranker.models import ConstantRegressor, DeepNN, LinearRegression, WideDeepNN, WideNN
from feed_ranker.training import OptimizerConfig, TrainConfig, Trainer
from feed_ranker.training.evaluation import METRICS, RegressionEvaluator

MODELS = ("deep_nn", "wide_deep_nn", "wide_nn", "linear_reg")


def run_experiment(
    directory: Path,
    output: Path,
    model_name: str,
    config: TrainConfig,
    hidden: list[int],
    dropout: float = 0.0,
    use_ids: bool = True,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory: {output}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model, dataset = build_model(directory, model_name, hidden, dropout, use_ids)
    verify_data(dataset, ("train", "val"))
    baselines = _fit_baselines(dataset, config.device)
    metadata = {
        "baseline_predictions": {
            name: baseline.prediction.item() for name, baseline in baselines.items()
        },
        "model_name": model_name,
        "hidden": hidden,
        "dropout": dropout,
        "use_ids": use_ids,
        "data_manifest_sha256": sha256(directory / "meta.json"),
        "protocol": dataset.meta["protocol"],
        **collect_run_provenance(),
    }
    trainer = build_trainer(model, dataset, config)
    result = trainer.fit(output, metadata)
    result.update(
        {
            "model": model_name,
            "training": asdict(config),
            "metadata": metadata,
            "baselines_val": {
                name: trainer.evaluator.evaluate(baseline, "val")
                for name, baseline in baselines.items()
            },
        }
    )
    write_json(output / "result.json", result)
    return result


def evaluate_checkpoint(
    directory: Path, checkpoint: Path, output: Path, device: str = "cpu"
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Final evaluation already exists; do not overwrite it")
    trainer, saved = _restore_for_test(directory, checkpoint, device)
    metadata = saved["metadata"]
    result = {
        "checkpoint_sha256": sha256(checkpoint),
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "model": metadata["model_name"],
        "seed": trainer.config.seed,
        "epoch": saved["epoch"],
        "exploratory": trainer.config.steps is not None,
        "test": trainer.evaluator.evaluate(trainer.model, "test"),
        "baselines_test": {
            name: trainer.evaluator.evaluate(ConstantRegressor(value).to(device), "test")
            for name, value in metadata["baseline_predictions"].items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def build_model(
    directory: Path, name: str, hidden: list[int], dropout: float, use_ids: bool = True
) -> tuple[nn.Module, KuaiRecDataset]:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name!r}")
    if not hidden or any(dimension <= 0 for dimension in hidden):
        raise ValueError("hidden must contain positive layer widths")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    metadata = json.loads((directory / "meta.json").read_text())
    features = resolve_feature_config(metadata, name, use_ids)
    dataset = KuaiRecDataset(DataConfig(features, LabelConfig("watch_ratio")), directory)
    dimensions = [features.deep_input_dims, *hidden, 1]
    if name == "deep_nn":
        model = DeepNN(features, dimensions, dropout)
    elif name == "wide_deep_nn":
        model = WideDeepNN(features, dimensions, dropout)
    elif name == "wide_nn":
        model = WideNN(features.wide_input_dims, 1, features)
    else:
        model = LinearRegression(features.deep_input_dims, 1, features)
    return model, dataset


def verify_data(dataset: KuaiRecDataset, splits: tuple[str, ...]) -> None:
    for split in splits:
        expected = dataset.meta["splits"][split]["sha256"]
        actual = sha256(dataset.directory / f"{split}.parquet")
        if actual != expected:
            raise ValueError(f"{split} changed since preprocessing")


def _fit_baselines(dataset: KuaiRecDataset, device: str) -> dict[str, ConstantRegressor]:
    # Baselines fit the same prepared labels as the ranker, never validation/test.
    frame = pd.read_parquet(dataset.directory / "train.parquet", columns=["watch_ratio"])
    labels = torch.from_numpy(frame["watch_ratio"].to_numpy(copy=True))
    return {name: ConstantRegressor.fit(labels, name).to(device) for name in ("mean", "median")}


def _restore_for_test(
    directory: Path, checkpoint: Path, device: str
) -> tuple[Trainer, dict[str, Any]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if sha256(directory / "meta.json") != metadata["data_manifest_sha256"]:
        raise ValueError("Checkpoint and data manifest differ")
    for filename, expected in metadata["source_sha256"].items():
        source = Path(filename)
        if not source.exists() or sha256(source) != expected:
            raise ValueError(
                f"Training source changed: {filename}; restore it before final evaluation"
            )

    model, dataset = build_model(
        directory,
        metadata["model_name"],
        metadata["hidden"],
        metadata["dropout"],
        metadata["use_ids"],
    )
    verify_data(dataset, ("test",))
    model.load_state_dict(saved["model"])
    training = dict(saved["training"])
    training["optimizer"] = OptimizerConfig(**training["optimizer"])
    training["device"] = device
    return build_trainer(model, dataset, TrainConfig(**training)), saved


def build_trainer(model: nn.Module, dataset: KuaiRecDataset, config: TrainConfig) -> Trainer:
    if config.weighting == "pair" and not dataset.has_pair_weights:
        raise ValueError("Pair weighting requires signature-dedup-v2 prepared data")
    losses = {"mae": nn.L1Loss, "mse": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}
    evaluator = RegressionEvaluator(
        loaders={split: dataset.loader(split, config.eval_batch_size) for split in ("val", "test")},
        metrics=config.metrics,
        device=config.device,
        metric_seed=config.metric_seed,
        xauc_pairs=config.xauc_pairs,
        user_pairs=config.user_pairs,
    )
    return Trainer(
        model,
        lambda epoch: dataset.loader("train", config.batch_size, config.seed + epoch),
        config,
        loss_fn=losses[config.loss](reduction="none" if config.weighting == "pair" else "mean"),
        evaluator=evaluator,
    )


# CLI options shared by single-run and multi-seed commands.


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 32])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, help="Exploratory smoke run only")
    parser.add_argument("--no-ids", action="store_true")
    parser.add_argument("--weighting", choices=("event", "pair"), default="event")
    parser.add_argument("--loss", choices=("mae", "mse", "smooth_l1"), default="mae")
    parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=["mae", "xauc", "user_xauc"]
    )
    parser.add_argument("--selection-metric", choices=METRICS, default="mae")
    parser.add_argument("--selection-mode", choices=("min", "max"), default="min")


def config_from_args(args: argparse.Namespace, seed: int) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        device=args.device,
        steps=args.steps,
        optimizer=OptimizerConfig(lr=args.lr, weight_decay=args.weight_decay),
        loss=args.loss,
        weighting=getattr(args, "weighting", "event"),
        metrics=tuple(args.metrics),
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    add_train_args(train)
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        result = run_experiment(
            directory=args.data,
            output=args.output,
            model_name=args.model,
            config=config_from_args(args, args.seed),
            hidden=args.hidden,
            dropout=args.dropout,
            use_ids=not args.no_ids,
        )
        report = {"best_epoch": result["best_epoch"], "val": result["val"]}
    else:
        report = evaluate_checkpoint(args.data, args.checkpoint, args.output, args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
