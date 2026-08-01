"""
Wide & Deep KuaiRec M1 — watch_ratio regression (MAE / L1).

Ablations: predict_mean → linear_reg → wide_nn → deep_nn → wide_deep_nn.

Run from repo root:
  python -m experiments.wide_deep_kuairec --model predict_mean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import (
    DataConfig,
    DenseFeatureConfig,
    EmbeddingFeatureConfig,
    FeatureConfig,
    KuaiRecDataset,
    LabelConfig,
    CrossFeatureConfig,
)
from data.kuairec_preprocessor import WIDE_CROSS_SPECS
from models import DeepNN, LinearRegression, WideDeepNN, WideNN
from training import OptimizerConfig, TrainConfig, Trainer

LABEL = "watch_ratio"
EMB_DIM = 8
ID_DIM = 16

# Not used as inputs: play_duration, watch_ratio (label),
# video_daily_play_duration / play_progress (soft proxies), date / timestamp.

WIDE_CROSS_COLS = [spec["name"] for spec in WIDE_CROSS_SPECS]

# Duplicate/constant columns dropped per data/kuairec_eda_findings.md
# (Cramer's V + bijective crosstab check): is_live_streamer (== is_live_author)
# and user_onehot_feat{0,1,2,3,5,6,7,8,9,10,11} (anonymized re-encodings of
# gender/age_range/phone_*/fre_*/platform, kept the human-readable column
# instead). is_video_author/is_photo_author are bijective in this sample too
# but semantically distinct, so both are kept pending a check on a larger
# dataset.
CAT_COLS = [
    "gender",
    "age_range",
    "user_active_degree",
    "is_video_author",
    "is_live_author",
    "is_photo_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "user_onehot_feat4",
    "user_onehot_feat12",
    "user_onehot_feat13",
    "user_onehot_feat14",
    "user_onehot_feat15",
    "user_onehot_feat16",
    "user_onehot_feat17",
    "phone_brand",
    "phone_model",
    "mod_price",
    "fre_country",
    "fre_country_region",
    "fre_province",
    "fre_city",
    "fre_city_level",
    "fre_community_type",
    "platform",
    "os_version",
    "app_version",
    "app_download_channel",
    "isp",
    "upload_dt",
    "upload_type",
    "visible_status",
    # video_type, is_lowactive_period dropped: constant (nunique == 1).
]

ID_COLS = [
    ("user_id", ID_DIM),
    ("video_id", ID_DIM),
    ("author_id", ID_DIM),
    ("music_id", EMB_DIM),
    ("video_tag_id", EMB_DIM),
]

DENSE_COLS = [
    "video_duration",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    "video_width",
    "video_height",
    "video_daily_show_cnt",
    "video_daily_show_user_num",
    "video_daily_play_cnt",
    "video_daily_play_user_num",
    "video_daily_complete_play_cnt",
    "video_daily_complete_play_user_num",
    "video_daily_valid_play_cnt",
    "video_daily_valid_play_user_num",
    "video_daily_long_time_play_cnt",
    "video_daily_long_time_play_user_num",
    "video_daily_short_time_play_cnt",
    "video_daily_short_time_play_user_num",
    "video_daily_comment_stay_duration",
    "video_daily_like_cnt",
    "video_daily_like_user_num",
    "video_daily_click_like_cnt",
    "video_daily_double_click_cnt",
    "video_daily_cancel_like_cnt",
    "video_daily_cancel_like_user_num",
    "video_daily_comment_cnt",
    "video_daily_comment_user_num",
    "video_daily_direct_comment_cnt",
    "video_daily_reply_comment_cnt",
    "video_daily_delete_comment_cnt",
    "video_daily_delete_comment_user_num",
    "video_daily_comment_like_cnt",
    "video_daily_comment_like_user_num",
    "video_daily_follow_cnt",
    "video_daily_follow_user_num",
    "video_daily_cancel_follow_cnt",
    "video_daily_cancel_follow_user_num",
    "video_daily_share_cnt",
    "video_daily_share_user_num",
    "video_daily_download_cnt",
    "video_daily_download_user_num",
    # report_cnt / reduce_similar_cnt were previously (incorrectly) excluded
    # as "high NaN" -- that was an artifact of the date-parsing bug in
    # KuaiRecPreprocessor._extract (fixed; see data/kuairec_eda_findings.md).
    # Both are 0% null post-fix, re-added.
    "video_daily_report_cnt",
    "video_daily_report_user_num",
    "video_daily_reduce_similar_cnt",
    "video_daily_reduce_similar_user_num",
    # video_daily_collect_cnt / cancel_collect_{cnt,user_num} / collect_user_num
    # dropped: KuaiRec started tracking "collect" (bookmark) actions partway
    # through the collection window (2020-07-05 - 2020-07-27 = 100% null in
    # item_daily_features.csv), so they're ~49% null in train (which covers
    # that period) vs 0% in val/test -- a real dataset characteristic, not an
    # ETL bug. Revisit with an explicit missing-indicator if needed later.
]


def _dense() -> list[DenseFeatureConfig]:
    return [DenseFeatureConfig(name=c, normalize=True) for c in DENSE_COLS]


def _embed_cats() -> list[EmbeddingFeatureConfig]:
    """IDs + categoricals as embeddings (linear_reg / deep_nn)."""
    ids = [
        EmbeddingFeatureConfig(name=n, embedding_dim=d) for n, d in ID_COLS
    ]
    cats = [
        EmbeddingFeatureConfig(name=c, embedding_dim=EMB_DIM) for c in CAT_COLS
    ]
    return ids + cats


def _cross_features() -> list[CrossFeatureConfig]:
    """Wide-path binary columns (materialized in KuaiRecPreprocessor)."""
    return [CrossFeatureConfig(name=c) for c in WIDE_CROSS_COLS]


def _dataset(feature: FeatureConfig) -> tuple[KuaiRecDataset, FeatureConfig]:
    ds = KuaiRecDataset(config=DataConfig(feature=feature, label=LabelConfig(name=LABEL)))
    feature.apply_vocab(ds.vocab_size())
    return ds, feature


def _trainer(
    model: nn.Module,
    dataset: KuaiRecDataset,
    weight_decay: float = 0.0,
    epochs: int = 15,
    steps: int | None = None,
    compute_xauc: bool = True,
) -> Trainer:
    cfg = TrainConfig(
        batch_size=1024,
        epochs=epochs,
        # None = one full shuffled pass over train per epoch (see
        # KuaiRecDataset.epoch_batches); pass an int to cap steps/epoch for
        # a quick smoke test instead.
        steps=steps,
        optimizer_config=OptimizerConfig(lr=1e-3, weight_decay=weight_decay),
        compute_auc=False,
        compute_xauc=compute_xauc,
        primary_metric="mae",
        show_plots=False,
    )
    return Trainer(cfg, model, dataset, nn.L1Loss())


def predict_mean() -> dict[str, float]:
    """Train-label mean baseline; val/test MAE only."""
    ds = KuaiRecDataset(config=DataConfig(label=LabelConfig(name=LABEL)))
    train, val, test = (ds.get_full(s) for s in ("train", "val", "test"))
    mu = train["label"].float().mean()
    l1 = nn.L1Loss()
    return {
        "model": "predict_mean",
        "train_label_mean": float(mu.item()),
        "val_mae": float(l1(mu.expand_as(val["label"]), val["label"].float()).item()),
        "test_mae": float(l1(mu.expand_as(test["label"]), test["label"].float()).item()),
    }


def linear_reg(
    weight_decay: float = 0.0,
    epochs: int = 15,
    steps: int | None = None,
    compute_xauc: bool = True,
) -> tuple:
    feat = FeatureConfig(embedding_configs=_embed_cats(), dense_configs=_dense())
    ds, feat = _dataset(feat)
    model = LinearRegression(feat.feature_dims, 1, feat)
    trainer = _trainer(model, ds, weight_decay, epochs, steps, compute_xauc).build()
    return trainer.fit(), trainer


def wide_nn(
    weight_decay: float = 0.0,
    epochs: int = 15,
    steps: int | None = None,
    compute_xauc: bool = True,
) -> tuple:
    feat = FeatureConfig(cross_feature_configs=_cross_features(), dense_configs=_dense())
    ds, feat = _dataset(feat)
    model = WideNN(feat.wide_input_dims, 1, feat)
    trainer = _trainer(model, ds, weight_decay, epochs, steps, compute_xauc).build()
    return trainer.fit(), trainer


DEEP_NN_DIMS_DEFAULT = [128, 32, 8, 1]
WIDE_DEEP_NN_DIMS_DEFAULT = [256, 64, 8, 1]


def deep_nn(
    hidden_dims: list[int] | None = None,
    dropout_ratio: float = 0.0,
    weight_decay: float = 0.0,
    epochs: int = 15,
    steps: int | None = None,
    compute_xauc: bool = True,
) -> tuple:
    feat = FeatureConfig(embedding_configs=_embed_cats(), dense_configs=_dense())
    ds, feat = _dataset(feat)
    dims = [feat.deep_input_dims, *(hidden_dims or DEEP_NN_DIMS_DEFAULT)]
    model = DeepNN(feat, dims, dropout_ratio)
    trainer = _trainer(model, ds, weight_decay, epochs, steps, compute_xauc).build()
    return trainer.fit(), trainer


def wide_deep_nn(
    hidden_dims: list[int] | None = None,
    dropout_ratio: float = 0.0,
    weight_decay: float = 0.0,
    epochs: int = 15,
    steps: int | None = None,
    compute_xauc: bool = True,
) -> tuple:
    # wide: crosses + dense; deep: all cat emb + dense
    feat = FeatureConfig(
        embedding_configs=_embed_cats(),
        cross_feature_configs=_cross_features(),
        dense_configs=_dense(),
    )
    ds, feat = _dataset(feat)
    dims = [feat.deep_input_dims, *(hidden_dims or WIDE_DEEP_NN_DIMS_DEFAULT)]
    model = WideDeepNN(feat, dims, dropout_ratio)
    trainer = _trainer(model, ds, weight_decay, epochs, steps, compute_xauc).build()
    return trainer.fit(), trainer


def run_experiment(
    model_name: str,
    hidden_dims: list[int] | None = None,
    dropout_ratio: float = 0.0,
    weight_decay: float = 0.0,
    epochs: int = 15,
    steps: int | None = None,
    compute_xauc: bool = True,
    seed: int = 42,
) -> dict:
    baseline = predict_mean()
    if model_name == "predict_mean":
        return {
            **baseline,
            "seed": seed,
            "beats_baseline_val": True,
            "beats_baseline_test": True,
        }

    # dropout_ratio only applies to deep_nn/wide_deep_nn's MLP hidden layers
    # -- linear_reg/wide_nn are single Linear layers with no hidden units,
    # so input dropout there would just be a noisier version of
    # weight_decay (dropout on a linear model is first-order equivalent to
    # L2; Wager, Wang & Liang 2013), not a distinct regularizer. weight_decay
    # /epochs/steps/compute_xauc apply uniformly to all 4 so ablations stay
    # apples-to-apples.
    is_mlp = model_name in ("deep_nn", "wide_deep_nn")
    runners = {
        "linear_reg": lambda: linear_reg(weight_decay, epochs, steps, compute_xauc),
        "wide_nn": lambda: wide_nn(weight_decay, epochs, steps, compute_xauc),
        "deep_nn": lambda: deep_nn(hidden_dims, dropout_ratio, weight_decay, epochs, steps, compute_xauc),
        "wide_deep_nn": lambda: wide_deep_nn(hidden_dims, dropout_ratio, weight_decay, epochs, steps, compute_xauc),
    }
    if model_name not in runners:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {sorted(['predict_mean', *runners])}")

    fit, trainer = runners[model_name]()
    val, test = trainer.evaluate_split("val"), trainer.evaluate_split("test")
    arch_dims = [
        trainer.model.config.deep_input_dims,  # type: ignore[union-attr]
        *(hidden_dims or (DEEP_NN_DIMS_DEFAULT if model_name == "deep_nn" else WIDE_DEEP_NN_DIMS_DEFAULT)),
    ] if is_mlp else None
    steps_per_epoch_used = steps if steps is not None else trainer.data_loader.steps_per_epoch(
        trainer.config.batch_size, split="train"
    )
    return {
        "model": model_name,
        "label": LABEL,
        "loss": "L1Loss",
        "seed": seed,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch_used,
        "nn_dims": arch_dims,
        "dropout_ratio": dropout_ratio if is_mlp else None,
        "weight_decay": weight_decay,
        "baseline_val_mae": baseline["val_mae"],
        "baseline_test_mae": baseline["test_mae"],
        "train_label_mean": baseline["train_label_mean"],
        "final_val_loss": fit.final_val_loss,
        "final_val_mae": fit.final_val_mae,
        "best_val_mae": fit.best_val_mae,
        "best_epoch": fit.best_epoch,
        "restored_best_weights": fit.restored_best_weights,
        "val_xauc": fit.best_epoch_xauc,
        "test_loss": test.loss,
        "test_mae": test.mae,
        "test_xauc": test.xauc,
        # Selection uses best_val_mae (the restored checkpoint), not
        # final_val_mae, so a config can't win just by landing on a lucky
        # last epoch. test_* is logged for monitoring only, never used to
        # pick a config -- see docs/milestones.md for the full rationale.
        "beats_baseline_val": fit.best_val_mae < baseline["val_mae"],
        "beats_baseline_test": test.mae < baseline["test_mae"],
        "final_grad_norm": fit.grad_norms[-1],
        "final_weight_norm": fit.weight_norms[-1],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wide & Deep KuaiRec M1")
    p.add_argument(
        "--model",
        choices=["predict_mean", "linear_reg", "wide_nn", "deep_nn", "wide_deep_nn"],
        required=True,
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--nn-dims",
        type=str,
        default=None,
        help="Comma-separated MLP hidden dims after the input layer, e.g. "
        "'128,32,1' (must end in 1). deep_nn/wide_deep_nn only; ignored otherwise.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Hidden-layer dropout ratio for deep_nn/wide_deep_nn's MLP. "
        "Ignored for linear_reg/wide_nn (single Linear, no hidden units to "
        "regularize between -- use --weight-decay for those instead).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight_decay, applied to all 4 models.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for torch.manual_seed + np.random.seed (weight init, "
        "dropout masks, and train-batch shuffling all derive from this). "
        "Vary this across runs of the same config to estimate noise/variance "
        "before trusting a small val MAE gap between two configs.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of full shuffled passes over train (see --steps to override).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Batches per epoch. Default (None) = one full pass over train "
        "(len(train)//batch_size, ~3511 at batch_size=1024). Set lower only "
        "for a quick smoke test -- it will not cover all of train.",
    )
    p.add_argument(
        "--compute-xauc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sampled pairwise concordance metric (see training/metrics.py). "
        "Cheap; on by default. Use --no-compute-xauc to skip.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hidden_dims = [int(d) for d in args.nn_dims.split(",")] if args.nn_dims else None
    summary = run_experiment(
        args.model,
        hidden_dims=hidden_dims,
        dropout_ratio=args.dropout,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        steps=args.steps,
        compute_xauc=args.compute_xauc,
        seed=args.seed,
    )
    print(f"RESULT_JSON: {json.dumps(summary, sort_keys=True)}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
