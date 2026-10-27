"""BPR optimization for the temporal binary bridge, reusing the observed sampler.

BPR: Rendle et al., https://arxiv.org/abs/1205.2618.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.artifacts import write_json
from feed_ranker.data.kuairand import EventDataset
from feed_ranker.data.recall import RecallTrainingState
from feed_ranker.evaluation.binary import BinaryEvaluator
from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures
from feed_ranker.models.legacy_kuairand import export_ranker
from feed_ranker.models.ranking_factory import ModelSpec, build_model
from feed_ranker.models.recall import BPRMF
from feed_ranker.training.optimizer import OptimizerConfig, build_optimizer
from feed_ranker.training.recall import ObservedSampler
from feed_ranker.training.trainer import TrainConfig, Trainer


def fit_mf(
    positive: list[np.ndarray],
    negative: list[np.ndarray],
    encoding: Encoding,
    *,
    epochs: int,
    seed: int,
    out: Path,
) -> tuple[BPRMF, np.ndarray, list[dict[str, Any]]]:
    supported = np.array([bool(len(p) and len(n)) for p, n in zip(positive, negative, strict=True)])
    # Existing observed sampler consumes these histories; no legacy manifest spoofing.
    sampler = ObservedSampler(
        RecallTrainingState(
            encoding.users,
            encoding.items,
            positive,
            negative,
            [np.empty(0, dtype=np.int64) for _ in positive],
            [np.union1d(p, n) for p, n in zip(positive, negative, strict=True)],
            supported,
        )
    )
    torch.manual_seed(seed)
    model = BPRMF(len(encoding.users), len(encoding.items), 32)
    optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3))
    rng = np.random.default_rng(seed)
    curve = []
    for epoch in range(epochs):
        users, pos, neg = sampler.sample(rng)
        total = 0.0
        for start in range(0, len(users), 4096):
            u, p, n = (
                users[start : start + 4096],
                pos[start : start + 4096],
                neg[start : start + 4096],
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-model(u, p, n)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite BPR loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            model.item_updated[p] = True
            model.item_updated[n] = True
            total += float(loss.detach()) * len(u)
        curve.append({"epoch": epoch, "train_loss": total / len(users), "pairs": len(users)})
        print(f"mf epoch={epoch} loss={curve[-1]['train_loss']:.6f}", flush=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs - 1,
            "policy": "fixed final epoch; no validation selection",
            "seed": seed,
            "dimension": 32,
        },
        out / "mf.pt",
    )
    write_json(out / "mf_history.json", curve)
    restored = BPRMF(len(encoding.users), len(encoding.items), 32)
    restored.load_state_dict(torch.load(out / "mf.pt", weights_only=True)["model"])
    with torch.no_grad():
        torch.testing.assert_close(
            model(users[:64], pos[:64], neg[:64]),
            restored(users[:64], pos[:64], neg[:64]),
            rtol=0,
            atol=0,
        )
    return model.eval(), supported, curve


def fit_legacy_deep(
    train: pd.DataFrame,
    val: pd.DataFrame,
    encoding: Encoding,
    features: KuaiRandFeatures,
    label: str,
    hidden: list[int],
    config: TrainConfig,
    directory: Path,
    manifest_hash: str,
    metadata_hash: str | None,
    *,
    resume_from: Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Run the existing v1 study inputs through the shared Trainer lifecycle.

    New callers use training.job.train_job with versioned prepared data instead.
    """
    train_data, val_data = [
        EventDataset(f, encoding, label, features.feature_set, features.metadata)
        for f in (train, val)
    ]
    torch.manual_seed(config.seed)
    model = build_model(features.config, ModelSpec("deep", tuple(hidden)))
    trainer = Trainer(
        model,
        lambda epoch: train_data.batches(config.batch_size, config.seed + epoch),
        config,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        evaluator=BinaryEvaluator(lambda: val_data.batches(config.eval_batch_size)),
    )
    result = trainer.fit(
        directory,
        {
            "data_manifest_sha256": manifest_hash,
            "model_name": "DeepNN",
            "hidden": hidden,
            "dropout": 0.0,
            "use_ids": features.feature_set not in ("no_user", "no_item"),
            "feature_set": features.feature_set,
            "feature_config": asdict(features.config),
            "side_features_sha256": metadata_hash if features.metadata is not None else None,
        },
        resume_from=resume_from,
    )
    export_ranker(
        directory / "inference.pt",
        model,
        encoding,
        features.feature_set,
        hidden,
        manifest_hash,
        features.metadata,
    )
    return model, result
