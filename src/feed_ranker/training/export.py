"""Offline export of fitted models and lookup state into inference-only artifacts.

Legacy conversion preserves saved embedding row order. It never refits a model,
selects a checkpoint, or reads validation outcomes to build feature lookups.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.config import FeatureConfig
from feed_ranker.data.recall import RecallData
from feed_ranker.data.recommendation import KuaiRecFeatures
from feed_ranker.models.ranking import ModelSpec, build_model
from feed_ranker.models.recall import CosineCooccurrence
from feed_ranker.recommendation.artifacts import VERSION, ArtifactManifest, TrainingProvenance
from feed_ranker.training.ranking import Recipe, feature_config
from feed_ranker.training.recall import incidence


def _write(
    directory: Path, kind: str, state: dict[str, Any], provenance: TrainingProvenance
) -> None:
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        torch.save(state, staging / "state.pt")
        manifest = ArtifactManifest(VERSION, kind, sha256(staging / "state.pt"), provenance)
        write_json(staging / "manifest.json", asdict(manifest))
        staging.rename(directory)


def fit_candidate_features(
    config: FeatureConfig, metadata: dict[str, Any], training_path: Path, duration_path: Path
) -> KuaiRecFeatures:
    """Extract only declared user attributes from training and frozen item durations."""
    crosses = {n: metadata["spec"]["crosses"][n] for n in config.cross_feature_cols}
    profile_names = sorted(
        (set(config.embedding_cols) | {n for c in crosses.values() for n in c})
        - {"user_id", "video_id", "duration_band"}
    )
    columns = ["raw_user_id", *profile_names]
    parts = [
        b.to_pandas().drop_duplicates()
        for b in pq.ParquetFile(training_path).iter_batches(batch_size=131072, columns=columns)
    ]
    profiles = pd.concat(parts, ignore_index=True).drop_duplicates()
    if profiles.raw_user_id.duplicated().any() or profiles.isna().any().any():
        raise ValueError("Conflicting or missing training profile codes")
    profiles = profiles.set_index("raw_user_id")
    durations = pd.read_parquet(duration_path).set_index("video_id")["median"]
    names = set(config.embedding_cols) | set(profile_names)
    return KuaiRecFeatures(
        config, {n: metadata["vocab"][n] for n in names}, profiles, durations, crosses
    )


def export_ranker(
    directory: Path,
    model: nn.Module,
    spec: ModelSpec,
    features: KuaiRecFeatures,
    provenance: TrainingProvenance,
) -> None:
    """Export resolved inputs with their model; no experiment preset is interpreted."""
    _write(
        directory,
        "ranker",
        {
            "model_spec": asdict(spec),
            "feature_config": asdict(features.config),
            "model": {n: v.detach().cpu() for n, v in model.state_dict().items()},
            "vocab": features.vocab,
            "crosses": features.crosses,
            "profile_users": torch.from_numpy(features.profiles.index.to_numpy(dtype=np.int64)),
            "profiles": {
                n: torch.from_numpy(features.profiles[n].to_numpy(dtype=np.int64))
                for n in features.profile_names
            },
            "duration_items": torch.from_numpy(features.durations.index.to_numpy(dtype=np.int64)),
            "durations": torch.from_numpy(features.durations.to_numpy(dtype=np.float32)),
        },
        provenance,
    )


def export_checkpoint(
    checkpoint: Path, ranking_data: Path, duration_path: Path, directory: Path
) -> None:
    """Convert architecture-study checkpoints; new checkpoints carry resolved specs."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    if sha256(ranking_data / "meta.json") != meta["data_manifest_sha256"]:
        raise ValueError("Frozen ranking data manifest mismatch")
    metadata = json.loads((ranking_data / "meta.json").read_text())
    if sha256(ranking_data / "train.parquet") != metadata["splits"]["train"]["sha256"]:
        raise ValueError("Prepared ranking training data changed")
    if "feature_config" in meta and "model_spec" in meta:
        config, spec = (
            FeatureConfig.from_dict(meta["feature_config"]),
            ModelSpec(**meta["model_spec"]),
        )
    else:
        # Explicit migration of the historical architecture-study format only.
        # Feature-study artifacts need their resolved config, not a guessed preset.
        if "features" in meta:
            raise ValueError(
                "Legacy feature-study checkpoint needs resolved feature_config/model_spec"
            )
        recipe = Recipe(**meta["recipe"])
        config, spec = (
            feature_config(metadata, recipe),
            ModelSpec(recipe.architecture, recipe.hidden),
        )
    features = fit_candidate_features(
        config, metadata, ranking_data / "train.parquet", duration_path
    )
    with torch.random.fork_rng(devices=[]):
        model = build_model(config, spec)
    model.load_state_dict(saved["model"], strict=True)
    provenance = TrainingProvenance(
        sha256(checkpoint),
        meta["data_manifest_sha256"],
        saved["training"]["seed"],
        saved["epoch"],
        meta.get("target_cap"),
        saved["training"]["selection_metric"],
    )
    export_ranker(directory, model, spec, features, provenance)


def export_retriever(data: RecallData, run: Path, directory: Path) -> None:
    """Convert an existing recall run, preserving its exact IDs and support policy."""
    config = json.loads((run / "config.json").read_text())
    data_hash = sha256(data.directory / "meta.json")
    if config["data_manifest_sha256"] != data_hash:
        raise ValueError("Retriever data mismatch")
    kind = config["method"]
    state: dict[str, Any] = {"items": torch.from_numpy(data.items.copy())}
    source_hash = sha256(run / "config.json")
    epoch = None
    if kind == "popularity":
        state["counts"] = torch.from_numpy(
            np.bincount(np.concatenate(data.training.positive), minlength=len(data.items)).astype(
                np.float32
            )
        )
    elif kind == "cooccurrence":
        model = CosineCooccurrence(incidence(data.training))
        state.update(
            users=torch.from_numpy(data.users.copy()),
            incidence=model.incidence,
            similarity=model.similarity,
        )
    elif kind == "mf":
        saved = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        if saved["config"] != config:
            raise ValueError("MF checkpoint/config mismatch")
        state.update(
            users=torch.from_numpy(data.users.copy()),
            supported_users=torch.from_numpy(data.training.supported.copy()),
            dimension=config["dimension"],
            model=saved["model"],
        )
        source_hash, epoch = sha256(run / "best.pt"), saved.get("epoch")
    else:
        raise ValueError("Unknown retriever")
    _write(
        directory,
        kind,
        state,
        TrainingProvenance(source_hash, data_hash, config.get("seed"), epoch),
    )
