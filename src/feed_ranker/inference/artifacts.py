"""Load versioned, self-contained inference artifacts without experiment datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from feed_ranker.artifacts import sha256
from feed_ranker.features.config import FeatureConfig
from feed_ranker.features.kuairec import KuaiRecFeatureBuilder
from feed_ranker.inference.rankers import ModelRanker
from feed_ranker.inference.retrievers import (
    CooccurrenceRetriever,
    MFRetriever,
    PopularityRetriever,
)
from feed_ranker.models.ranking_factory import ModelSpec, build_model
from feed_ranker.models.recall import BPRMF, CosineCooccurrence
from feed_ranker.recommendation.contracts import Retriever

VERSION = "recommendation-artifact-v1"


@dataclass(frozen=True)
class TrainingProvenance:
    source_sha256: str
    data_manifest_sha256: str
    seed: int | None = None
    epoch: int | None = None
    target_cap: float | None = None
    selection_metric: str | None = None


@dataclass(frozen=True)
class ArtifactManifest:
    version: str
    kind: str
    state_sha256: str
    provenance: TrainingProvenance

    @classmethod
    def read(cls, directory: Path) -> ArtifactManifest:
        value = json.loads((directory / "manifest.json").read_text())
        value["provenance"] = TrainingProvenance(**value["provenance"])
        manifest = cls(**value)
        if manifest.version != VERSION or manifest.kind not in (
            "ranker",
            "mf",
            "popularity",
            "cooccurrence",
        ):
            raise ValueError("Unsupported inference artifact version/kind")
        if sha256(directory / "state.pt") != manifest.state_sha256:
            raise ValueError("Inference artifact state checksum mismatch")
        return manifest


@dataclass(frozen=True)
class LoadedRanker:
    ranker: ModelRanker
    spec: ModelSpec
    features: KuaiRecFeatureBuilder
    manifest: ArtifactManifest


def _state(directory: Path) -> dict[str, Any]:
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)

    def validate(value: Any) -> None:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite inference state")
        if isinstance(value, dict):
            for child in value.values():
                validate(child)

    validate(state)
    return state


def _load_weights(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    for name, value in weights.items():
        if name in expected and value.dtype != expected[name].dtype:
            raise ValueError(f"Incompatible weight dtype: {name}")
    model.load_state_dict(weights, strict=True)


def load_ranker(directory: Path) -> LoadedRanker:
    manifest = ArtifactManifest.read(directory)
    if manifest.kind != "ranker":
        raise ValueError("Expected a ranking artifact")
    state = _state(directory)
    spec = ModelSpec(**state["model_spec"])
    features = KuaiRecFeatureBuilder(
        FeatureConfig.from_dict(state["feature_config"]),
        state["vocab"],
        pd.DataFrame(
            {n: v.numpy() for n, v in state["profiles"].items()},
            index=state["profile_users"].numpy(),
        ),
        pd.Series(state["durations"].numpy(), index=state["duration_items"].numpy()),
        state["crosses"],
    )
    # Initialization is overwritten by saved weights and must not perturb caller RNG.
    with torch.random.fork_rng(devices=[]):
        model = build_model(features.config, spec)
    _load_weights(model, state["model"])
    ranker = ModelRanker(model, features, sha256(directory / "manifest.json"))
    return LoadedRanker(ranker, spec, features, manifest)


def load_retriever(directory: Path) -> Retriever:
    manifest = ArtifactManifest.read(directory)
    if manifest.kind == "ranker":
        raise ValueError("Expected a retrieval artifact")
    state = _state(directory)
    artifact_id = sha256(directory / "manifest.json")
    items = state["items"].numpy()
    if manifest.kind == "popularity":
        return PopularityRetriever(items, state["counts"].numpy(), artifact_id)
    users = state["users"].numpy()
    if manifest.kind == "cooccurrence":
        model = CosineCooccurrence(state["incidence"], similarity=state["similarity"])
        return CooccurrenceRetriever(users, items, model, artifact_id)
    supported = state["supported_users"]
    if supported.dtype != torch.bool:
        raise ValueError("MF support mask must be boolean")
    with torch.random.fork_rng(devices=[]):
        model = BPRMF(len(users), len(items), state["dimension"])
    _load_weights(model, state["model"])
    return MFRetriever(model, users, items, supported.numpy(), artifact_id)
