"""Write fitted model and lookup state as self-contained inference artifacts."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.features.kuairec import KuaiRecFeatureBuilder
from feed_ranker.inference.artifacts import VERSION, ArtifactManifest, TrainingProvenance
from feed_ranker.inference.retrievers import CooccurrenceRetriever, MFRetriever, PopularityRetriever
from feed_ranker.models.ranking_factory import ModelSpec


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


def export_ranker(
    directory: Path,
    model: nn.Module,
    spec: ModelSpec,
    features: KuaiRecFeatureBuilder,
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


def export_retriever(
    directory: Path,
    retriever: PopularityRetriever | CooccurrenceRetriever | MFRetriever,
    provenance: TrainingProvenance,
) -> None:
    """Serialize fitted state; no training dataset or historical run is inspected."""
    state: dict[str, Any] = {"items": torch.from_numpy(retriever.items.to_numpy(copy=True))}
    if isinstance(retriever, PopularityRetriever):
        state["counts"] = torch.from_numpy(retriever.counts.copy())
    elif isinstance(retriever, CooccurrenceRetriever):
        state.update(
            users=torch.from_numpy(retriever.users.to_numpy(copy=True)),
            incidence=retriever.model.incidence,
            similarity=retriever.model.similarity,
        )
    elif isinstance(retriever, MFRetriever):
        state.update(
            users=torch.from_numpy(retriever.users.to_numpy(copy=True)),
            supported_users=torch.from_numpy(retriever.supported.copy()),
            dimension=retriever.model.user.embedding_dim,
            model={n: v.detach().cpu() for n, v in retriever.model.state_dict().items()},
        )
    else:
        raise TypeError("Unsupported fitted retriever")
    _write(directory, retriever.name, state, provenance)
