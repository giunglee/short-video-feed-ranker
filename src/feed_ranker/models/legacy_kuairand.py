"""Bounded readers for current KuaiRand v1/v2 rankers and recall bundles.

These are migration inputs, not a promise of arbitrary historical compatibility.
Artifact reconstruction returns fitted state; recommendation owns adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from feed_ranker.data.metadata import MetadataEncoding
from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures
from feed_ranker.models.artifacts import ModelArtifact, _validate_ids, load_weights
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence


@dataclass
class LegacyRanker:
    model: torch.nn.Module
    features: KuaiRandFeatures
    identity: str


def load_legacy_ranker(path: Path) -> LegacyRanker:
    """Load fitted tensors and recipe without raw data or training-code imports."""
    import torch

    from feed_ranker.artifacts import sha256
    from feed_ranker.data.metadata import MetadataEncoding
    from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures
    from feed_ranker.models.ranking_factory import ModelSpec, build_model

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] not in ("kuairand-ranker-v1", "kuairand-ranker-v2"):
        raise ValueError("Unsupported KuaiRand ranker artifact")
    encoding = Encoding(*(state[name].numpy() for name in ["users", "items", "duration"]))
    metadata = (
        MetadataEncoding.from_state(state["metadata"])
        if state["version"] == "kuairand-ranker-v2"
        else None
    )
    features = KuaiRandFeatures(encoding, state["feature_set"], metadata)
    with torch.random.fork_rng(devices=[]):
        model = build_model(features.config, ModelSpec("deep", tuple(state["hidden"])))
    load_weights(model, state["model"])
    return LegacyRanker(model, features, sha256(path))


def load_legacy_recall(directory: Path) -> list[ModelArtifact]:
    """Reconstruct fitted recall channels with no event tables or evaluation labels."""
    import json

    import torch

    from feed_ranker.artifacts import sha256
    from feed_ranker.models.recall import BPRMF

    manifest = json.loads((directory / "retrieval.json").read_text())
    for name in ["recall_state.npz", "mf.pt"]:
        if sha256(directory / name) != manifest["files"][name]:
            raise ValueError(f"Changed fitted recall artifact: {name}")
    with np.load(directory / "recall_state.npz", allow_pickle=False) as state:
        users, items = state["users"], state["items"]
        offsets, ids = state["positive_offsets"], state["positive_items"]
        supported = state["supported"]
    positive = [ids[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    cosine = SparseCosineCooccurrence(positive, len(items))
    checkpoint = torch.load(directory / "mf.pt", map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        mf = BPRMF(len(users), len(items), checkpoint["dimension"])
    load_weights(mf, checkpoint["model"])
    _validate_ids(users, "user")
    _validate_ids(items, "item")
    if supported.dtype != np.bool_ or supported.shape != users.shape:
        raise ValueError("Invalid legacy MF support mask")
    identity = manifest["data_manifest_sha256"]
    return [
        ModelArtifact(
            "popularity", cosine.counts, identity + ":popularity", {}, users=users, items=items
        ),
        ModelArtifact("cosine", cosine, identity + ":cosine-exact", {}, users=users, items=items),
        ModelArtifact(
            "mf",
            mf,
            sha256(directory / "mf.pt"),
            {},
            users=users,
            items=items,
            supported_users=supported,
        ),
    ]


def export_ranker(
    path: Path,
    model: "torch.nn.Module",
    encoding: Encoding,
    feature_set: str,
    hidden: list[int],
    manifest_sha256: str,
    metadata: MetadataEncoding | None = None,
) -> None:
    """Self-contained label-free handoff to ModelRanker, with fitted-state provenance."""
    if path.exists():
        raise FileExistsError(path)
    torch.save(
        {
            "version": "kuairand-ranker-v2" if metadata is not None else "kuairand-ranker-v1",
            "metadata": metadata.to_state() if metadata is not None else None,
            "model": model.state_dict(),
            "users": torch.from_numpy(encoding.users),
            "items": torch.from_numpy(encoding.items),
            "duration": torch.from_numpy(encoding.duration),
            "feature_set": feature_set,
            "hidden": hidden,
            "data_manifest_sha256": manifest_sha256,
        },
        path,
    )
