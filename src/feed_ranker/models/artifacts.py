"""Self-contained fitted scorer artifacts; no training or recommendation imports."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.prepared import FeatureState
from feed_ranker.models.deep import DeepModel, ModelConfig, build_model
from feed_ranker.models.recall import BPRMF
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence

VERSION = "fitted-scorer-v1"
KINDS = ("deep", "mf", "popularity", "cosine")


@dataclass
class ModelArtifact:
    kind: str
    model: DeepModel | BPRMF | SparseCosineCooccurrence | np.ndarray
    identity: str
    provenance: dict[str, Any]
    config: ModelConfig | None = None
    feature_state: FeatureState | None = None
    users: np.ndarray | None = None
    items: np.ndarray | None = None
    supported_users: np.ndarray | None = None


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: v.detach().cpu() for n, v in model.state_dict().items()}


def load_weights(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    if set(weights) != set(expected):
        raise ValueError("Artifact parameter names disagree with configuration")
    for name, value in weights.items():
        if (
            value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid artifact parameter shape/dtype/value: {name}")
    model.load_state_dict(weights, strict=True)
    model.eval()


def _validate_ids(values: np.ndarray, name: str) -> None:
    if (
        values.dtype != np.int64
        or values.ndim != 1
        or not len(values)
        or (values < 0).any()
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError(f"Invalid {name} vocabulary")


def save_artifact(path: Path, artifact: ModelArtifact) -> None:
    """Validate in staging, then publish a complete immutable artifact directory."""
    if path.exists():
        raise FileExistsError(path)
    if artifact.kind not in KINDS:
        raise ValueError("Unsupported fitted scorer kind")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        manifest: dict[str, Any] = {
            "version": VERSION,
            "kind": artifact.kind,
            "provenance": artifact.provenance,
        }
        model = artifact.model
        if artifact.kind == "deep":
            if (
                not isinstance(model, DeepModel)
                or artifact.config != model.config
                or artifact.feature_state is None
            ):
                raise ValueError("Deep artifact requires matching model/config/feature state")
            if model.schema != {n: artifact.feature_state.schema[n] for n in model.config.features}:
                raise ValueError("Model and fitted feature meanings disagree")
            artifact.feature_state.save(staging)
            torch.save(_weights(model), staging / "weights.pt")
            manifest["model_config"] = asdict(model.config)
            manifest["output_schema"] = {
                model.config.output: {"dtype": "float32", "shape": ["B"], "meaning": "binary logit"}
            }
        else:
            if artifact.users is None or artifact.items is None:
                raise ValueError("Recall artifacts require raw user/item mappings")
            _validate_ids(artifact.users, "user")
            _validate_ids(artifact.items, "item")
            arrays = {"users": artifact.users, "items": artifact.items}
            if artifact.kind == "mf":
                if not isinstance(model, BPRMF) or artifact.supported_users is None:
                    raise ValueError("MF artifact requires model and support mask")
                arrays["supported_users"] = artifact.supported_users
                manifest["dimension"] = model.user.embedding_dim
                torch.save(_weights(model), staging / "weights.pt")
            elif artifact.kind == "cosine":
                if not isinstance(model, SparseCosineCooccurrence):
                    raise ValueError("Cosine artifact requires fitted sparse state")
                arrays["positive_items"] = np.concatenate(model.positive)
                arrays["positive_offsets"] = np.r_[
                    0, np.cumsum([len(v) for v in model.positive])
                ].astype(np.int64)
            else:
                if not isinstance(model, np.ndarray):
                    raise ValueError("Popularity artifact requires fitted count array")
                arrays["counts"] = model
            np.savez(staging / "recall-state.npz", **arrays)
        manifest["files"] = {p.name: sha256(p) for p in staging.iterdir()}
        write_json(staging / "manifest.json", manifest)
        load_artifact(staging)
        staging.rename(path)


def load_artifact(path: Path) -> ModelArtifact:
    manifest = json.loads((path / "manifest.json").read_text())
    kind = manifest.get("kind")
    if manifest.get("version") != VERSION or kind not in KINDS:
        raise ValueError("Unsupported fitted scorer version/kind")
    required = (
        {"feature-state.npz", "feature-schema.json", "weights.pt"}
        if kind == "deep"
        else {"recall-state.npz"} | ({"weights.pt"} if kind == "mf" else set())
    )
    if set(manifest["files"]) != required:
        raise ValueError("Artifact file inventory disagrees with kind")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    artifact = ModelArtifact(
        kind, np.empty(0), sha256(path / "manifest.json"), manifest["provenance"]
    )
    with torch.random.fork_rng(devices=[]):
        if kind == "deep":
            artifact.feature_state = FeatureState.load(path)
            artifact.config = ModelConfig(**manifest["model_config"])
            if manifest["output_schema"] != {
                artifact.config.output: {
                    "dtype": "float32",
                    "shape": ["B"],
                    "meaning": "binary logit",
                }
            }:
                raise ValueError("Output schema/meaning disagrees with model")
            artifact.model = build_model(artifact.config, artifact.feature_state.schema)
        else:
            with np.load(path / "recall-state.npz", allow_pickle=False) as state:
                artifact.users, artifact.items = state["users"], state["items"]
                _validate_ids(artifact.users, "user")
                _validate_ids(artifact.items, "item")
                if kind == "mf":
                    mask = state["supported_users"]
                    if mask.dtype != np.bool_ or mask.shape != artifact.users.shape:
                        raise ValueError("Invalid MF supported-user mask")
                    artifact.supported_users = mask
                    dimension = manifest["dimension"]
                    if type(dimension) is not int or dimension < 1:
                        raise ValueError("Invalid MF dimension")
                    artifact.model = BPRMF(len(artifact.users), len(artifact.items), dimension)
                elif kind == "cosine":
                    offsets, ids = state["positive_offsets"], state["positive_items"]
                    if (
                        offsets.dtype != np.int64
                        or ids.dtype != np.int64
                        or offsets.shape != (len(artifact.users) + 1,)
                        or ids.ndim != 1
                        or offsets[0] != 0
                        or offsets[-1] != len(ids)
                        or (np.diff(offsets) < 0).any()
                    ):
                        raise ValueError("Invalid sparse history offsets")
                    artifact.model = SparseCosineCooccurrence(
                        [ids[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)],
                        len(artifact.items),
                    )
                else:
                    counts = state["counts"]
                    if (
                        counts.shape != artifact.items.shape
                        or not np.isfinite(counts).all()
                        or (counts < 0).any()
                    ):
                        raise ValueError("Invalid popularity counts")
                    artifact.model = counts
        if kind in ("deep", "mf"):
            load_weights(
                artifact.model,
                torch.load(path / "weights.pt", map_location="cpu", weights_only=True),
            )
    return artifact
