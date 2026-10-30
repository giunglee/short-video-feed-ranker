"""Self-contained fitted scorer artifacts; no training or recommendation imports."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import numpy as np
import torch
from torch import nn

from feed_ranker.models.ranking import DeepConfig, DeepModel
from feed_ranker.models.recall import BPRMF, SparseCosineCooccurrence
from feed_ranker.persistence import sha256, write_json
from feed_ranker.preprocessing.features import FeatureState

VERSION = "fitted-scorer-v5"
KINDS = ("deep", "mf", "popularity", "cosine")


@dataclass
class RankerArtifact:
    model: DeepModel
    feature_state: FeatureState
    identity: str
    provenance: dict[str, Any]
    kind: ClassVar[str] = "deep"

    def __post_init__(self) -> None:
        _validate_ranker(self.model, self.feature_state)


@dataclass
class MFArtifact:
    model: BPRMF
    users: np.ndarray
    items: np.ndarray
    supported_users: np.ndarray
    identity: str
    provenance: dict[str, Any]
    kind: ClassVar[str] = "mf"


@dataclass
class CosineArtifact:
    model: SparseCosineCooccurrence
    users: np.ndarray
    items: np.ndarray
    identity: str
    provenance: dict[str, Any]
    kind: ClassVar[str] = "cosine"


@dataclass
class PopularityArtifact:
    model: np.ndarray
    users: np.ndarray
    items: np.ndarray
    identity: str
    provenance: dict[str, Any]
    kind: ClassVar[str] = "popularity"


ModelArtifact: TypeAlias = RankerArtifact | MFArtifact | CosineArtifact | PopularityArtifact


def _validate_ranker(model: DeepModel, features: FeatureState) -> None:
    config = model.config
    if (
        set(config.feature_names) - set(features.feature_names)
        or any(features.vocab_sizes.get(n) != size for n, size in config.sparse.items())
        or any(n in features.vocab_sizes for n in config.dense)
    ):
        raise ValueError("Model inputs and fitted feature tables disagree")


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    weights = {n: v.detach().cpu() for n, v in model.state_dict().items()}
    for name, value in weights.items():
        expected_dtype = (
            torch.bool if isinstance(model, BPRMF) and name == "item_updated" else torch.float32
        )
        if value.dtype != expected_dtype:
            raise ValueError(f"Artifact weights require {expected_dtype}: {name}")
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Cannot export nonfinite model weights")
    return weights


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


@contextmanager
def _model_directory(
    path: Path, kind: str, config: dict[str, Any], provenance: dict[str, Any]
) -> Iterator[Path]:
    """Publish only a complete directory; serialization never reconstructs a model."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        write_json(staging / "config.json", config)
        yield staging
        manifest = {
            "version": VERSION,
            "kind": kind,
            "provenance": provenance,
            "files": {
                str(p.relative_to(staging)): sha256(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        }
        if kind == "deep":
            manifest["output_schema"] = {
                config["output"]: {"dtype": "float32", "shape": ["B"], "meaning": "binary logit"}
            }
        write_json(staging / "manifest.json", manifest)
        staging.rename(path)


def save_ranker(
    path: Path, model: DeepModel, features: FeatureState, provenance: dict[str, Any]
) -> None:
    _validate_ranker(model, features)
    weights = _weights(model)
    with _model_directory(path, "deep", model.config.to_dict(), provenance) as output:
        torch.save(weights, output / "weights.pt")
        (output / "features").mkdir()
        features.save(output / "features")


def save_mf(
    path: Path,
    model: BPRMF,
    users: np.ndarray,
    items: np.ndarray,
    supported_users: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    _validate_ids(users, "user")
    _validate_ids(items, "item")
    if model.user.embedding_dim < 1:
        raise ValueError("Invalid MF dimension")
    if (
        model.user.num_embeddings != len(users)
        or model.item.num_embeddings != len(items)
        or model.user.embedding_dim != model.item.embedding_dim
    ):
        raise ValueError("MF model and vocabulary sizes disagree")
    if supported_users.dtype != np.bool_ or supported_users.shape != users.shape:
        raise ValueError("Invalid MF supported-user mask")
    weights = _weights(model)
    with _model_directory(
        path, "mf", {"dimension": model.user.embedding_dim}, provenance
    ) as output:
        torch.save(weights, output / "weights.pt")
        np.savez(
            output / "recall-state.npz", users=users, items=items, supported_users=supported_users
        )


def save_cosine(
    path: Path,
    model: SparseCosineCooccurrence,
    users: np.ndarray,
    items: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    _validate_ids(users, "user")
    _validate_ids(items, "item")
    if len(model.positive) != len(users) or len(model.counts) != len(items):
        raise ValueError("Cosine model and vocabulary sizes disagree")
    for history in model.positive:
        if (
            history.dtype != np.int64
            or history.ndim != 1
            or (history < 0).any()
            or (history >= len(items)).any()
            or len(np.unique(history)) != len(history)
        ):
            raise ValueError("Histories must contain unique in-vocabulary int64 item IDs")
    with _model_directory(path, "cosine", {"n_items": len(items)}, provenance) as output:
        np.savez(
            output / "recall-state.npz",
            users=users,
            items=items,
            positive_items=np.concatenate(model.positive),
            positive_offsets=np.r_[0, np.cumsum([len(v) for v in model.positive])].astype(np.int64),
        )


def save_popularity(
    path: Path,
    counts: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    _validate_ids(users, "user")
    _validate_ids(items, "item")
    if counts.shape != items.shape or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid popularity counts")
    with _model_directory(path, "popularity", {"n_items": len(items)}, provenance) as output:
        np.savez(output / "recall-state.npz", users=users, items=items, counts=counts)


def load_artifact(path: Path) -> ModelArtifact:
    """Validate files, then decode one concrete scorer; legacy exports remain readable."""
    manifest = json.loads((path / "manifest.json").read_text())
    kind = manifest.get("kind")
    if (
        manifest.get("version")
        not in ("fitted-scorer-v2", "fitted-scorer-v3", "fitted-scorer-v4", VERSION)
        or kind not in KINDS
    ):
        raise ValueError("Unsupported fitted scorer version/kind")
    current = manifest["version"] == VERSION
    feature_dir = path / "features" if current else path
    if kind == "deep":
        prefix = "features/" if current else ""
        required = {prefix + "feature-state.npz", prefix + "feature-tables.json", "weights.pt"}
    else:
        required = {"recall-state.npz"}
        if kind == "mf":
            required.add("weights.pt")
    if current:
        required.add("config.json")
    if set(manifest["files"]) != required:
        raise ValueError("Artifact file inventory disagrees with kind")
    for name, digest in manifest["files"].items():
        if not (path / name).is_file() or sha256(path / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if current:
        config = json.loads((path / "config.json").read_text())
    elif kind == "deep":
        config = manifest["model_config"]
    elif kind == "mf":
        config = {"dimension": manifest["dimension"]}
    else:
        config = {}
    identity, provenance = sha256(path / "manifest.json"), manifest["provenance"]
    with torch.random.fork_rng(devices=[]):
        if kind == "deep":
            return _load_ranker(path, feature_dir, config, manifest, identity)
        with np.load(path / "recall-state.npz", allow_pickle=False) as state:
            arrays = dict(state)
        users, items = arrays["users"], arrays["items"]
        _validate_ids(users, "user")
        _validate_ids(items, "item")
        if current and kind in ("cosine", "popularity") and config != {"n_items": len(items)}:
            raise ValueError("Model configuration and item vocabulary disagree")
        if kind == "mf":
            return _load_mf(path, arrays, config, identity, provenance)
        if kind == "cosine":
            return _load_cosine(arrays, identity, provenance)
        counts = arrays["counts"]
        if counts.shape != items.shape or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Invalid popularity counts")
        return PopularityArtifact(counts, users, items, identity, provenance)


def _load_ranker(
    path: Path,
    feature_dir: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    identity: str,
) -> RankerArtifact:
    features = FeatureState.load(feature_dir)
    model_config = DeepConfig(**config)
    if manifest["output_schema"] != {
        model_config.output: {"dtype": "float32", "shape": ["B"], "meaning": "binary logit"}
    }:
        raise ValueError("Output schema/meaning disagrees with model")
    model = DeepModel(model_config)
    load_weights(model, torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    return RankerArtifact(model, features, identity, manifest["provenance"])


def _load_mf(
    path: Path,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    identity: str,
    provenance: dict[str, Any],
) -> MFArtifact:
    users, items, supported = arrays["users"], arrays["items"], arrays["supported_users"]
    if supported.dtype != np.bool_ or supported.shape != users.shape:
        raise ValueError("Invalid MF supported-user mask")
    dimension = config["dimension"]
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Invalid MF dimension")
    model = BPRMF(len(users), len(items), dimension)
    load_weights(model, torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    return MFArtifact(model, users, items, supported, identity, provenance)


def _load_cosine(
    arrays: dict[str, np.ndarray],
    identity: str,
    provenance: dict[str, Any],
) -> CosineArtifact:
    users, items = arrays["users"], arrays["items"]
    offsets, ids = arrays["positive_offsets"], arrays["positive_items"]
    if (
        offsets.dtype != np.int64
        or ids.dtype != np.int64
        or offsets.shape != (len(users) + 1,)
        or ids.ndim != 1
        or offsets[0] != 0
        or offsets[-1] != len(ids)
        or (np.diff(offsets) < 0).any()
    ):
        raise ValueError("Invalid sparse history offsets")
    model = SparseCosineCooccurrence(
        [ids[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)], len(items)
    )
    return CosineArtifact(model, users, items, identity, provenance)
