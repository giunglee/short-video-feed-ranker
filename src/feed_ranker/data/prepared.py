"""Versioned KuaiRand preparation and label-free entity gathers.

Dataset: Gao et al., CIKM 2022, https://kuairand.com/.
Tables hold fitted values once; event batches gather by raw entity ID. Learned
statistics use training events only. Snapshot availability remains an assumption.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.kuairand import TRAIN_END, VAL_END, read_cohort
from feed_ranker.data.metadata import ANONYMOUS, MetadataEncoding

VERSION = "kuairand-prepared-v2"


class Batch(TypedDict, total=False):
    features: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    target_masks: dict[str, torch.Tensor]
    target_weights: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: str
    dtype: str
    entity: str = "event"
    units: str = "none"
    transform: str = "identity-v1"
    missing: str = "not allowed"
    availability: str = "training events only"
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        if not self.name or self.role not in ("feature", "target", "metadata"):
            raise ValueError("Require a field name and explicit role")
        if self.dtype not in ("int64", "float32") or self.entity not in ("user", "item", "event"):
            raise ValueError("Unsupported field dtype/entity")
        if self.vocab_size is not None and (self.dtype != "int64" or self.vocab_size < 1):
            raise ValueError("Categorical fields require positive vocabulary sizes")


@dataclass
class FeatureState:
    """Raw keys and transformed entity columns, with row zero reserved for OOV.

    Tables are shared across datasets and recipes; no event-sized feature copies.
    Model architecture, widths and feature selection are deliberately absent.
    """

    users: np.ndarray
    items: np.ndarray
    columns: dict[str, np.ndarray]
    schema: dict[str, FieldSpec]

    def __post_init__(self) -> None:
        for keys in (self.users, self.items):
            if keys.dtype != np.int64 or keys.ndim != 1 or (keys < 0).any():
                raise ValueError("Entity keys must be nonnegative int64 vectors")
            if len(keys) and not np.all(keys[1:] > keys[:-1]):
                raise ValueError("Entity keys must be sorted and unique")
        self._users = pd.Index(self.users)
        self._items = pd.Index(self.items)
        if set(self.columns) != set(self.schema):
            raise ValueError("Feature columns and schema differ")
        for name, values in self.columns.items():
            spec = self.schema[name]
            size = len(self.users if spec.entity == "user" else self.items) + 1
            if spec.role != "feature" or spec.entity not in ("user", "item"):
                raise ValueError(f"{name}: entity feature required")
            if values.shape != (size,) or str(values.dtype) != spec.dtype:
                raise ValueError(f"{name}: feature shape/dtype mismatch")
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: nonfinite fitted feature")
            if spec.vocab_size is not None and (
                (values < 0).any() or (values >= spec.vocab_size).any()
            ):
                raise ValueError(f"{name}: categorical code outside fitted vocabulary")

    def gather(
        self, users: np.ndarray, items: np.ndarray, names: tuple[str, ...] | None = None
    ) -> dict[str, torch.Tensor]:
        if (
            users.ndim != 1
            or users.shape != items.shape
            or users.dtype != np.int64
            or items.dtype != np.int64
        ):
            raise ValueError("Require aligned int64 raw user/item vectors")
        if (users < 0).any() or (items < 0).any():
            raise ValueError("Raw IDs cannot be negative")
        rows = {
            "user": self._users.get_indexer(users) + 1,
            "item": self._items.get_indexer(items) + 1,
        }
        selected = tuple(self.columns) if names is None else names
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - self.columns.keys()
        ):
            raise ValueError("Require unique known feature names")
        return {
            name: torch.from_numpy(self.columns[name][rows[self.schema[name].entity]])
            for name in selected
        }

    def save(self, directory: Path) -> None:
        np.savez(
            directory / "feature-state.npz", users=self.users, items=self.items, **self.columns
        )
        write_json(
            directory / "feature-schema.json", {n: asdict(s) for n, s in self.schema.items()}
        )

    @classmethod
    def load(cls, directory: Path) -> FeatureState:
        schema = {
            n: FieldSpec(**s)
            for n, s in json.loads((directory / "feature-schema.json").read_text()).items()
        }
        with np.load(directory / "feature-state.npz", allow_pickle=False) as state:
            if set(state.files) != {"users", "items", *schema}:
                raise ValueError("Unexpected fitted-state arrays")
            return cls(state["users"], state["items"], {n: state[n] for n in schema}, schema)


def fit_features(
    train: pd.DataFrame, basic: pd.DataFrame | None = None, profiles: pd.DataFrame | None = None
) -> FeatureState:
    users = np.sort(train.user_id.unique()).astype(np.int64)
    warm_items = np.sort(train.video_id.unique()).astype(np.int64)
    if not len(users) or not len(warm_items):
        raise ValueError("Training events cannot be empty")
    if (basic is None) != (profiles is None):
        raise ValueError("Provide both item and user snapshots, or neither")
    metadata = None if basic is None else MetadataEncoding.fit(users, warm_items, basic, profiles)
    items = warm_items if metadata is None else np.union1d(warm_items, metadata.videos)
    columns: dict[str, np.ndarray] = {}
    schema: dict[str, FieldSpec] = {}

    def add(name: str, values: np.ndarray, entity: str, **kwargs: Any) -> None:
        columns[name] = values
        schema[name] = FieldSpec(name, "feature", str(values.dtype), entity, **kwargs)

    add(
        "user_code",
        np.arange(len(users) + 1, dtype=np.int64),
        "user",
        vocab_size=len(users) + 1,
        missing="zero OOV",
    )
    add(
        "item_code",
        np.r_[0, pd.Index(warm_items).get_indexer(items) + 1],
        "item",
        vocab_size=len(warm_items) + 1,
        missing="zero OOV",
    )

    def duration_fields(prefix: str, values: np.ndarray, availability: str) -> None:
        # Match the previous torch float32 definition exactly, now applied in DATA.
        add(
            prefix + "_log1p",
            torch.log1p(torch.from_numpy(values).clamp(min=0)).numpy(),
            "item",
            units="log1p(milliseconds)",
            transform="torch-float32-log1p-v1",
            missing="zero with separate indicator",
            availability=availability,
        )
        add(
            prefix + "_missing",
            (values == 0).astype(np.float32),
            "item",
            missing="one indicates missing",
            availability=availability,
        )
        bands = np.searchsorted([10_000, 30_000, 60_000], values, side="left").astype(np.int64) + 1
        bands[values == 0] = 0
        add(
            prefix + "_band",
            bands,
            "item",
            vocab_size=5,
            transform="duration-bands-v1",
            missing="zero OOV/missing",
            availability=availability,
        )

    medians = train.loc[train.duration_ms.gt(0)].groupby("video_id").duration_ms.median()
    duration_fields(
        "train_duration",
        np.r_[0, medians.reindex(items).fillna(0)].astype(np.float32),
        "train-fitted item median",
    )
    if metadata is not None:
        rows = pd.Index(metadata.videos).get_indexer(items) + 1
        duration_fields(
            "metadata_duration",
            np.r_[0, metadata.basic_duration[rows]].astype(np.float32),
            "snapshot; historical availability unverified",
        )
        add(
            "author_code",
            np.r_[0, metadata.author_codes[rows]],
            "item",
            vocab_size=len(metadata.author_vocabulary) + 1,
            missing="zero OOV",
            availability="snapshot; historical availability unverified",
        )
        for i, name in enumerate(ANONYMOUS):
            add(
                name,
                metadata.anonymous_codes[:, i].copy(),
                "user",
                vocab_size=len(metadata.anonymous_vocabularies[i]) + 1,
                missing="zero null/sentinel/OOV",
                availability="anonymous snapshot; historical availability unverified",
            )
    return FeatureState(users, items, columns, schema)


def prepare_kuairand(
    raw: Path,
    out: Path,
    *,
    targets: tuple[str, ...] = ("long_view",),
    users: int = 32,
    cohort_seed: int = 2026,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
    basic: Path | None = None,
    profiles: Path | None = None,
) -> PreparedDataset:
    if not targets or len(set(targets)) != len(targets) or set(targets) - {"long_view", "is_click"}:
        raise ValueError("Configure unique supported binary target columns")
    train, val, source = read_cohort(
        raw,
        label=targets[0],
        users=users,
        cohort_seed=cohort_seed,
        train_end=train_end,
        val_end=val_end,
    )
    for frame in (train, val):
        for target in targets:
            if target not in frame or not frame[target].isin([0, 1]).all():
                raise ValueError(f"{target}: expected observed binary targets")
    state = fit_features(
        train,
        None if basic is None else pd.read_csv(basic),
        None if profiles is None else pd.read_csv(profiles),
    )
    out.mkdir(parents=True, exist_ok=False)
    state.save(out)
    schema = {n: asdict(s) for n, s in state.schema.items()}
    for name in targets:
        schema[name] = asdict(
            FieldSpec(name, "target", "float32", availability="observed outcome; never model input")
        )
    keys = ["event_id", "user_id", "video_id", "time_ms"]
    for name in keys:
        schema[name] = asdict(FieldSpec(name, "metadata", "int64"))
    for name, frame in (("train", train), ("validation", val)):
        frame = frame[keys + list(targets)].astype(
            {**{k: "int64" for k in keys}, **{t: "float32" for t in targets}}
        )
        frame.to_parquet(out / f"{name}.parquet", index=False)
    files = ["train.parquet", "validation.parquet", "feature-state.npz", "feature-schema.json"]
    write_json(
        out / "manifest.json",
        {
            "version": VERSION,
            "schema": schema,
            "targets": list(targets),
            **source,
            "train_end": train_end,
            "val_end": val_end,
            "cohort_seed": cohort_seed,
            "traffic": "standard/tab1",
            "dedup": "complete source records before projection",
            "snapshot_sha256": {p.name: sha256(p) for p in (basic, profiles) if p is not None},
            "files": {name: sha256(out / name) for name in files},
        },
    )
    return open_prepared(out)


class PreparedDataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        if self.manifest.get("version") != VERSION:
            raise ValueError("Unsupported prepared-data version")
        self.schema = {n: FieldSpec(**s) for n, s in self.manifest["schema"].items()}
        if any(n != s.name for n, s in self.schema.items()):
            raise ValueError("Schema field names disagree")
        if self.manifest["train_end"] >= self.manifest["val_end"]:
            raise ValueError("Invalid temporal boundaries")
        for n in self.manifest["targets"]:
            if n not in self.schema or self.schema[n].role != "target":
                raise ValueError("Invalid configured target role")
        self.identity = sha256(path / "manifest.json")

    def _check(self, name: str) -> None:
        if (
            name not in self.manifest["files"]
            or sha256(self.path / name) != self.manifest["files"][name]
        ):
            raise ValueError(f"Prepared input changed: {name}")

    def read_split(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        if name not in ("train", "validation"):
            raise ValueError("Only train and validation are exposed by this protocol")
        self._check(name + ".parquet")
        if columns is not None and (
            not columns
            or len(set(columns)) != len(columns)
            or any(c not in self.schema or self.schema[c].entity != "event" for c in columns)
        ):
            raise ValueError("Require unique event/target columns")
        return pd.read_parquet(self.path / (name + ".parquet"), columns=columns)

    def load_feature_state(self) -> FeatureState:
        for name in ("feature-state.npz", "feature-schema.json"):
            self._check(name)
        state = FeatureState.load(self.path)
        if state.schema != {n: s for n, s in self.schema.items() if s.role == "feature"}:
            raise ValueError("Manifest and fitted feature schema disagree")
        return state

    def validate(self) -> None:
        self.load_feature_state()
        seen: set[int] = set()
        for split in ("train", "validation"):
            frame = self.read_split(split)
            for name in frame:
                spec = self.schema[name]
                if str(frame[name].dtype) != spec.dtype or not np.isfinite(frame[name]).all():
                    raise ValueError(f"{split}/{name}: invalid dtype/values")
                if spec.role == "target" and not frame[name].isin([0, 1]).all():
                    raise ValueError(f"{name}: invalid binary target")
            if (
                frame.empty
                or frame.event_id.duplicated().any()
                or seen.intersection(frame.event_id)
            ):
                raise ValueError("Empty split or duplicate event identities")
            seen.update(frame.event_id)
            valid = (
                frame.time_ms.lt(self.manifest["train_end"])
                if split == "train"
                else frame.time_ms.ge(self.manifest["train_end"])
                & frame.time_ms.lt(self.manifest["val_end"])
            )
            if not valid.all():
                raise ValueError("Temporal boundaries disagree with manifest")


def open_prepared(path: Path) -> PreparedDataset:
    """Read only a manifest; callers explicitly choose which content to load."""
    return PreparedDataset(path)


class EventDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        state: FeatureState,
        features: tuple[str, ...],
        targets: tuple[str, ...],
    ) -> None:
        if set(features) - state.schema.keys() or not features:
            raise ValueError("Select known model features")
        if not targets or set(targets) - set(frame.columns):
            # Target semantics are validated against the prepared schema by the job.
            raise ValueError("Select target columns present in events")
        self.state, self.features = state, features
        self.metadata = {
            n: torch.from_numpy(frame[n].to_numpy(dtype=np.int64, copy=True))
            for n in ("event_id", "user_id", "video_id", "time_ms")
        }
        self.targets = {
            n: torch.from_numpy(frame[n].to_numpy(dtype=np.float32, copy=True)) for n in targets
        }
        self.metadata["warm_item"] = state.gather(
            self.metadata["user_id"].numpy(), self.metadata["video_id"].numpy(), ("item_code",)
        )["item_code"].gt(0)

    def __len__(self) -> int:
        return len(self.metadata["event_id"])

    def __getitem__(self, index: int | list[int]) -> Batch:
        scalar = isinstance(index, int)
        ids = [index] if scalar else index
        metadata = {n: v[ids] for n, v in self.metadata.items()}
        batch: Batch = {
            "features": self.state.gather(
                metadata["user_id"].numpy(), metadata["video_id"].numpy(), self.features
            ),
            "targets": {n: v[ids] for n, v in self.targets.items()},
            "metadata": metadata,
        }
        return (
            {role: {n: v[0] for n, v in values.items()} for role, values in batch.items()}
            if scalar
            else batch
        )


def make_loader(dataset: EventDataset, batch_size: int, seed: int | None = None) -> DataLoader:
    """Batch-index sampler preserves NumPy ordering without per-event collation."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    order = (
        np.arange(len(dataset))
        if seed is None
        else np.random.default_rng(seed).permutation(len(dataset))
    )
    batches = (order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size))
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=batches,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
