"""Read and validate prepared KuaiRand events and fitted feature state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feed_ranker.data.entity_features import EntityFeatureLookup
from feed_ranker.persistence import sha256

VERSION = "kuairand-prepared-v3"


class PreparedDataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        if self.manifest.get("version") != VERSION:
            raise ValueError("Unsupported prepared-data version")
        self.targets = tuple(self.manifest["targets"])
        self.event_dtypes = self.manifest["event_dtypes"]
        expected = {
            **dict.fromkeys(("event_id", "user_id", "video_id", "time_ms"), "int64"),
            **dict.fromkeys(self.targets, "float32"),
        }
        if not self.targets or set(self.targets) - {"long_view", "is_click"}:
            raise ValueError("Invalid prepared binary targets")
        if self.event_dtypes != expected:
            raise ValueError("Invalid event column definitions")
        if self.manifest["train_end"] >= self.manifest["val_end"]:
            raise ValueError("Invalid temporal boundaries")
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
            or any(c not in self.event_dtypes for c in columns)
        ):
            raise ValueError("Require unique event/target columns")
        return pd.read_parquet(self.path / (name + ".parquet"), columns=columns)

    def load_feature_state(self) -> EntityFeatureLookup:
        for name in ("feature-state.npz", "feature-tables.json"):
            self._check(name)
        return EntityFeatureLookup.load(self.path)

    def validate(self) -> None:
        self.load_feature_state()
        seen: set[int] = set()
        for split in ("train", "validation"):
            frame = self.read_split(split)
            if set(frame) != set(self.event_dtypes):
                raise ValueError("Event columns disagree with manifest")
            for name in frame:
                if (
                    str(frame[name].dtype) != self.event_dtypes[name]
                    or not np.isfinite(frame[name]).all()
                ):
                    raise ValueError(f"{split}/{name}: invalid dtype/values")
                if name in self.targets and not frame[name].isin([0, 1]).all():
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
