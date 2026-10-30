"""Read materialized event splits; audit saved files independently of batching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from feed_ranker.persistence import sha256

if TYPE_CHECKING:
    from feed_ranker.preprocessing.features import FeatureState

VERSION = "kuairand-prepared-v4"


class ParquetReader:
    """A manifest and its Parquet files, not a PyTorch Dataset.

    Opening reads only the manifest. Split reads verify the requested file and
    column types; validate() explicitly audits all splits and fitted state.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        if self.manifest.get("version") != VERSION:
            raise ValueError("Unsupported prepared-data version; regenerate materialized splits")
        self.targets = tuple(self.manifest["targets"])
        self.feature_dtypes = self.manifest["feature_dtypes"]
        self.vocab_sizes = self.manifest["vocab_sizes"]
        self.metadata_dtypes = self.manifest["metadata_dtypes"]
        self.splits = self.manifest["splits"]
        if (
            not self.targets
            or len(set(self.targets)) != len(self.targets)
            or set(self.targets) - {"long_view", "is_click"}
        ):
            raise ValueError("Invalid prepared binary targets")
        if self.metadata_dtypes != {
            **dict.fromkeys(("event_id", "user_id", "video_id", "time_ms"), "int64"),
            "warm_item": "bool",
        }:
            raise ValueError("Invalid event metadata definitions")
        if (
            not self.feature_dtypes
            or set(self.feature_dtypes) & (set(self.metadata_dtypes) | set(self.targets))
            or set(self.vocab_sizes) - self.feature_dtypes.keys()
            or any(type(v) is not int or v < 1 for v in self.vocab_sizes.values())
            or any(
                dtype != ("int64" if name in self.vocab_sizes else "float32")
                for name, dtype in self.feature_dtypes.items()
            )
        ):
            raise ValueError("Invalid feature column definitions")
        if not self.splits or set(self.splits) - {"train", "validation", "test"}:
            raise ValueError("Invalid declared splits")
        for bounds in self.splits.values():
            start, end = bounds.get("start"), bounds.get("end")
            if (
                set(bounds) - {"start", "end"}
                or any(type(v) is not int or v < 0 for v in (start, end) if v is not None)
                or (start is not None and end is not None and start >= end)
            ):
                raise ValueError("Invalid temporal boundaries")
        if (
            self.splits.get("train") != {"end": self.manifest["train_end"]}
            or self.splits.get("validation")
            != {"start": self.manifest["train_end"], "end": self.manifest["val_end"]}
            or self.manifest["train_end"] >= self.manifest["val_end"]
            or (
                "test" in self.splits
                and (
                    self.splits["test"].get("start") is None
                    or self.splits["test"]["start"] < self.manifest["val_end"]
                )
            )
        ):
            raise ValueError("Split boundaries disagree with the temporal protocol")
        self.column_dtypes = {
            **self.metadata_dtypes,
            **self.feature_dtypes,
            **dict.fromkeys(self.targets, "float32"),
        }
        self.identity = sha256(path / "manifest.json")

    def _check(self, name: str) -> None:
        if (
            name not in self.manifest["files"]
            or not (self.path / name).is_file()
            or sha256(self.path / name) != self.manifest["files"][name]
        ):
            raise ValueError(f"Prepared input changed: {name}")

    def read_split(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        if name not in self.splits:
            raise ValueError(f"Split {name!r} is not declared in this dataset")
        if columns is not None and (
            not columns
            or len(set(columns)) != len(columns)
            or set(columns) - self.column_dtypes.keys()
        ):
            raise ValueError("Require unique known columns")
        self._check(name + ".parquet")
        frame = pd.read_parquet(self.path / (name + ".parquet"), columns=columns)
        expected = set(self.column_dtypes if columns is None else columns)
        if not frame.columns.is_unique or set(frame.columns) != expected:
            raise ValueError("Event columns disagree with manifest")
        for column in frame:
            if str(frame[column].dtype) != self.column_dtypes[column]:
                raise ValueError(f"{name}/{column}: dtype disagrees with manifest")
        return frame

    def load_feature_state(self) -> FeatureState:
        """Load state for candidate scoring/export; ordinary split reads need none."""
        from feed_ranker.preprocessing.features import FeatureState

        for name in ("feature-state.npz", "feature-tables.json"):
            self._check(name)
        state = FeatureState.load(self.path)
        if (
            set(state.feature_names) != set(self.feature_dtypes)
            or state.vocab_sizes != self.vocab_sizes
        ):
            raise ValueError("Feature state and materialized schema disagree")
        return state

    def validate(self) -> None:
        """Full audit, run by preparation and the explicit validate-data command."""
        self.load_feature_state()
        seen: set[int] = set()
        for split, bounds in self.splits.items():
            frame = self.read_split(split)
            for name in frame:
                values = frame[name].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{split}/{name}: invalid values")
                if name in self.targets and not frame[name].isin([0, 1]).all():
                    raise ValueError(f"{name}: invalid binary target")
                if values.dtype == np.int64 and (values < 0).any():
                    raise ValueError(f"{name}: negative ID or category code")
                if name in self.vocab_sizes and (values >= self.vocab_sizes[name]).any():
                    raise ValueError(f"{name}: categorical code outside fitted vocabulary")
            if (
                frame.empty
                or frame.event_id.duplicated().any()
                or seen.intersection(frame.event_id)
            ):
                raise ValueError("Empty split or duplicate event identities")
            seen.update(frame.event_id)
            if not frame.warm_item.eq(frame.item_code.gt(0)).all():
                raise ValueError("warm_item disagrees with fitted item identity")
            if (bounds.get("start") is not None and frame.time_ms.lt(bounds["start"]).any()) or (
                bounds.get("end") is not None and frame.time_ms.ge(bounds["end"]).any()
            ):
                raise ValueError("Temporal boundaries disagree with manifest")
