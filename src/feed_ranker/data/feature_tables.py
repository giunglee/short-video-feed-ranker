"""Train-fitted entity tables shared by event loading and candidate scoring.

Dataset: Gao et al., CIKM 2022, https://kuairand.com/.
Numeric transforms and raw-ID mappings belong here, never inside the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from feed_ranker.persistence import write_json


@dataclass
class FeatureTables:
    """Frozen preprocessing results. Row zero is the unknown-entity row.

    columns/ entities describe physical lookup; vocab_sizes describe fitted
    categorical encodings. descriptions are provenance, not executable schema.
    No model selection, embedding widths, labels or event rows live here.
    """

    users: np.ndarray
    items: np.ndarray
    columns: dict[str, np.ndarray]
    entities: dict[str, str]
    vocab_sizes: dict[str, int]
    descriptions: dict[str, dict[str, Any]]
    _users: pd.Index = field(init=False, repr=False)
    _items: pd.Index = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for keys in (self.users, self.items):
            if keys.dtype != np.int64 or keys.ndim != 1 or (keys < 0).any():
                raise ValueError("Entity keys must be nonnegative int64 vectors")
            if len(keys) and not np.all(keys[1:] > keys[:-1]):
                raise ValueError("Entity keys must be sorted and unique")
        self._users = pd.Index(self.users)
        self._items = pd.Index(self.items)
        if set(self.columns) != set(self.entities) or set(self.vocab_sizes) - self.columns.keys():
            raise ValueError("Feature columns and lookup definitions differ")
        for name, values in self.columns.items():
            entity = self.entities[name]
            if entity not in ("user", "item"):
                raise ValueError(f"{name}: user or item lookup required")
            size = len(self.users if entity == "user" else self.items) + 1
            categorical = name in self.vocab_sizes
            dtype = np.int64 if categorical else np.float32
            if values.shape != (size,) or values.dtype != dtype:
                raise ValueError(f"{name}: feature shape/dtype mismatch")
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: nonfinite fitted feature")
            if categorical and (
                type(self.vocab_sizes[name]) is not int
                or self.vocab_sizes[name] < 1
                or (values < 0).any()
                or (values >= self.vocab_sizes[name]).any()
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
        selected = tuple(self.columns) if names is None else names
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - self.columns.keys()
        ):
            raise ValueError("Require unique known feature names")
        rows = {
            "user": self._users.get_indexer(users) + 1,
            "item": self._items.get_indexer(items) + 1,
        }
        return {
            name: torch.from_numpy(self.columns[name][rows[self.entities[name]]])
            for name in selected
        }

    def save(self, directory: Path) -> None:
        np.savez(
            directory / "feature-state.npz", users=self.users, items=self.items, **self.columns
        )
        write_json(
            directory / "feature-tables.json",
            {
                "entities": self.entities,
                "vocab_sizes": self.vocab_sizes,
                "descriptions": self.descriptions,
            },
        )

    @classmethod
    def load(cls, directory: Path) -> FeatureTables:
        definition = json.loads((directory / "feature-tables.json").read_text())
        with np.load(directory / "feature-state.npz", allow_pickle=False) as state:
            if set(state.files) != {"users", "items", *definition["entities"]}:
                raise ValueError("Unexpected fitted-state arrays")
            return cls(
                state["users"],
                state["items"],
                {n: state[n] for n in definition["entities"]},
                **definition,
            )
