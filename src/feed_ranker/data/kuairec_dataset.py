"""Prepared KuaiRec batches: categorical int64, numeric/label float32, raw user int64.

Training caches selected columns and shuffles with a local NumPy generator.
Evaluation streams parquet in stored order. Both paths retain incomplete batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from feed_ranker.data.config import DataConfig
from feed_ranker.data.schema import PROTOCOL, RecommendationBatch


class KuaiRecDataset:
    """Own the prepared-data manifest and selected-column training cache.

    Bulk dataframe conversion avoids constructing one Python object per example.
    Validation/test rows are never included in the training cache.
    """

    def __init__(self, config: DataConfig, directory: Path) -> None:
        self.config = config
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "meta.json").read_text())
        if self.meta.get("protocol") != PROTOCOL:
            raise ValueError("Expected big-to-small data; legacy extracts are not compatible")
        self._columns = list(dict.fromkeys([*config.columns, "raw_user_id", "raw_video_id"]))
        self.has_pair_weights = self.meta.get("preprocessing_version") == "signature-dedup-v2"
        if self.has_pair_weights:
            self._columns.append("pair_weight")
        self._train: pd.DataFrame | None = None

    def vocab_size(self) -> dict[str, int]:
        return {name: len(values) + 1 for name, values in self.meta["vocab"].items()}

    def loader(self, split: str, batch_size: int, shuffle_seed: int | None = None) -> DataLoader:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if shuffle_seed is not None and split != "train":
            raise ValueError("Only training may be shuffled")

        batches = _ParquetBatches(self, split, batch_size, shuffle_seed)
        # DataLoader consumes a base seed on iterator creation. Keep that RNG local
        # so evaluation does not perturb model dropout or initialization.
        generator = torch.Generator().manual_seed(0)
        return DataLoader(batches, batch_size=None, num_workers=0, generator=generator)

    def fixed_train_loader(self, rows: int, seed: int, batch_size: int) -> DataLoader:
        """Uniform row sample without replacement, fixed independently of training RNG."""
        if rows <= 0 or batch_size <= 0:
            raise ValueError("Training diagnostic rows and batch size must be positive")
        if self._train is None:
            self._train = pd.read_parquet(self.directory / "train.parquet", columns=self._columns)
        indices = np.sort(
            np.random.default_rng(seed).choice(
                len(self._train), size=min(rows, len(self._train)), replace=False
            )
        )
        sample = self._train.iloc[indices]
        batches = [
            self._pack(sample.iloc[start : start + batch_size])
            for start in range(0, len(sample), batch_size)
        ]
        return DataLoader(
            batches, batch_size=None, num_workers=0, generator=torch.Generator().manual_seed(0)
        )

    def _shuffled_train_frames(self, batch_size: int, seed: int) -> Iterator[pd.DataFrame]:
        if self._train is None:
            self._train = pd.read_parquet(self.directory / "train.parquet", columns=self._columns)
        order = np.random.default_rng(seed).permutation(len(self._train))
        for start in range(0, len(order), batch_size):
            yield self._train.iloc[order[start : start + batch_size]]

    def _stream_frames(self, split: str, batch_size: int) -> Iterator[pd.DataFrame]:
        with pq.ParquetFile(self.directory / f"{split}.parquet") as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=self._columns):
                yield batch.to_pandas()

    def _pack(self, frame: pd.DataFrame) -> RecommendationBatch:
        features: dict[str, torch.Tensor] = {}
        categorical = set(self.config.feature.categorical_cols)
        for name in self.config.feature_cols:
            dtype = np.int64 if name in categorical else np.float32
            values = frame[name].to_numpy(dtype=dtype, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite feature: {name}")
            features[name] = torch.from_numpy(values)
        labels = frame[self.config.label_col].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(labels).all():
            raise ValueError("Invalid labels; never impute outcomes")
        user_ids = frame["raw_user_id"].to_numpy(dtype=np.int64, copy=True)
        result: RecommendationBatch = {
            "feature": features,
            "label": torch.from_numpy(labels),
            "raw_user_id": torch.from_numpy(user_ids),
            "raw_video_id": torch.from_numpy(
                frame["raw_video_id"].to_numpy(dtype=np.int64, copy=True)
            ),
        }

        if self.has_pair_weights:
            weights = frame["pair_weight"].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Pair weights must be finite and positive")
            result["pair_weight"] = torch.from_numpy(weights)
        return result


class _ParquetBatches(IterableDataset):
    """Prebatched columnar reads; single-worker policy prevents duplicated streams."""

    def __init__(self, dataset: KuaiRecDataset, split: str, size: int, seed: int | None) -> None:
        self.dataset = dataset
        self.split = split
        self.size = size
        self.seed = seed

    def __iter__(self) -> Iterator[RecommendationBatch]:
        if get_worker_info() is not None:
            raise RuntimeError("Parquet batch streaming currently requires num_workers=0")
        if self.seed is None:
            frames = self.dataset._stream_frames(self.split, self.size)
        else:
            frames = self.dataset._shuffled_train_frames(self.size, self.seed)
        for frame in frames:
            yield self.dataset._pack(frame)
