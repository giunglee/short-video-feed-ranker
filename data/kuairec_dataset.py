"""
KuaiRec dataset reader — loads preprocessed parquet splits.

Expects extract/ from KuaiRecPreprocessor (train/val/test.parquet + meta.json).
Wide cross columns are precomputed float 0/1 fields when selected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import torch

from data.config import DataConfig


class KuaiRecDataset:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.data_base_dir = Path("data/extract/kuairec")
        self.train_data_path = self.data_base_dir / "train.parquet"
        self.val_data_path = self.data_base_dir / "val.parquet"
        self.test_data_path = self.data_base_dir / "test.parquet"
        self.meta_data_path = self.data_base_dir / "meta.json"

        self._cache: dict[str, pd.DataFrame] = {}

    def vocab_size(self) -> dict[str, int]:
        """Vocab sizes from preprocessor meta.json (includes reserved index 0)."""
        with open(self.meta_data_path, encoding="utf-8") as f:
            return {k: int(v) for k, v in json.load(f).items()}

    def get_full(
        self,
        split: Literal["train", "val", "test"] = "train",
        task: Literal["ctr", "watch", "multitask", "seq"] = "watch",
    ) -> dict[str, Any]:
        if task != "watch":
            raise NotImplementedError(
                f"Currently watch task is only supported. {task} will be supported later"
            )
        df = self._load_split(split)
        return self._pack(df)

    def steps_per_epoch(
        self,
        batch_size: int,
        split: Literal["train", "val", "test"] = "train",
        drop_last: bool = True,
    ) -> int:
        """Number of batches in one full pass over `split` (see epoch_batches)."""
        n = len(self._load_split(split))
        return n // batch_size if drop_last else -(-n // batch_size)

    def epoch_batches(
        self,
        batch_size: int,
        split: Literal["train", "val", "test"] = "train",
        drop_last: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield shuffled, non-overlapping batches covering `split` exactly once."""
        df = self._load_split(split)
        n = len(df)
        perm = np.random.permutation(n)
        n_batches = n // batch_size if drop_last else -(-n // batch_size)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            yield self._pack(df.iloc[idx])

    def _load_split(self, split: Literal["train", "val", "test"]) -> pd.DataFrame:
        if split not in self._cache:
            path = {
                "train": self.train_data_path,
                "val": self.val_data_path,
                "test": self.test_data_path,
            }[split]
            self._cache[split] = pd.read_parquet(path, columns=self.config.columns)
        return self._cache[split]

    def _pack(self, df: pd.DataFrame) -> dict[str, Any]:
        categorical_cols = set(self.config.feature.categorical_cols)
        dense_cols = set(self.config.feature.dense_cols)
        cross_cols = set(self.config.feature.cross_feature_cols)

        features: dict[str, torch.Tensor] = {}
        for col in self.config.feature_cols:
            if col in categorical_cols:
                features[col] = torch.as_tensor(df[col].to_numpy(), dtype=torch.long)
            elif col in dense_cols or col in cross_cols:
                # Crosses are ETL float 0/1; denser numerics may be normalized in encoder.
                series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                features[col] = torch.as_tensor(
                    series.to_numpy(dtype="float32"), dtype=torch.float32
                )
            else:
                raise ValueError(
                    f"{col} is not listed as embedding, cross, or dense in FeatureConfig"
                )

        label_col = self.config.label_col
        label_series = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0)
        label = torch.as_tensor(
            label_series.to_numpy(dtype="float32"), dtype=torch.float32
        )

        return {"feature": features, "label": label}
