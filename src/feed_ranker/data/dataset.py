"""Tensor access to materialized events and deterministic PyTorch batching."""

from collections.abc import Iterator
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


class Batch(TypedDict, total=False):
    features: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    target_masks: dict[str, torch.Tensor]
    target_weights: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]


class EventDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features: tuple[str, ...],
        targets: tuple[str, ...],
    ) -> None:
        metadata_columns = ("event_id", "user_id", "video_id", "time_ms", "warm_item")
        if (
            not features
            or len(set(features)) != len(features)
            or set(features) & set(metadata_columns)
        ):
            raise ValueError("Select unique known model features")
        if (
            not targets
            or len(set(targets)) != len(targets)
            or set(targets) & (set(metadata_columns) | set(features))
        ):
            raise ValueError("Targets must be unique outcomes, separate from features/metadata")
        required = {*metadata_columns, *features, *targets}
        if not frame.columns.is_unique or required - set(frame.columns):
            raise ValueError("Require unique event columns with metadata and selected targets")
        for name in metadata_columns[:-1]:
            values = frame[name].to_numpy()
            if (
                values.dtype.kind not in "iu"
                or (values < 0).any()
                or (values > np.iinfo(np.int64).max).any()
            ):
                raise ValueError(f"{name}: require nonnegative integer event metadata")
        if frame.event_id.duplicated().any():
            raise ValueError("Event IDs must be unique")
        if frame.warm_item.dtype != np.bool_:
            raise ValueError("warm_item: require boolean cohort metadata")
        for name in features:
            values = frame[name].to_numpy()
            if values.dtype not in (np.dtype("int64"), np.dtype("float32")):
                raise ValueError(f"{name}: features require int64 codes or float32 values")
            if not np.isfinite(values).all() or (values.dtype == np.int64 and (values < 0).any()):
                raise ValueError(f"{name}: invalid materialized feature values")
        for name in targets:
            if not frame[name].isin([0, 1]).all():
                raise ValueError(f"{name}: require observed binary targets")
        self.features = {n: torch.from_numpy(frame[n].to_numpy(copy=True)) for n in features}
        self.metadata = {
            n: torch.from_numpy(
                frame[n].to_numpy(dtype=np.bool_ if n == "warm_item" else np.int64, copy=True)
            )
            for n in metadata_columns
        }
        self.targets = {
            n: torch.from_numpy(frame[n].to_numpy(dtype=np.float32, copy=True)) for n in targets
        }

    def __len__(self) -> int:
        return len(self.metadata["event_id"])

    def __getitem__(self, index: int | list[int]) -> Batch:
        scalar = isinstance(index, int)
        ids = [index] if scalar else index
        batch: Batch = {
            "features": {n: v[ids] for n, v in self.features.items()},
            "targets": {n: v[ids] for n, v in self.targets.items()},
            "metadata": {n: v[ids] for n, v in self.metadata.items()},
        }
        return (
            {role: {n: v[0] for n, v in values.items()} for role, values in batch.items()}
            if scalar
            else batch
        )


class EpochBatchSampler(BatchSampler):
    """NumPy seed-plus-epoch order, independent of model/dropout RNG state."""

    def __init__(self, size: int, batch_size: int, seed: int | None) -> None:
        super().__init__(range(size), batch_size, drop_last=False)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = (
            np.arange(len(self.sampler))
            if self.seed is None
            else np.random.default_rng(self.seed + self.epoch).permutation(len(self.sampler))
        )
        for start in range(0, len(order), self.batch_size):
            yield order[start : start + self.batch_size].tolist()


def make_loader(dataset: EventDataset, batch_size: int, seed: int | None = None) -> DataLoader:
    """Batch stored tensors; the sampler supplies deterministic epoch ordering."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    batches = EpochBatchSampler(len(dataset), batch_size, seed)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=batches,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
