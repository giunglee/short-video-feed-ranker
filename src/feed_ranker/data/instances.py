"""Named training instances and deterministic, vectorized PyTorch loading."""

from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from feed_ranker.data.feature_tables import FeatureTables


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
        state: FeatureTables,
        features: tuple[str, ...],
        targets: tuple[str, ...],
    ) -> None:
        metadata_columns = ("event_id", "user_id", "video_id", "time_ms")
        if (
            not features
            or len(set(features)) != len(features)
            or set(features) - state.columns.keys()
        ):
            raise ValueError("Select unique known model features")
        if (
            not targets
            or len(set(targets)) != len(targets)
            or set(targets) & (set(metadata_columns) | set(state.columns))
        ):
            raise ValueError("Targets must be unique outcomes, separate from features/metadata")
        required = {*metadata_columns, *targets}
        if not frame.columns.is_unique or required - set(frame.columns):
            raise ValueError("Require unique event columns with metadata and selected targets")
        for name in metadata_columns:
            values = frame[name].to_numpy()
            if (
                values.dtype.kind not in "iu"
                or (values < 0).any()
                or (values > np.iinfo(np.int64).max).any()
            ):
                raise ValueError(f"{name}: require nonnegative integer event metadata")
        if frame.event_id.duplicated().any():
            raise ValueError("Event IDs must be unique")
        for name in targets:
            if not frame[name].isin([0, 1]).all():
                raise ValueError(f"{name}: require observed binary targets")
        self.state, self.features = state, features
        self.metadata = {
            n: torch.from_numpy(frame[n].to_numpy(dtype=np.int64, copy=True))
            for n in metadata_columns
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
    batches = BatchSampler(order, batch_size=batch_size, drop_last=False)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=batches,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
