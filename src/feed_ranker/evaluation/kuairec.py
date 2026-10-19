"""KuaiRec observed-pool benchmark adapter; outcomes stay outside recommendation."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feed_ranker.artifacts import sha256
from feed_ranker.data.recall import RecallData
from feed_ranker.evaluation.funnel import GainPool
from feed_ranker.recommendation.contracts import CandidateScope, Request


class ObservedBenchmark:
    def __init__(self, data: RecallData) -> None:
        self.users = data.users.copy()
        self.items = data.items[data.pools.items]
        self.offsets = data.pools.offsets.copy()
        self.pool = self._make_pool(data)

    def __iter__(self) -> Iterator[tuple[Request, CandidateScope]]:
        for u, user in enumerate(self.users):
            lo, hi = self.offsets[u : u + 2]
            yield Request(str(int(user)), int(user)), CandidateScope(self.items[lo:hi])

    def score_frame(self, scores: np.ndarray) -> pd.DataFrame:
        if scores.shape != self.items.shape:
            raise ValueError("Scores differ from query rows")
        return pd.DataFrame(
            {
                "query_id": np.repeat(self.users.astype(str), np.diff(self.offsets)),
                "video_id": self.items,
                "score": scores,
            }
        )

    @staticmethod
    def _make_pool(data: RecallData) -> GainPool:
        val = pd.read_parquet(data.directory / "val.parquet").set_index(["user_id", "video_id"])
        keys = pd.MultiIndex.from_arrays(
            [data.users[data.pools.users], data.items[data.pools.items]]
        )
        aligned = val.reindex(keys)
        if not np.array_equal(aligned.outcome.eq(1).to_numpy(), data.pools.relevant):
            raise ValueError("First-event labels do not match frozen evaluation pool")
        return GainPool(
            data.users,
            data.items[data.pools.items],
            data.pools.offsets,
            aligned.watch_ratio.to_numpy(),
            aligned.video_duration.to_numpy(),
        )


def verify_split_membership(data: RecallData, ranking_data: Path) -> dict[str, str]:
    """Verify source and exact train/validation identities once before a benchmark."""
    meta = json.loads((ranking_data / "meta.json").read_text())
    if meta["sources"]["big_matrix.csv"] != data.meta["raw_big_sha256"]:
        raise ValueError("Ranking and recall use different source data")
    hashes = {}
    for split in ("train", "val"):
        path = ranking_data / f"{split}.parquet"
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != meta["splits"][split]["sha256"]:
            raise ValueError("Prepared ranking split changed")
        codes = [
            pair_codes(b.to_pandas(), "raw_")
            for b in pq.ParquetFile(path).iter_batches(
                batch_size=131072, columns=["raw_user_id", "raw_video_id"]
            )
        ]
        recall = pd.read_parquet(
            data.directory / f"{split}.parquet", columns=["user_id", "video_id"]
        )
        if not np.array_equal(np.unique(np.concatenate(codes)), np.unique(pair_codes(recall))):
            raise ValueError(f"Ranking/recall {split} pair membership differs")
    return hashes


def pair_codes(frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """Collision-free packing for the nonnegative uint32 KuaiRec identifiers."""
    values = []
    for name in ("user_id", "video_id"):
        v = frame[prefix + name].to_numpy()
        if v.dtype.kind not in "iu" or (v < 0).any() or (v >= 2**32).any():
            raise ValueError("Pair IDs must be uint32-compatible integers")
        values.append(v.astype(np.uint64))
    return (values[0] << np.uint64(32)) | values[1]
