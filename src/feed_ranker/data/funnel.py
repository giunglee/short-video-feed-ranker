"""KuaiRec observed-pool benchmark adapter; outcomes stay outside recommendation."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feed_ranker.artifacts import sha256
from feed_ranker.data.recall import RecallData
from feed_ranker.data.recommendation import pair_codes
from feed_ranker.recommendation.contracts import CandidateScope, Recommendation, Request, Retriever
from feed_ranker.training.funnel_evaluation import GainPool


class ObservedQueries:
    def __init__(self, data: RecallData) -> None:
        self.users = data.users.copy()
        self.items = data.items[data.pools.items]
        self.offsets = data.pools.offsets.copy()

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


def make_pool(data: RecallData) -> GainPool:
    val = pd.read_parquet(data.directory / "val.parquet").set_index(["user_id", "video_id"])
    keys = pd.MultiIndex.from_arrays([data.users[data.pools.users], data.items[data.pools.items]])
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


def result_positions(
    pool: GainPool, results: list[Recommendation]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(results) != len(pool.users):
        raise ValueError("Missing/extra recommendation queries")
    candidates, finals = [], []
    for u, (user, result) in enumerate(zip(pool.users, results, strict=True)):
        if result.request != Request(str(int(user)), int(user)):
            raise ValueError("Recommendation query order/identity differs")
        lo, hi = pool.offsets[u : u + 2]
        positions = pd.Index(pool.items[lo:hi]).get_indexer(result.retrieved.items)
        if (positions < 0).any():
            raise ValueError("Recommendation outside observed evaluation scope")
        candidates.append(lo + positions)
        finals.append(lo + positions[result.final_positions])
    return candidates, finals


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


def load_retriever(data: RecallData, directory: Path) -> Retriever:
    """Bind saved training vocabularies/support to label-free retrieval adapters."""
    import torch

    from feed_ranker.models.recall import BPRMF, incidence
    from feed_ranker.recommendation.retrieval import (
        CooccurrenceRetriever,
        MFRetriever,
        PopularityRetriever,
    )

    config = json.loads((directory / "config.json").read_text())
    if config["data_manifest_sha256"] != sha256(data.directory / "meta.json"):
        raise ValueError("Retriever data mismatch")
    artifact = sha256(directory / "config.json")
    if config["method"] == "popularity":
        counts = np.bincount(np.concatenate(data.positive), minlength=len(data.items)).astype(
            np.float32
        )
        return PopularityRetriever(data.items, counts, artifact)
    if config["method"] == "cooccurrence":
        return CooccurrenceRetriever(data.users, data.items, incidence(data), artifact)
    if config["method"] != "mf":
        raise ValueError("Unknown retriever")
    saved = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    if saved["config"] != config:
        raise ValueError("MF checkpoint/config mismatch")
    model = BPRMF(len(data.users), len(data.items), config["dimension"])
    model.load_state_dict(saved["model"])
    return MFRetriever(model, data.users, data.items, data.supported, sha256(directory / "best.pt"))
