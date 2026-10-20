"""KuaiRand adapter for the shared request pipeline; no outcomes or split readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feed_ranker.inference.rankers import ModelRanker
from feed_ranker.inference.retrievers import select
from feed_ranker.models.sparse_cooccurrence import SparseCosineCooccurrence
from feed_ranker.recommendation.contracts import CandidateScope, Request, RetrievalResult, Retriever


class SparseCooccurrenceRetriever:
    name = "cooccurrence"

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        model: SparseCosineCooccurrence,
        artifact_id: str,
    ) -> None:
        if len(users) != len(model.positive) or len(items) != len(model.counts):
            raise ValueError("Cooccurrence vocabulary mismatch")
        self.users, self.items = pd.Index(users), pd.Index(items)
        self.model, self.artifact_id = model, artifact_id

    def retrieve(self, request: Request, scope: CandidateScope, limit: int) -> RetrievalResult:
        user = self.users.get_indexer([request.user_id])[0]
        ids = self.items.get_indexer(scope.items)
        known = ids >= 0
        items, ids = scope.items[known], ids[known]
        if user < 0:
            return select(request, items[:0], np.empty(0), limit)
        scores = self.model.score(int(user), ids)
        supported = scores > 0
        return select(request, items[supported], scores[supported], limit)


def load_ranker(path: "Path") -> "ModelRanker":
    """Load fitted tensors and recipe without raw data or training-code imports."""
    import torch

    from feed_ranker.artifacts import sha256
    from feed_ranker.features.kuairand import Encoding, KuaiRandFeatures
    from feed_ranker.inference.rankers import ModelRanker
    from feed_ranker.models.ranking_factory import ModelSpec, build_model

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != "kuairand-ranker-v1":
        raise ValueError("Unsupported KuaiRand ranker artifact")
    encoding = Encoding(*(state[name].numpy() for name in ["users", "items", "duration"]))
    features = KuaiRandFeatures(encoding, state["feature_set"])
    model = build_model(features.config, ModelSpec("deep", tuple(state["hidden"])))
    model.load_state_dict(state["model"])
    return ModelRanker(model, features, sha256(path))


def load_retrievers(directory: Path) -> list["Retriever"]:
    """Reconstruct fitted recall channels with no event tables or evaluation labels."""
    import json

    import torch

    from feed_ranker.artifacts import sha256
    from feed_ranker.inference.retrievers import MFRetriever, PopularityRetriever
    from feed_ranker.models.recall import BPRMF

    manifest = json.loads((directory / "retrieval.json").read_text())
    for name in ["recall_state.npz", "mf.pt"]:
        if sha256(directory / name) != manifest["files"][name]:
            raise ValueError(f"Changed fitted recall artifact: {name}")
    with np.load(directory / "recall_state.npz", allow_pickle=False) as state:
        users, items = state["users"], state["items"]
        offsets, ids = state["positive_offsets"], state["positive_items"]
        supported = state["supported"]
    positive = [ids[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    cosine = SparseCosineCooccurrence(positive, len(items))
    checkpoint = torch.load(directory / "mf.pt", map_location="cpu", weights_only=True)
    mf = BPRMF(len(users), len(items), checkpoint["dimension"])
    mf.load_state_dict(checkpoint["model"])
    identity = manifest["data_manifest_sha256"]
    return [
        PopularityRetriever(items, cosine.counts, identity + ":popularity"),
        SparseCooccurrenceRetriever(users, items, cosine, identity + ":cosine-exact"),
        MFRetriever(mf, users, items, supported, sha256(directory / "mf.pt")),
    ]
