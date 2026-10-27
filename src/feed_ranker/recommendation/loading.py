"""Turn fitted model state into runtime adapters; never open event datasets."""

from pathlib import Path

from feed_ranker.models.artifacts import load_artifact
from feed_ranker.recommendation.contracts import Retriever
from feed_ranker.recommendation.kuairand import SparseCooccurrenceRetriever
from feed_ranker.recommendation.rankers import ModelRanker
from feed_ranker.recommendation.retrievers import MFRetriever, PopularityRetriever


def load_ranker(path: Path, *, batch_size: int = 8192, output: str | None = None) -> ModelRanker:
    artifact = load_artifact(path)
    if artifact.kind != "deep":
        raise ValueError("Expected a Deep ranking artifact")
    return ModelRanker(
        artifact.model, artifact.feature_state, artifact.identity, batch_size, output=output
    )


def load_retriever(path: Path) -> Retriever:
    artifact = load_artifact(path)
    if artifact.kind == "popularity":
        return PopularityRetriever(artifact.items, artifact.model, artifact.identity)
    if artifact.kind == "cosine":
        return SparseCooccurrenceRetriever(
            artifact.users, artifact.items, artifact.model, artifact.identity
        )
    if artifact.kind == "mf":
        return MFRetriever(
            artifact.model,
            artifact.users,
            artifact.items,
            artifact.supported_users,
            artifact.identity,
        )
    raise ValueError("Expected a fitted recall artifact")
