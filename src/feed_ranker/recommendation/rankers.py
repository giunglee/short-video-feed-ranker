"""Ranking inference over candidate features; independent of training and labels."""

from pathlib import Path

import numpy as np
import torch

from feed_ranker.artifacts import RankerArtifact, load_artifact
from feed_ranker.recommendation.contracts import Request, ScoredItems


class ModelRanker:
    """Own one fitted model and its matching ETL state; score candidates in batches.

    Input selection comes from the loaded model. Callers choose candidates, never
    feature subsets. Outputs are logits; funnel policy owns selection/ordering.
    """

    def __init__(self, fitted: RankerArtifact, batch_size: int = 8192) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Require a positive batch size")
        self.fitted = fitted
        self.model = fitted.model.cpu().eval()
        self.artifact_id = fitted.identity
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if not features:
            raise ValueError("Require features")
        length = len(next(iter(features.values())))
        if any(len(v) != length for v in features.values()):
            raise ValueError("Feature batch lengths differ")
        self.model.eval()
        outputs = []
        for start in range(0, length, self.batch_size):
            batch = {
                n: torch.from_numpy(v[start : start + self.batch_size].copy())
                for n, v in features.items()
            }
            values = self.model(batch)[self.model.config.output]
            if (
                values.shape != (min(self.batch_size, length - start),)
                or not torch.isfinite(values).all()
            ):
                raise ValueError("Invalid ranker predictions")
            outputs.append(values.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    def score(self, request: Request, items: np.ndarray) -> ScoredItems:
        users = np.full(len(items), request.user_id, dtype=np.int64)
        features = self.fitted.feature_state.transform(
            users, items, self.model.config.feature_names
        )
        return ScoredItems(items, self.predict(features))


def load_ranker(path: Path, *, batch_size: int = 8192, output: str | None = None) -> ModelRanker:
    artifact = load_artifact(path)
    if not isinstance(artifact, RankerArtifact):
        raise ValueError("Expected a Deep ranking artifact")
    if output is not None and output != artifact.model.config.output:
        raise ValueError("Selected output does not match fitted model")
    return ModelRanker(artifact, batch_size)
