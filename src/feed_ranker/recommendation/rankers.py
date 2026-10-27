"""Selected-candidate model inference; no training or benchmark dependencies."""

from typing import Protocol

import numpy as np
import torch
from torch import nn

from feed_ranker.data.prepared import FeatureState
from feed_ranker.models.deep import DeepModel
from feed_ranker.recommendation.contracts import Request, ScoredItems


class CandidateFeatures(Protocol):
    def build(self, users: np.ndarray, items: np.ndarray) -> dict[str, np.ndarray]: ...


class ModelRanker:
    """Own the model, its matching feature builder, and inference batching policy."""

    def __init__(
        self,
        model: nn.Module,
        features: CandidateFeatures | FeatureState,
        artifact_id: str,
        batch_size: int = 8192,
        *,
        output: str | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Require a positive batch size")
        self.output = (
            (model.config.output if output is None else output)
            if isinstance(model, DeepModel)
            else None
        )
        if isinstance(model, DeepModel) and (
            self.output != model.config.output or not isinstance(features, FeatureState)
        ):
            raise ValueError("Named ranker output/features disagree with model")
        if not isinstance(model, DeepModel) and output is not None:
            raise ValueError("Legacy rankers have one unnamed output")
        self.model = model.cpu().eval()
        self.features, self.artifact_id, self.batch_size = features, artifact_id, batch_size

    @torch.no_grad()
    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if not features:
            raise ValueError("Require features")
        n = len(next(iter(features.values())))
        if any(len(v) != n for v in features.values()):
            raise ValueError("Feature batch lengths differ")
        self.model.eval()
        outputs = []
        for start in range(0, n, self.batch_size):
            batch = {
                name: torch.from_numpy(v[start : start + self.batch_size].copy())
                for name, v in features.items()
            }
            rows = min(self.batch_size, n - start)
            if self.output is not None:
                value = self.model({"features": batch})[self.output]
                if value.shape != (rows,):
                    raise ValueError("Named ranker must return [B]")
            else:
                value = self.model(batch)
                if value.shape == (rows, 1):
                    value = value[:, 0]
                elif value.shape != (rows,):
                    raise ValueError("Legacy ranker must return [B] or [B,1]")
            value = value.cpu().numpy()
            if len(value) != min(self.batch_size, n - start) or not np.isfinite(value).all():
                raise ValueError("Invalid ranker predictions")
            outputs.append(value)
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    def score(self, request: Request, items: np.ndarray) -> ScoredItems:
        users = np.full(len(items), request.user_id, dtype=np.int64)
        features = (
            {
                n: v.numpy()
                for n, v in self.features.gather(users, items, self.model.config.features).items()
            }
            if isinstance(self.features, FeatureState)
            else self.features.build(users, items)
        )
        return ScoredItems(items, self.predict(features))
