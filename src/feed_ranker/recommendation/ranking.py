"""Frozen model ranking using train-fitted feature lookups; no evaluation imports."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from feed_ranker.artifacts import sha256
from feed_ranker.data.recommendation import KuaiRecFeatures
from feed_ranker.models.ranking import Recipe, build_architecture, feature_config
from feed_ranker.recommendation.contracts import Request, ScoredItems


@torch.no_grad()
def predict(
    model: nn.Module, features: dict[str, np.ndarray], batch_size: int = 8192
) -> np.ndarray:
    if batch_size < 1 or not features:
        raise ValueError("Require features and a positive batch size")
    n = len(next(iter(features.values())))
    if any(len(v) != n for v in features.values()):
        raise ValueError("Feature batch lengths differ")
    model.eval()
    outputs = []
    for start in range(0, n, batch_size):
        batch = {
            name: torch.from_numpy(v[start : start + batch_size].copy())
            for name, v in features.items()
        }
        value = model(batch).reshape(-1).cpu().numpy()
        if len(value) != min(batch_size, n - start) or not np.isfinite(value).all():
            raise ValueError("Invalid ranker predictions")
        outputs.append(value)
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


class ModelRanker:
    """A model and its feature builder are replaced together, outside the pipeline."""

    def __init__(self, model: nn.Module, features: KuaiRecFeatures, artifact_id: str) -> None:
        self.model, self.features, self.artifact_id = model.cpu().eval(), features, artifact_id

    def score(self, request: Request, items: np.ndarray) -> ScoredItems:
        features = self.features.build(np.full(len(items), request.user_id, dtype=np.int64), items)
        return ScoredItems(items, predict(self.model, features))


def load_deep_ranker(
    checkpoint: Path, expected_sha256: str, ranking_data: Path, duration_path: Path
) -> tuple[ModelRanker, dict[str, Any]]:
    if sha256(checkpoint) != expected_sha256:
        raise ValueError("Checkpoint differs from the frozen finalist")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    meta = saved["metadata"]
    recipe = Recipe(**meta["recipe"])
    if recipe.architecture != "deep" or recipe.feature_set != "profiles" or meta["target_cap"] != 2:
        raise ValueError("Require frozen capped Deep/profile recipe")
    if saved["training"]["selection_metric"] != "user_xauc":
        raise ValueError("Preserve the standalone user-XAUC checkpoint selector")
    if sha256(ranking_data / "meta.json") != meta["data_manifest_sha256"]:
        raise ValueError("Frozen ranking data manifest mismatch")
    metadata = json.loads((ranking_data / "meta.json").read_text())
    if sha256(ranking_data / "train.parquet") != metadata["splits"]["train"]["sha256"]:
        raise ValueError("Prepared ranking training data changed")
    features = KuaiRecFeatures.load(metadata, ranking_data / "train.parquet", duration_path)
    model = build_architecture(feature_config(metadata, recipe), recipe, saved["training"]["seed"])
    model.load_state_dict(saved["model"])
    return ModelRanker(model, features, expected_sha256), saved
