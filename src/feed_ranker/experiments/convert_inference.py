"""Explicit migration of historical ranking checkpoints and recall runs.

Preserve saved weights and embedding row order; reconstruct deterministic baseline
statistics from training histories. No optimization or checkpoint selection occurs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.recall import RecallData
from feed_ranker.evaluation.kuairec import verify_split_membership
from feed_ranker.experiments.ranking_recipe import Recipe, feature_config
from feed_ranker.features.config import FeatureConfig
from feed_ranker.features.kuairec import fit_candidate_features
from feed_ranker.inference.artifacts import LoadedRanker, TrainingProvenance, load_ranker
from feed_ranker.inference.retrievers import CooccurrenceRetriever, MFRetriever, PopularityRetriever
from feed_ranker.models.ranking_factory import ModelSpec, build_model
from feed_ranker.models.recall import BPRMF
from feed_ranker.training.export import export_ranker, export_retriever
from feed_ranker.training.recall import fit_baseline


def convert_checkpoint(
    checkpoint: Path, ranking_data: Path, duration_path: Path, directory: Path
) -> None:
    """Convert architecture-study checkpoints; new checkpoints carry resolved specs."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    if sha256(ranking_data / "meta.json") != meta["data_manifest_sha256"]:
        raise ValueError("Frozen ranking data manifest mismatch")
    metadata = json.loads((ranking_data / "meta.json").read_text())
    if sha256(ranking_data / "train.parquet") != metadata["splits"]["train"]["sha256"]:
        raise ValueError("Prepared ranking training data changed")
    if "feature_config" in meta and "model_spec" in meta:
        config, spec = (
            FeatureConfig.from_dict(meta["feature_config"]),
            ModelSpec(**meta["model_spec"]),
        )
    else:
        # Explicit migration of the historical architecture-study format only.
        # Feature-study artifacts need their resolved config, not a guessed preset.
        if "features" in meta:
            raise ValueError(
                "Legacy feature-study checkpoint needs resolved feature_config/model_spec"
            )
        recipe = Recipe(**meta["recipe"])
        config, spec = (
            feature_config(metadata, recipe),
            ModelSpec(recipe.architecture, recipe.hidden),
        )
    features = fit_candidate_features(
        config, metadata, ranking_data / "train.parquet", duration_path
    )
    with torch.random.fork_rng(devices=[]):
        model = build_model(config, spec)
    model.load_state_dict(saved["model"], strict=True)
    provenance = TrainingProvenance(
        sha256(checkpoint),
        meta["data_manifest_sha256"],
        saved["training"]["seed"],
        saved["epoch"],
        meta.get("target_cap"),
        saved["training"]["selection_metric"],
    )
    export_ranker(directory, model, spec, features, provenance)


def convert_retriever(data: RecallData, run: Path, directory: Path) -> None:
    """Migrate a historical recall run without changing its IDs or support policy."""
    config = json.loads((run / "config.json").read_text())
    data_hash = sha256(data.directory / "meta.json")
    if config["data_manifest_sha256"] != data_hash:
        raise ValueError("Retriever data mismatch")
    kind = config["method"]
    source_hash, epoch = sha256(run / "config.json"), None
    retriever: PopularityRetriever | CooccurrenceRetriever | MFRetriever
    if kind in ("popularity", "cooccurrence"):
        retriever = fit_baseline(data.training, kind, source_hash)
    elif kind == "mf":
        saved = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        if saved["config"] != config:
            raise ValueError("MF checkpoint/config mismatch")
        with torch.random.fork_rng(devices=[]):
            model = BPRMF(len(data.users), len(data.items), config["dimension"])
        model.load_state_dict(saved["model"], strict=True)
        source_hash, epoch = sha256(run / "best.pt"), saved.get("epoch")
        retriever = MFRetriever(model, data.users, data.items, data.training.supported, source_hash)
    else:
        raise ValueError("Unknown retriever")
    export_retriever(
        directory, retriever, TrainingProvenance(source_hash, data_hash, config.get("seed"), epoch)
    )


@dataclass(frozen=True)
class FrozenDeepReference:
    """Historical benchmark eligibility policy, separate from generic model loading."""

    seed: int
    epoch: int
    recipe: Recipe
    checkpoint_sha256: str

    @classmethod
    def read(cls, checkpoint: Path, expected_sha256: str) -> "FrozenDeepReference":
        if sha256(checkpoint) != expected_sha256:
            raise ValueError("Checkpoint differs from the frozen finalist")
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        recipe = Recipe(**saved["metadata"]["recipe"])
        if (
            recipe.architecture != "deep"
            or recipe.feature_set != "profiles"
            or saved["metadata"]["target_cap"] != 2
        ):
            raise ValueError("Require frozen capped Deep/profile recipe")
        if saved["training"]["selection_metric"] != "user_xauc":
            raise ValueError("Preserve the standalone user-XAUC checkpoint selector")
        return cls(saved["training"]["seed"], saved["epoch"], recipe, expected_sha256)

    def verify_predictions(
        self, loaded: LoadedRanker, ranking_data: Path, reference_trace: Path
    ) -> dict[str, int | float]:
        if loaded.manifest.provenance.source_sha256 != self.checkpoint_sha256:
            raise ValueError("Loaded ranker differs from benchmark reference")
        check = next(
            pq.ParquetFile(ranking_data / "val.parquet").iter_batches(batch_size=8192)
        ).to_pandas()
        raw = pd.DataFrame(
            {
                "user_id": check.raw_user_id,
                "video_id": check.raw_video_id,
                "video_duration": check.video_duration,
            }
        )
        direct = loaded.features.encode(raw)
        for name, values in direct.items():
            np.testing.assert_array_equal(
                values, check[name].to_numpy(), err_msg=f"Feature parity: {name}"
            )
        predictions = loaded.ranker.predict(direct)
        trace = next(
            pq.ParquetFile(reference_trace).iter_batches(batch_size=len(check))
        ).to_pandas()
        np.testing.assert_array_equal(trace.raw_user_id, check.raw_user_id)
        np.testing.assert_array_equal(trace.raw_video_id, check.raw_video_id)
        np.testing.assert_allclose(predictions, trace.prediction, atol=1e-6, rtol=1e-6)
        return {
            "feature_parity_rows": len(check),
            "prediction_parity_max_abs": float(
                np.abs(predictions - trace.prediction.to_numpy()).max()
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert historical experiment runs once, before inference/evaluation"
    )
    sub = parser.add_subparsers(dest="kind", required=True)
    rank = sub.add_parser("ranker")
    for name in ("checkpoint", "ranking-data", "recall-data", "output"):
        rank.add_argument(f"--{name}", type=Path, required=True)
    rank.add_argument("--checkpoint-sha256", required=True)
    rank.add_argument(
        "--frozen-deep-trace",
        type=Path,
        help="Also enforce the historical Deep finalist and prediction parity",
    )
    recall = sub.add_parser("retriever")
    for name in ("data", "run", "output"):
        recall.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("Require positive thread count")
    torch.set_num_threads(args.threads)
    if args.kind == "retriever":
        convert_retriever(RecallData(args.data), args.run, args.output)
        return
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint checksum mismatch")
    reference = (
        FrozenDeepReference.read(args.checkpoint, args.checkpoint_sha256)
        if args.frozen_deep_trace
        else None
    )
    hashes = verify_split_membership(RecallData(args.recall_data), args.ranking_data)
    convert_checkpoint(
        args.checkpoint, args.ranking_data, args.recall_data / "item_duration.parquet", args.output
    )
    report = {
        "checkpoint_sha256": args.checkpoint_sha256,
        "input_hashes": hashes,
        "test_evaluated": False,
    }
    if reference is not None:
        report.update(
            reference.verify_predictions(
                load_ranker(args.output), args.ranking_data, args.frozen_deep_trace
            )
        )
    write_json(args.output / "conversion.json", report)


if __name__ == "__main__":
    main()
