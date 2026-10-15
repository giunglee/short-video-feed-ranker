"""Frozen IDs/duration ranker integration in its verified training-source process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.recall import RecallData
from feed_ranker.training.recall_evaluation import evaluate


def candidate_features(data: RecallData) -> pd.DataFrame:
    """Build labels-free raw candidate features using training-only item medians."""
    duration = pd.read_parquet(data.directory / "item_duration.parquet").set_index("video_id")
    videos = data.items[data.pools.items]
    values = duration["median"].reindex(videos).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing/invalid training-derived candidate duration")
    return pd.DataFrame(
        {"user_id": data.users[data.pools.users], "video_id": videos, "video_duration": values}
    )


# Only frozen ranking imports are used in this subprocess. The source tree and
# metadata are verified before loading weights; the current working tree is untouched.
FROZEN_SCORER = r"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from feed_ranker.experiments.wide_deep_kuairec import build_model
from feed_ranker.artifacts import sha256
import feed_ranker.experiments.wide_deep_kuairec as ranking_module
checkpoint, data_dir, input_path, output_path = map(Path, sys.argv[1:])
torch.set_num_threads(4)
saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
meta = saved["metadata"]
if sha256(data_dir / "meta.json") != meta["data_manifest_sha256"]:
    raise ValueError("Frozen ranker manifest mismatch")
for path, expected in meta["source_sha256"].items():
    if sha256(Path(path)) != expected:
        raise ValueError("Frozen ranker source mismatch: " + path)
if str(Path(ranking_module.__file__).resolve()) not in meta["source_sha256"]:
    raise ValueError("Imported ranker is not from the recorded source snapshot")
model, dataset = build_model(data_dir, meta["model_name"], meta["hidden"], meta["dropout"], meta["use_ids"])
if dataset.meta["spec"]["categorical"] != ["user_id", "video_id"] or dataset.meta["spec"]["dense"] != ["video_duration"] or dataset.meta["spec"]["crosses"]:
    raise ValueError("Integration currently supports only verified IDs/duration rankers")
model.load_state_dict(saved["model"])
model.eval()
# Check raw-ID encoding and prediction parity against prepared validation rows.
# This checks feature transforms, without using held-out watch outcomes as inputs.
check = pd.read_parquet(data_dir / "val.parquet").iloc[:8192]
direct = {}
for name in ("user_id", "video_id"):
    mapping = {str(value): i+1 for i,value in enumerate(dataset.meta["vocab"][name])}
    encoded = check["raw_" + name].astype("string").map(mapping).fillna(0).to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(encoded, check[name].to_numpy())
    direct[name] = torch.from_numpy(encoded)
direct["video_duration"] = torch.tensor(check.video_duration.to_numpy(), dtype=torch.float32)
prepared = dataset._pack(check)["feature"]
with torch.no_grad():
    torch.testing.assert_close(model(direct), model(prepared), rtol=0, atol=0)
raw = pd.read_parquet(input_path)
features = {"video_duration": raw.video_duration.to_numpy(dtype=np.float32)}
for name in ("user_id", "video_id"):
    mapping = {str(value): i+1 for i,value in enumerate(dataset.meta["vocab"][name])}
    features[name] = raw[name].astype("string").map(mapping).fillna(0).to_numpy(dtype=np.int64)
outputs = []
with torch.no_grad():
    for start in range(0,len(raw),8192):
        batch = {k:torch.from_numpy(v[start:start+8192]) for k,v in features.items()}
        result = model(batch).reshape(-1).numpy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite ranker prediction")
        outputs.append(result)
np.save(output_path,np.concatenate(outputs) if outputs else np.empty(0,dtype=np.float32))
(output_path.parent / "ranker_verification.json").write_text(json.dumps({
    "feature_parity_rows": len(check), "prediction_parity": "exact",
    "checkpoint_epoch_zero_based": saved["epoch"],
    "imported_source": str(Path(ranking_module.__file__).resolve()),
    "verified_source_files": len(meta["source_sha256"]),
}))
"""


def run(
    data_path: Path,
    runs: list[Path],
    ranking_data: Path,
    checkpoint: Path,
    frozen_source: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    data = RecallData(data_path)
    output.mkdir(parents=True)
    features = candidate_features(data)
    features_path = output / "candidate_features.parquet"
    features.to_parquet(features_path, index=False)
    observed = pd.read_parquet(data_path / "val.parquet").set_index(["user_id", "video_id"])
    observed_duration = observed.video_duration.reindex(
        pd.MultiIndex.from_frame(features[["user_id", "video_id"]])
    ).to_numpy()
    shift = np.abs(observed_duration - features.video_duration.to_numpy())
    write_json(
        output / "duration_audit.json",
        {
            "pairs": len(shift),
            "different_pairs": int((shift > 0).sum()),
            "difference_over_1s_pairs": int((shift > 1000).sum()),
            "absolute_difference_ms_quantiles": np.quantile(shift, [0, 0.5, 0.9, 0.99, 1]).tolist(),
            "policy": "Training-only median lookup; observed duration used only for this audit",
        },
    )
    env = dict(os.environ, PYTHONPATH=str(frozen_source.resolve()))
    subprocess.run(
        [
            sys.executable,
            "-c",
            FROZEN_SCORER,
            str(checkpoint.resolve()),
            str(ranking_data.resolve()),
            str(features_path.resolve()),
            str((output / "rank_scores.npy").resolve()),
        ],
        env=env,
        check=True,
    )
    rank_scores = np.load(output / "rank_scores.npy")
    reports = {}
    for directory in runs:
        config = json.loads((directory / "config.json").read_text())
        if config["data_manifest_sha256"] != sha256(data_path / "meta.json"):
            raise ValueError("Retriever was evaluated on different data")
        scores = np.load(directory / "scores.npy")
        report, per_user, _ = evaluate(data, scores, rank_scores)
        name = config["method"]
        per_user.to_csv(output / f"{name}_per_user.csv", index=False)
        reports[name] = report
    # A common full-pool ranking reference distinguishes retrieval from ranker loss.
    full, per_user, _ = evaluate(data, rank_scores, rank_scores)
    reports["full_pool_ranker"] = full
    per_user.to_csv(output / "full_pool_ranker_per_user.csv", index=False)
    result = {
        "checkpoint_sha256": sha256(checkpoint),
        "ranker_data_manifest_sha256": sha256(ranking_data / "meta.json"),
        "recall_data_manifest_sha256": sha256(data_path / "meta.json"),
        "feature_file_sha256": sha256(features_path),
        "source_sha256": sha256(Path(__file__)),
        "methods": reports,
        "note": "Train-derived median duration; no snapshot inputs or test scoring. "
        "Full-pool reference ranks all eligible videos for top-10, "
        "its reported candidate oracle is still bounded by each M.",
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--ranking-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.data, args.runs, args.ranking_data, args.checkpoint, args.frozen_source, args.output
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
