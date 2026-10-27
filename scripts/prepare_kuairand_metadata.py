"""Prepare training-fitted KuaiRand metadata lookups for explicit snapshot studies.

Field definitions: https://kuairand.com/ . This does not establish historical
availability of the supplied snapshots; read the emitted assumption manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.kuairand import load_prepared
from feed_ranker.data.metadata import ANONYMOUS, MetadataEncoding


def prepare_metadata(data: Path, raw: Path, output: Path) -> None:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    _, _, encoding, _ = load_prepared(data)
    basic_path, user_path = raw / "video_features_basic_1k.csv", raw / "user_features_1k.csv"
    basic = pd.read_csv(basic_path, usecols=["video_id", "author_id", "video_duration"])
    profiles = pd.read_csv(user_path, usecols=["user_id", *ANONYMOUS])
    fitted = MetadataEncoding.fit(encoding.users, encoding.items, basic, profiles)
    manifest = {
        "version": "kuairand-side-features-v1",
        "assumptions": {
            "author": "Video-author mapping assumed fixed and available before scoring.",
            "basic_duration": "Fixed-media duration assumed available at scoring, including cold items.",
            "anonymous": "Snapshot sensitivity: semantics and historical availability are unverified.",
            "catalog": "Metadata lookup coverage does not establish retrieval eligibility.",
        },
        "included": {"basic": ["author_id", "video_duration"], "users": list(ANONYMOUS)},
        "unknown_policy": "null/-124/unseen ->0; legitimate raw zero is a fitted category",
        "source_sha256": {p.name: sha256(p) for p in (basic_path, user_path)},
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "training_users": len(fitted.users),
        "lookup_videos": len(fitted.videos),
        "author_vocabulary": len(fitted.author_vocabulary),
        "anonymous_cardinality": {
            name: len(vocab)
            for name, vocab in zip(ANONYMOUS, fitted.anonymous_vocabularies, strict=True)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoding": fitted.to_state(),
            "data_manifest_sha256": manifest["data_manifest_sha256"],
            "manifest": manifest,
        },
        output,
    )
    write_json(output.with_suffix(".json"), manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare_metadata(args.data, args.raw, args.out)


if __name__ == "__main__":
    main()
