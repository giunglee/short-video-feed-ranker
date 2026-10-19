"""Add train-fitted attributes to frozen train/validation rows without reading test."""

from __future__ import annotations

import ast
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranker.artifacts import sha256, write_json
from feed_ranker.data.kuairec_preprocessor import validate_ids

BASE_FEATURES = ("user_id", "video_id", "age_range", "follow_user_num_range", "duration_band")
PROFILE_ADDITIONS = ("user_active_degree", "fans_user_num_range", "register_days_range")
BASE_COLUMNS = [
    *BASE_FEATURES,
    "video_duration",
    "watch_ratio",
    "raw_user_id",
    "raw_video_id",
    "pair_weight",
]


def parse_item_categories(value: str) -> list[str]:
    """Read an unordered set of public category IDs; reject malformed values."""
    try:
        values = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"Invalid item category list: {value!r}") from error
    if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values):
        raise ValueError(f"Expected nonnegative category IDs: {value!r}")
    return [str(v) for v in sorted(set(values))]


def prepare_feature_data(base: Path, raw: Path, output: Path) -> dict[str, Any]:
    """Preserve row order, labels, weights and existing encodings exactly.

    New vocabularies use only entities present in training. Public attributes are
    snapshot annotations; this transform does not establish historical availability.
    The output intentionally contains no test split or test file.
    """
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((base / "meta.json").read_text())
    if original["preprocessing_version"] != "signature-dedup-v2":
        raise ValueError("Expected deduplicated base data")
    for split in ("train", "val"):
        if sha256(base / f"{split}.parquet") != original["splits"][split]["sha256"]:
            raise ValueError(f"Frozen {split} data changed")

    entities = pd.read_parquet(base / "train.parquet", columns=["raw_user_id", "raw_video_id"])
    users, videos = entities.raw_user_id.unique(), entities.raw_video_id.unique()
    del entities
    profile_path, category_path = raw / "user_features_raw.csv", raw / "item_categories.csv"
    profiles = pd.read_csv(profile_path, usecols=["user_id", *PROFILE_ADDITIONS])
    items = pd.read_csv(category_path, usecols=["video_id", "feat"])
    for frame, key, path in (
        (profiles, "user_id", profile_path),
        (items, "video_id", category_path),
    ):
        validate_ids(frame, [key], path)
        if frame[key].duplicated().any():
            raise ValueError(f"Duplicate {key} in {path.name}")
    profiles = profiles.set_index("user_id")
    categories = items.set_index("video_id").feat.map(parse_item_categories)
    vocab = deepcopy(original["vocab"])
    lookups: dict[str, pd.Series] = {}
    for name in PROFILE_ADDITIONS:
        values = profiles[name].astype("string")
        vocab[name] = sorted(values.reindex(users).dropna().unique().tolist())
        if not vocab[name]:
            raise ValueError(f"No training values for {name}")
        mapping = {value: i + 1 for i, value in enumerate(vocab[name])}
        lookups[name] = values.map(mapping).fillna(0).astype("int64")
    vocab["item_categories"] = sorted(set(categories.reindex(videos).dropna().explode().dropna()))
    if not vocab["item_categories"]:
        raise ValueError("No training item categories")
    mapping = {value: i + 1 for i, value in enumerate(vocab["item_categories"])}
    category_ids = categories.map(lambda values: [mapping.get(v, 0) for v in values] or [0])
    metadata = deepcopy(original)
    metadata.update(
        vocab=vocab,
        spec={
            "categorical": [*BASE_FEATURES, *PROFILE_ADDITIONS, "item_categories"],
            "dense": ["video_duration"],
            "crosses": {},
        },
        splits={},
        parent_manifest_sha256=sha256(base / "meta.json"),
        feature_preparer_sha256=sha256(Path(__file__)),
        added_source_sha256={p.name: sha256(p) for p in (profile_path, category_path)},
        evaluation_scope="big-only train/validation; no test file",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        for split in ("train", "val"):
            target = staging / f"{split}.parquet"
            writer = None
            rows = 0
            try:
                for batch in pq.ParquetFile(base / f"{split}.parquet").iter_batches(
                    batch_size=100_000, columns=BASE_COLUMNS
                ):
                    frame = batch.to_pandas()
                    for name, lookup in lookups.items():
                        frame[name] = frame.raw_user_id.map(lookup).fillna(0).astype("int64")
                    frame["item_categories"] = frame.raw_video_id.map(category_ids).map(
                        lambda values: values if isinstance(values, list) else [0]
                    )
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(target, table.schema)
                    writer.write_table(table)
                    rows += len(frame)
            finally:
                if writer is not None:
                    writer.close()
            if rows != original["splits"][split]["rows"]:
                raise ValueError(f"{split} row count changed")
            metadata["splits"][split] = {"rows": rows, "sha256": sha256(target)}
        write_json(staging / "meta.json", metadata)
        staging.rename(output)
    return metadata
