"""KuaiRec pair holdout: big matrix supplies train/validation; small supplies test.

Category vocabularies fit training rows only. Repeated pair events stay together;
undated test rows are retained. Optional user attributes require an explicit feature
spec and prediction-time availability review. Completed datasets include source
fingerprints and are published only after every split and its manifest is written.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feed_ranker.artifacts import sha256
from feed_ranker.data.config import FeatureSpec
from feed_ranker.data.schema import (
    KEYS,
    LABEL,
    MATRIX_COLUMNS,
    PROTOCOL,
    KuaiRecInstance,
    validate_ids,
)


def split_pairs(frame: pd.DataFrame, seed: int, val_ratio: float) -> np.ndarray:
    """Return a row mask for random user-video-pair holdout within the big matrix.

    Policy:
    - Sort distinct (user_id, video_id) pairs, then shuffle using a fixed split seed.
    - Hold out floor(number_of_pairs * val_ratio) pairs, not that fraction of rows.
    - Keep every repeated event for a pair on the same side of the split.
    - Allow users and videos to appear on both sides; this is neither a cold-start
      split nor a temporal forecast, and it is not stratified by user or label.

    The default preparation uses 90/10 pairs and seed 2026. Training seeds do not
    change this split. The small matrix is reserved separately for final testing.
    Sorting makes pair membership independent of the input row order.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")

    pairs = frame[KEYS].drop_duplicates().sort_values(KEYS)
    validation_count = int(len(pairs) * val_ratio)
    if validation_count < 1 or validation_count >= len(pairs):
        raise ValueError("Need enough distinct pairs for nonempty train and validation")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))[:validation_count]
    held_out = pd.MultiIndex.from_frame(pairs.iloc[indices])
    return pd.MultiIndex.from_frame(frame[KEYS]).isin(held_out)


def deduplicate_signatures(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Retain the first complete event signature; unknown timestamps remain distinct."""
    signature = [*KEYS, "timestamp", "play_duration", "video_duration", LABEL]
    missing = set(signature) - set(frame.columns)
    if missing:
        raise ValueError(f"Signature deduplication requires columns: {sorted(missing)}")
    complete = frame[signature].notna().all(axis=1)
    repeated = complete & frame.duplicated(signature)
    return frame.loc[~repeated].copy(), int(repeated.sum())


class KuaiRecPreprocessor:
    """Prepare one dataset with a fixed split and train-fitted categorical mappings.

    User attributes are optional: matrix-only specs never read profile CSVs.
    """

    def __init__(
        self,
        raw_dir: Path,
        output: Path,
        spec: dict[str, Any],
        split_seed: int = 2026,
        val_ratio: float = 0.1,
    ) -> None:
        if type(split_seed) is not int or split_seed < 0:
            raise ValueError("split_seed must be a nonnegative integer")
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between zero and one")

        self.raw_dir = raw_dir
        self.output = output
        self.features = FeatureSpec.from_dict(spec)
        self.split_seed = split_seed
        self.val_ratio = val_ratio

    def run(self) -> dict[str, Any]:
        if self.output.exists():
            raise FileExistsError(f"Use a new protocol directory: {self.output}")
        big, small, sources = self._load_matrices()
        removal_counts = {}
        big, removal_counts["big_matrix"] = deduplicate_signatures(big)
        small, removal_counts["small_matrix"] = deduplicate_signatures(small)
        validation_mask = split_pairs(big, self.split_seed, self.val_ratio)
        splits = {
            "train": big.loc[~validation_mask].copy(),
            "val": big.loc[validation_mask].copy(),
            "test": small,
        }
        # IDs and duration already exist in the interaction matrices. Only other
        # explicitly selected inputs require the separate user-feature tables.
        if "duration_band" in self.features.required_columns:
            for frame in splits.values():
                frame["duration_band"] = duration_band(frame["video_duration"])
        user_columns = self.features.required_columns - set(MATRIX_COLUMNS) - {"duration_band"}
        if user_columns:
            user_features, user_sources = self._load_user_features(user_columns)
            sources.extend(user_sources)
            splits = {
                name: frame.merge(
                    user_features, on="user_id", how="left", sort=False, validate="many_to_one"
                )
                for name, frame in splits.items()
            }
        vocabulary = self._fit_category_vocabulary(splits["train"])
        metadata = self._build_metadata(vocabulary, sources)

        metadata["deduplication"] = {
            "policy": "complete_signature_keep_first",
            "removed_rows": removal_counts,
            "signature": [*KEYS, "timestamp", "play_duration", "video_duration", LABEL],
            "missing_timestamp": "retain",
        }
        return self._write_dataset(splits, vocabulary, metadata)

    def _load_matrices(self) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
        sources = [self.raw_dir / f"{name}_matrix.csv" for name in ("big", "small")]
        big, small = [
            pd.read_csv(path, usecols=[*MATRIX_COLUMNS, "timestamp", "play_duration"])
            for path in sources
        ]
        KuaiRecInstance.validate_frame(big, sources[0])
        KuaiRecInstance.validate_frame(small, sources[1])

        big_pairs = pd.MultiIndex.from_frame(big[KEYS])
        small_pairs = pd.MultiIndex.from_frame(small[KEYS])
        if big_pairs.isin(small_pairs).any():
            raise ValueError("Train/test user-video pair overlap")
        return big, small, sources

    def _load_user_features(self, required_columns: set[str]) -> tuple[pd.DataFrame, list[Path]]:
        """Load selected user attributes, such as user_onehot_feat0, by user_id.

        Prefer user_features_raw.csv; use user_features.csv for unresolved columns.
        These are snapshot attributes, not interaction aggregates. Their availability
        at prediction time must be reviewed before selecting them in the feature spec.
        """
        remaining = required_columns.copy()
        profiles: pd.DataFrame | None = None
        sources: list[Path] = []
        renamed_columns = {f"onehot_feat{i}": f"user_onehot_feat{i}" for i in range(18)}

        for filename in ("user_features_raw.csv", "user_features.csv"):
            path = self.raw_dir / filename
            if not remaining or not path.exists():
                continue
            frame = pd.read_csv(path).rename(columns=renamed_columns)
            selected = sorted(remaining & set(frame.columns))
            if not selected:
                continue

            subset = frame[["user_id", *selected]]
            validate_ids(subset, ["user_id"], path)
            if subset["user_id"].duplicated().any():
                raise ValueError(f"Duplicate profile IDs in {filename}")
            if profiles is None:
                profiles = subset
            else:
                profiles = profiles.merge(subset, on="user_id", how="outer", validate="one_to_one")
            remaining -= set(selected)
            sources.append(path)

        if remaining or profiles is None:
            raise ValueError(
                f"Unsupported features {sorted(remaining)}; "
                "this version excludes daily item aggregates"
            )
        return profiles, sources

    def _fit_category_vocabulary(self, train: pd.DataFrame) -> dict[str, list[str]]:
        """Fit the learned part of preprocessing on train only.

        Sorted categories define indices 1..K; zero is reserved for missing/unseen
        values. Dense conversion and explicitly configured crosses require no fit.
        """
        vocabulary: dict[str, list[str]] = {}
        for name in self.features.categorical:
            values = sorted(train[name].dropna().astype(str).unique().tolist())
            if not values:
                raise ValueError(f"No observed training categories for {name}")
            vocabulary[name] = values
        return vocabulary

    def _transform_features(
        self, frame: pd.DataFrame, vocabulary: dict[str, list[str]]
    ) -> pd.DataFrame:
        """Transform any split with the same train-fitted vocabulary and feature spec.

        Categories become int64 indices. Targets, dense inputs, and binary equality
        crosses become float32. Cross conditions use raw values, not encoded indices.
        This method does not learn new categories or select cross conditions.
        """
        encoded = frame[KEYS].rename(columns={name: f"raw_{name}" for name in KEYS}).copy()
        encoded[LABEL] = _to_float32(frame[LABEL], LABEL)

        for name, values in vocabulary.items():
            mapping = {value: index + 1 for index, value in enumerate(values)}
            # Index zero represents both missing and unseen categories.
            encoded[name] = frame[name].astype("string").map(mapping).fillna(0).astype("int64")
        counts = frame.groupby(KEYS)[LABEL].transform("size")
        encoded["pair_weight"] = (1.0 / counts).astype("float32")
        for name in self.features.dense:
            encoded[name] = _to_float32(frame[name], name, fill_missing=True)
        for name, conditions in self.features.crosses.items():
            matches = np.ones(len(frame), dtype=bool)
            for column, expected in conditions.items():
                condition = frame[column].astype("string").eq(str(expected)).fillna(False)
                matches &= condition.to_numpy(dtype=bool)
            encoded[name] = matches.astype("float32")
        return encoded

    def _build_metadata(
        self,
        vocabulary: dict[str, list[str]],
        sources: list[Path],
    ) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "preprocessing_version": "signature-dedup-v2",
            "split_policy": {
                "name": "random_pair_holdout",
                "group_by": KEYS.copy(),
                "validation_size": "floor(distinct_pairs * val_ratio)",
                "training_source": "big_matrix",
                "test_source": "small_matrix",
            },
            "split_seed": self.split_seed,
            "val_ratio": self.val_ratio,
            "spec": asdict(self.features),
            "derived_features": {
                "duration_band": "milliseconds; [0,5000], (5000,10000], (10000,20000], "
                "(20000,40000], (40000,60000], (60000,inf); missing/negative -> OOV"
            }
            if "duration_band" in self.features.required_columns
            else {},
            "vocab": vocabulary,
            "preprocessor_sha256": sha256(Path(__file__)),
            "sources": {path.name: sha256(path) for path in sources},
            "limitations": [
                "Pair holdout, not a temporal forecast",
                "Profile snapshot timing requires review",
                "Small labels were used in historical experiments",
                "Distinct event signatures retained; no label aggregation",
            ],
        }

    def _write_dataset(
        self,
        splits: dict[str, pd.DataFrame],
        vocabulary: dict[str, list[str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish a complete directory and return its manifest without mutating inputs."""
        metadata = {**metadata, "splits": {}}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{self.output.name}-", dir=self.output.parent
        ) as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            for name, frame in splits.items():
                encoded = self._transform_features(frame, vocabulary)
                path = staging / f"{name}.parquet"
                encoded.to_parquet(path, index=False)
                metadata["splits"][name] = {
                    "rows": len(encoded),
                    "unknown_fraction": {
                        column: float(encoded[column].eq(0).mean())
                        for column in self.features.categorical
                    },
                    "sha256": sha256(path),
                }
            manifest = staging / "meta.json"
            manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            if self.output.exists():
                raise FileExistsError(f"Use a new protocol directory: {self.output}")
            staging.rename(self.output)
        return metadata


# Internal value validation and transformations.


def duration_band(values: pd.Series) -> pd.Series:
    """Fixed duration bins in milliseconds; boundaries match the descriptive EDA."""
    return pd.cut(
        values,
        [0, 5000, 10000, 20000, 40000, 60000, np.inf],
        labels=["0–5s", "5–10s", "10–20s", "20–40s", "40–60s", ">60s"],
        include_lowest=True,
    ).astype("string")


def _to_float32(values: pd.Series, name: str, fill_missing: bool = False) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name}: expected numeric values") from error
    if fill_missing:
        numeric = numeric.fillna(0)
    if not np.isfinite(numeric).all() or (numeric.abs() > np.finfo(np.float32).max).any():
        raise ValueError(f"{name}: values must be finite and representable as float32")
    return numeric.astype("float32")


# Command-line entry point.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/kuairec"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Explicit JSON with categorical, dense, crosses",
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preprocessor = KuaiRecPreprocessor(
        raw_dir=args.raw_dir,
        output=args.output,
        spec=json.loads(args.features.read_text()),
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
    )
    metadata = preprocessor.run()
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
