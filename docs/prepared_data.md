# Prepared KuaiRand data

`feed_ranker.data.prepared.prepare_kuairand` produces versioned event Parquet
tables, a manifest, a JSON field schema and validated NPZ entity arrays. Numeric
transforms are persisted once. Train-median and snapshot durations have distinct
names; metadata for cold items does not grant warm-candidate eligibility.

`open_prepared(path)` reads only the manifest. `read_split(name, columns)` verifies
and reads that split; `load_feature_state()` verifies only the fitted state.
`validate()` audits all splits. Only train/validation are exposed by this protocol.

`EventDataset` owns event IDs and configured target tensors and shares one fitted
`FeatureState`. `make_loader` gathers full index batches without per-row collation,
preserves the NumPy permutation and final partial batch, and isolates loader RNG.
The feature state has one row per entity plus OOV, not one row per event/recipe.
Memory is O(events + entity fields); shuffled loading also holds O(events) NumPy
indices. Batch lists and transformed gathers are allocated one batch at a time.
Use one state instance across recipes; callers choose batch/device policy.

The current reader chunks source CSV ingestion but retains selected cohort events
in memory. This is not a streaming or 27K scalability claim. Tests use public
synthetic fixtures; real 1K memory/throughput measurement remains a separately
scoped operational check. No dataset run is needed to test storage integrity,
target projection, split boundaries, OOV behavior or train/scoring parity.
