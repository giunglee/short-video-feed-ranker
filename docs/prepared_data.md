# Prepared KuaiRand data

Preparation owns raw interpretation, splits, fitted transformations and lookup
state. It does not depend on model architecture or an experiment recipe.

`preprocessing/kuairand.py:prepare_kuairand()` writes train/validation event Parquet, a
versioned manifest, `feature-state.npz` and `feature-tables.json`. The manifest
records event dtypes, targets, source identity and split assumptions. Descriptive
feature provenance stays with the tables; there is no shared `FieldSpec` object.

`preprocessing/features.py:fit_features()` constructs the fitted lookup tables using
training events and declared snapshots. The runtime `data/` package imports no
preparation code; inference can load these tables independently.

`data/prepared.py:PreparedDataset(path)` reads only the manifest. `read_split(name, columns)`
verifies and reads that split; `load_feature_state()` verifies only the fitted
tables. `validate()` audits all splits. This protocol exposes no test split.

`data/feature_tables.py:FeatureTables` owns raw user/item keys, transformed column
arrays, their entity lookup assignments, fitted categorical vocabulary sizes and
provenance. Its cached indexes resolve raw IDs. `gather()` uses the same mappings
for training events and recommendation candidates; `save()`/`load()` preserve
those mappings without event-table access. Row zero is the unknown-entity row.
Train-median and snapshot durations remain distinct; cold-item metadata does not
grant membership in the warm candidate vocabulary.

`data/instances.py:EventDataset` owns event metadata and target tensors and shares
one `FeatureTables` instance. Construction itself validates required/unique
columns, role separation, integer nonnegative metadata, unique event IDs and binary
outcomes; it does not rely on the job to make instances valid. Its feature list
comes from the model's resolved input configuration. `make_loader()` uses PyTorch `BatchSampler`/`DataLoader`,
vectorized gathers, NumPy seed-plus-epoch ordering and a separate loader RNG.
The same validation loader can be iterated repeatedly. Final partial batches are
retained. Features and targets are separate named tensor dictionaries; raw IDs,
time and cohort flags are metadata.

Memory is O(events + entity fields), with O(events) ordering indices. Batch index
lists and transformed gathers are allocated one batch at a time. There are no
per-recipe entity-table copies or user-by-item Cartesian tables. Source ingestion
is chunked, but selected cohort events remain in memory. This is not a streaming
or 27K readiness claim; real 1K memory/throughput measurement is separately scoped.

The current format is `kuairand-prepared-v3`. Earlier unmerged schema-based
prepared directories must be regenerated into a fresh directory. Existing source
and result directories are never overwritten.
