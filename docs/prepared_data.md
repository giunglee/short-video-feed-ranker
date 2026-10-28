# Prepared KuaiRand data

Preparation owns raw interpretation, splits, fitted transformations and lookup
state. It does not depend on model architecture or an experiment recipe.

`preprocessing/kuairand.py:prepare_kuairand()` writes train/validation event Parquet, a
versioned manifest, `feature-state.npz` and `feature-tables.json`. The manifest
records event dtypes, targets, source identity and split assumptions. Descriptive
feature provenance stays with the tables; there is no shared `FieldSpec` object.

`preprocessing/features.py:fit_entity_features()` builds one `EntityFeatureLookup`
from training events and declared snapshots. Private functions align snapshots,
encode categories and transform duration; there is no intermediate metadata-state
class. The runtime `data/` package imports no preparation code.

`data/prepared.py:PreparedDataset(path)` reads only the manifest. `read_split(name, columns)`
verifies and reads that split; `load_feature_state()` verifies only the fitted
tables. `validate()` audits all splits. This protocol exposes no test split.

`data/entity_features.py:EntityFeatureLookup` owns raw user/item keys, transformed column
arrays, their entity lookup assignments, fitted categorical vocabulary sizes and
provenance. Its cached indexes resolve raw IDs. `gather()` uses the same mappings
for training events and recommendation candidates; `save()`/`load()` preserve
those mappings without event-table access. Row zero is the unknown-entity row.
Train-median and snapshot durations remain distinct; cold-item metadata does not
grant membership in the warm candidate vocabulary.

The lookup stores precomputed attributes once per user/item, not once per event.
It contains categorical codes, not learned embeddings. Its `user_ids` and
`item_ids` are raw keys; each feature column has an extra row zero for unknown
entities. `entities` selects the user or item lookup, while `vocab_sizes` describes
the separate categorical code range consumed by an embedding layer.

For training items `[100, 300]` and snapshot items `[100, 200, 300]`, the lookup
has three item rows, but `item_code` is `[0, 1, 0, 2]` including the fallback row.
Item 200 can have a known author and duration while its item identity is unknown.
An entirely absent item uses the fallback row for every item feature. Raw category
zero remains a legitimate category; null, sentinel -124 and unseen categories
encode to zero. Category vocabularies are fitted only on training entities.

```python
# Preparation: fit once, independently of the model's feature recipe.
lookup = fit_entity_features(train, basic, profiles)

# Event loading and candidate scoring both use this operation:
features = lookup.gather(user_ids, item_ids, names=model.config.feature_names)
outputs = model(features)
```

`EventDataset` attaches labels to the gathered tensors; `ModelRanker` gathers
features for candidate items without labels. The model receives tensors only.
`RankerArtifact` packages the model and its matching lookup. This is a materialized
snapshot, not a live feature service: new metadata requires updating/rebuilding
the lookup. Fitted vocabularies and numeric transforms must remain consistent
with the model when doing so.

`data/instances.py:EventDataset` owns event metadata and target tensors and shares
one `EntityFeatureLookup` instance. Construction itself validates required/unique
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

The Python naming cleanup preserves the current on-disk filenames, NPZ keys and
values. Existing prepared-v3 and fitted-scorer-v2/v3 artifacts remain readable;
it does not require regenerating the completed regression study.
