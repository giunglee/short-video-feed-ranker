# Materialized training data and tensor loading

Preparation writes complete scalar-feature examples. Loading selects columns and
batches tensors; it does not join entity tables or fit transformations.

## Files and ownership

`preprocessing/kuairand.py:prepare_kuairand()` reads complete source records,
deduplicates before projection, selects the cohort and temporal splits, fits
features on training data, and writes `train.parquet` and `validation.parquet`.
Every row contains raw event/user/item identities, time, `warm_item`, all supported
features, and the configured named targets. A model may select fewer features
without rebuilding these files. Unknown IDs remain categorical code zero.

The manifest records feature/metadata dtypes, target names, vocabulary sizes,
feature descriptions, split bounds, raw/snapshot hashes, and output hashes.
Feature, target and metadata roles are disjoint. The current preparation protocol
declares only train and validation; May 2–8 remains unscored. The reader can read
an explicitly declared test split, but never discovers one implicitly.

`data/parquet.py:ParquetReader(path)` reads only the manifest at construction.
`read_split(name, columns)` checks the requested file hash and column types,
then returns a DataFrame using Parquet column projection. It needs no fitted
feature arrays. `validate()` is the full audit of all splits and fitted state:
finite values, category ranges, binary targets, event identity, cohort flags and
temporal bounds. Preparation runs that audit before returning; `validate-data`
can run it again explicitly. It is not repeated by each batch.

`data/dataset.py:EventDataset` accepts an already loaded DataFrame and explicit
feature/target names. It validates the selected inputs, copies them into owned
tensors once, and returns indexed examples or batches. Categorical inputs are
int64; dense inputs and targets are float32. Raw IDs/time are int64 metadata;
`warm_item` is boolean metadata. Features and labels stay in separate dictionaries.
No state lookup, transformation, or file access occurs in `__getitem__`.

```python
from pathlib import Path
from feed_ranker.data.parquet import ParquetReader
from feed_ranker.data.dataset import EventDataset, make_loader

reader = ParquetReader(Path("runs/prepared"))
features = ("user_code", "item_code", "train_duration_log1p")
targets = ("long_view",)
columns = [*reader.metadata_dtypes, *features, *targets]
events = reader.read_split("train", columns)
dataset = EventDataset(events, features, targets)
loader = make_loader(dataset, batch_size=1024, seed=42)
```

`make_loader()` uses standard PyTorch `BatchSampler`/`DataLoader` with vectorized
tensor indexing. NumPy seed-plus-epoch permutation, isolated loader RNG, repeatable
validation passes, and final partial batches are preserved. Training/scoring owns
device transfer. Model inputs are resolved from manifest types/vocabulary sizes;
the manifest contains no embedding dimensions, model presets or loss settings.

## Fitted state and new candidates

`preprocessing/features.py:FeatureState` retains frozen `user_features` and
`item_features` separately. Each group follows its sorted raw IDs, with fallback
row zero. There is no separate feature-to-entity dictionary. `fit_entity_features()`
constructs this state; `transform(user_ids, item_ids, names)` returns NumPy feature
columns used both to materialize training rows and to score new candidate pairs.
Candidate scoring reads no event table or outcomes and never refits state.

Lookup positions and embedding codes remain distinct. With training items
`[100, 300]` and snapshot items `[100, 200, 300]`, item lookup row 2 describes
item 200, while its identity code is zero. Its metadata may still be known.
Metadata presence does not make it part of the warm candidate vocabulary.

`feature-state.npz` and `feature-tables.json` persist grouped arrays, vocabulary
sizes, and descriptions. `ParquetReader.load_feature_state()` checks these files
independently for model export, recall fitting, or candidate scoring. Ordinary
event loading does not call it. `RankerArtifact` packages this state with matching
weights/configuration. Runtime scoring imports the reusable feature module, but
does not import raw preparation or training jobs.

## Resource and compatibility limits

Tensor memory is proportional to events times selected columns. Materialization
duplicates entity attributes across events; Parquet compression may reduce disk
cost, not the in-memory tensor size. Source reading is chunked, but selected
cohort events, output frames and tensor datasets are in memory. This is not a
streaming implementation or a 27K memory/throughput claim.

The new format is `kuairand-prepared-v4`. Old event-only prepared-v3 files must be
regenerated into a fresh directory. They are rejected explicitly, never joined
silently during loading. Existing source, result and prepared directories are not
overwritten. Changed data identity also prevents resuming an old run against new
files; current-format CPU continuation remains supported.

New model exports use `fitted-scorer-v4` with grouped feature definitions. Existing
fitted-scorer-v2/v3 artifacts still load: only their stored flat feature ownership
description is converted to the two groups. Feature arrays and weights are not
refitted or changed. No old Dataset/lookup import aliases are retained.

Current train-fitted duration and snapshot assumptions are preserved. A storage
change does not establish historical snapshot availability or add time-varying
features. History construction and negative sampling require a separate concrete
experiment; no future sequence format is introduced here.
