# Current KuaiRand workflow

The core interfaces are preprocessing, runtime data access, models, training,
fitted artifacts, evaluation and recommendation. The local job and external study compose them. Prior PRs
supply behavior references, not structural constraints. Standard-tab-1 temporal
`long_view` semantics remain unchanged; May 2–8 is outside this workflow.

## Interface index

These are the current interfaces, in data-flow order. Read each constructor's
state and its public methods before following the private implementation.

- **Preparation functions** — `preprocessing/kuairand.py:prepare_kuairand()` owns
  raw interpretation, cohort/split selection and writing prepared outputs.
  `preprocessing/features.py:fit_entity_features()` fits entity features using training
  events and explicitly supplied snapshots. Neither selects model architectures.
- **`ParquetReader`** (`data/parquet.py`) — state: path, manifest, column roles/dtypes,
  vocabulary sizes, declared splits and identity.
  Methods: `read_split(name, columns)`, `load_feature_state()`, `validate()`.
- **`FeatureState` (dataclass, `preprocessing/features.py`)** — state: raw
  `user_ids`/`item_ids`, separate user/item feature arrays, vocabulary sizes,
  descriptions and cached indexes. Methods: `transform(user_ids, item_ids, names)`,
  `save(directory)`, `load(directory)`. Preparation and candidate scoring use this
  state; training examples already contain transformed values.
- **`EventDataset(Dataset)`** (`data/dataset.py`) — state: selected feature tensors,
  target tensors and event metadata, loaded once from materialized rows.
  Methods: `__len__()`, `__getitem__(index)`; no entity lookups during batching.
  `make_loader(dataset, batch_size, seed)` supplies the standard PyTorch DataLoader.
- **`DeepModel(nn.Module)`** — state: DeepConfig, embeddings and MLP.
  Method: `forward(features)` returns named logits, plus inherited PyTorch
  `parameters()`, `state_dict()`, `train()` and `eval()`.
- **`BPRMF(nn.Module)`** — state: user/item embeddings and updated-item support.
  Methods: `forward(users, positive, negative)` returns BPR margins;
  `score_pairs(users, items)` returns dot-product scores. The separate
  `SparseCosineCooccurrence` in the same recall module holds positive histories, counts and sparse indexes;
  `score(user, items)` returns fitted cosine scores.
- **`Objective(nn.Module)`** — state: TaskLossConfig records.
  `forward(outputs, batch)` applies output/target mappings, masks, example weights
  and task weights; returns scalar loss plus reduction diagnostics. It lives in
  `training/objectives.py` and is supplied to the trainer independently of the model.
- **`Trainer`** — state: supplied model, optimizer, Objective and TrainConfig.
  `fit(train_loader, validation_loader, evaluator, checkpoint_dir=..., run_identity=...)`
  owns updates, validation, history and checkpoint selection. It leaves the model
  at the final epoch; the job explicitly loads selected weights before export.
  `training/checkpoint.py` owns plain checkpoint save/load and CPU continuation.
- **`ObservedSampler`** — state: supported-user mask and flattened positive/negative
  histories; `sample(rng)` returns aligned user/positive/negative tensors.
  `training/bpr.py:fit_mf(model, sampler, optimizer, epochs=..., seed=...)` updates
  supplied state and returns epoch history. BPR keeps its fixed-final-epoch policy;
  `training/jobs.py:fit_recall_job()` owns initialization, construction and export.
- **`PredictionEvaluator`** — state: output-to-target and optional mask mappings.
  Methods: `collect(model, batches)` and `evaluate(model, batches)`.
  **`Predictions` (dataclass)** stores aligned labels, logits, users, warm flags and
  validity. `evaluate_predictions(records)` computes metrics without a model.
- **Fitted artifact dataclasses** — `RankerArtifact` holds model plus FeatureState;
  `MFArtifact` holds model, vocabularies and supported users; `CosineArtifact` and
  `PopularityArtifact` hold fitted scores/state plus vocabularies. Each carries
  identity and provenance. `load_artifact()` returns these records; direct
  `save_ranker()`, `save_mf()`, `save_cosine()` and `save_popularity()` functions
  accept fitted state without requiring an artifact object first.
- **Recommendation adapters** — `ModelRanker` owns RankerArtifact and inference
  batching; `score(request, items)` returns ScoredItems. Each fitted retriever owns
  its scorer, ID lookup and support state; `retrieve(request, scope, limit)` returns
  RetrievalResult. `load_ranker()` lives with ModelRanker in `rankers.py`;
  `load_retriever()` lives with fitted retrieval adapters in `retrievers.py`.
  **`MergedRetriever`** owns sources, policy and explicit `source_depth` (integer
  or None); it implements the same `retrieve()` interface.
  `RecommendationPipeline` owns retriever, ranker, budget and k;
  `recommend(request, scope)` returns Recommendation.
- **`WindowBenchmark`** — state: validation queries, relevant item IDs and coverage;
  `cache(retriever, path)` freezes source rankings and `evaluate(results)` calculates
  funnel metrics. `evaluation/funnel.py` also owns `open_benchmark()` and
  `evaluate_pipeline()`, the local composition and report writing.

Configuration records are `DeepConfig` (dense/sparse inputs and Deep dimensions),
`TaskLossConfig` (output/target/mask/weight mapping), `TrainConfig` (epoch loop and
selection), `OptimizerConfig` (job optimizer settings), and `TrainingJobConfig` (local
assembly). Trainer records the actual supplied optimizer class, group settings
and parameter order for continuation. They configure behavior; they do not execute it. `Batch` is a TypedDict,
not a runtime class. `Retriever` and `Ranker` are structural Protocols, not base
classes that models must inherit. Request/response dataclasses are `Request`,
`CandidateScope`, `ScoredItems`, `RetrievalResult` and `Recommendation`.

## KuaiRand data audit

`preprocessing/kuairand_audit.py` supplies descriptive EDA functions, not runtime performance
profiling or fitted preprocessing. `read_csv_sample()` reads a sorted-file prefix
and reports truncation; zero requests all rows. `profile_fields()` reports dtype,
nulls, sentinel values, invalid known values, distinct values and distributions.
`profile_events()` reports complete duplicates, traffic, labels, repeated pairs,
timestamp ties and train/validation warm/cold coverage. The external
`experiments/eda_kuairand.py` orchestrates these functions and writes the report/plot.

A bounded prefix is not a representative sample; its observed training vocabulary
can understate warm coverage relative to the full dataset. The validity rules are
KuaiRand-specific and descriptive, not a replacement for preparation validation.
These functions do not select model inputs, fit transforms or score a final model.

## Module layout

```text
preprocessing/ kuairand.py, features.py, kuairand_audit.py
data/          parquet.py, dataset.py
models/        ranking.py, recall.py, components/mlp.py
training/      objectives.py, trainer.py, checkpoint.py, bpr.py, optimizer.py, jobs.py
artifacts.py   fitted scorer records and validated save/load
persistence.py JSON writes, hashes and run/source provenance
evaluation/    metrics.py, predictions.py, funnel.py
recommendation/ rankers.py, retrievers.py, merged.py, merging.py, pipeline.py
                contracts.py, ordering.py, replay.py, traces.py
```

Preprocessing writes materialized splits; ordinary Parquet/tensor reads import no
feature preparation. Fitted-state loading and scoring reuse preprocessing/features.py,
but never import the raw preparation entry point. Models own architecture and scoring,
without data or training imports.
Training composes models with supervision and optimization. Artifact loading uses
model definitions and fitted data state without constructing training objectives.
Runtime recommendation consumes those artifacts; evaluation owns held-out labels.

BPR's observed-negative sampling defines training comparisons, so `histories()`,
`ObservedSampler` and `fit_mf()` live together in `training/bpr.py`. Ordinary
example ordering/batching stays with the Dataset/DataLoader in `data/dataset.py`.
Jobs own construction and export; `optimizer.py` retains the shared decay policy.

## Responsibilities and state

- **Data:** [materialized datasets](prepared_data.md) contain complete feature/target
  rows. ParquetReader owns file access/validation; EventDataset/DataLoader own tensor
  construction and batching. Preprocessing owns fitting and feature assembly.
- **Model:** `models/ranking.py:DeepModel` owns its dense/sparse input configuration,
  embeddings and MLP. `forward(features)` returns named logits. Training objectives
  independently resolve labels, masks and weights. It never receives
  a dataset schema. Sparse vocabulary sizes are resolved from ETL once; model
  configuration owns embedding widths and layers. The input mapping is immutable;
  `to_dict()` produces an independent serialization value.
- **Trainer:** `training/trainer.py:Trainer` consumes concrete, re-iterable loaders
  and a prediction evaluator. `EpochBatchSampler` supplies the existing NumPy
  seed-plus-epoch ordering without consuming dropout RNG. The optional
  `train_eval_loader` records frozen-model training metrics.
  Trainer writes only epoch history and checkpoints. Jobs own configuration,
  source provenance, final results and inference export. There are no loader,
  validation or metric-storage callbacks. `checkpoint.py` validates both last
  and best checkpoint contracts and history before restoring CPU continuation.
- **Evaluator:** `evaluation/predictions.py:PredictionEvaluator` owns output/label
  interpretation. `collect(model, batches)` produces aligned `Predictions`;
  `evaluate_predictions(records)` calculates metrics without a model or loader.
  `evaluate(model, batches)` combines the two for training-time validation. Full
  prediction AUC, user exclusions and warm/cold cohorts are preserved. Training
  update loss and optional frozen-model training metrics are distinct records.
- **Fitted packages:** `artifacts.py` has concrete `RankerArtifact`,
  `MFArtifact`, `CosineArtifact` and `PopularityArtifact` records. Required state
  is visible in each loaded record. Direct writers validate their inputs and
  atomically publish a complete directory without reconstructing a model.
  Ranker loading and saving validate model/feature agreement.
  Fitted packages exclude optimizer/continuation state and executable objectives.
  Training jobs record objective configuration in run/checkpoint provenance.
  `persistence.py` contains generic storage helpers, not scorer reconstruction.
- **Recommendation:** `ModelRanker` owns a fitted model, its matching feature
  tables as a validated `RankerArtifact`, plus inference batch size. Its caller
  supplies candidates, never another
  feature selection. `RecommendationPipeline` continues to own retrieval,
  candidate merging, ranking and traces. Evaluation labels remain outside it.

The model can be trained from already-produced instances without calling the job:

```python
model = DeepModel(model_config).to(train_config.device)
optimizer = build_optimizer(model, optimizer_config)
objective = Objective(task_configs)
evaluator = PredictionEvaluator({"long_view": "long_view"})
trainer = Trainer(model, optimizer, objective, train_config)
result = trainer.fit(
    train_loader,
    validation_loader,
    evaluator,
    checkpoint_dir=run_directory / "checkpoints",
    run_identity=input_and_model_identity,
)
load_model_weights(model, Path(result["best_checkpoint"]))
save_ranker(run_directory / "model", model, feature_state, provenance)
```

`training/jobs.py` is the optional local composition, including path handling,
input resolution and both ranking/recall export. External `experiments/kuairand.py` adds comparisons.
Their configuration uses `loading` for batch sizes and `training` for optimization
and selection. The experiment feature list is resolved once into the model's
concrete dense/sparse configuration and saved with the fitted package.

## Evaluation layers

- `metrics.py` contains array-to-number calculations: binary classification metrics,
  equal-user AUC, and candidate recall/NDCG/oracle NDCG. It reads no data and runs no model.
- `predictions.py` collects aligned event predictions from a model/loader, then
  calculates metrics with explicit validity and warm/cold cohorts. Saved prediction
  records can be evaluated without a model; global AUC uses the full population.
- `funnel.py` implements the KuaiRand observed-window protocol: define query scopes
  and positive-ever pair relevance, freeze complete source rankings, run retrieval
  and ranking, and write coverage/shortfall and recommendation metrics. Candidate
  recall uses all retrieved candidates; NDCG evaluates top10. Oracle NDCG can
  reorder only retrieved candidates, with the full warm-positive denominator.
  `all_target_recall` also includes positives outside the warm catalog in its denominator.

Event prediction quality and retrieval/ranking quality answer different questions;
better event AUC alone does not establish better final recommendations.

## Retrieval depth and score meaning

`MergedRetriever(channels, policy, source_depth=...)` is the single fusion adapter.
Policies are `round_robin`, `rrf` and `anchored`. A positive integer caps each source
before fusion; None requests the full supported scope. `supported_count` is the
union available after this depth choice, before the final limit. Channel identity,
depth and policy settings enter the artifact hash; source order does not.

The CLI and study retain their top-50 source depth and `rr_all`/`rrf_all` reporting
labels. Candidate IDs, order, support and funnel metrics are unchanged. Merged
trace scores now consistently mean RRF contributions for RRF, descending allocation
ranks for round-robin/anchored. The former bounded adapter used zero-based negative
ranks for both policies; these trace values and merger identity strings change.
Neither representation is an engagement prediction. Saved older traces/results
remain historical evidence; this change does not rewrite them.

## Commands and operational limits

Install with `python -m pip install -e '.[dev]'` on Python 3.10+. Run from the
checkout for external studies; package operations also work from another directory.

```sh
python -m feed_ranker.cli prepare --raw data/raw/kuairand --out runs/prepared --users 32
python -m feed_ranker.cli validate-data --data runs/prepared
python -m feed_ranker.cli train --config configs/kuairand_job.json --out runs/deep
python -m feed_ranker.cli fit-recall --data runs/prepared --out runs/recall --epochs 5 --seed 42
python -m feed_ranker.cli evaluate --data runs/prepared --ranker runs/deep/model --recall runs/recall/models/popularity runs/recall/models/cosine runs/recall/models/mf --out runs/evaluation
```

These are executable examples, not a new experiment budget. The job template uses
one epoch as a smoke configuration, explicitly selects maximum validation AUC with
earliest ties, and is not a promoted model or a replacement for a declared study
budget. Set paths, feature selection, checkpoint policy and budget before a real
run. Add `--basic` and `--profiles` to preparation to materialize the existing
snapshot feature families; select `metadata_duration_*`, `author_code` and the
named anonymous fields explicitly. Their historical availability remains unverified.

Resume with `train ... --resume-from runs/deep --out runs/deep_extended` and a
larger epoch cap. Other training/objective/model/data changes fail. The selected
model is exported separately from optimizer/RNG/continuation state. Original
directories are never overwritten. CPU epoch-boundary continuation is tested;
GPU-exact and MF continuation are not supported.

New ranking runs have one predictable layout:

```text
run/
  config.json                resolved inputs, objectives and training settings
  provenance.json            source revision/diff and data identity
  history.json               complete epoch curve and diagnostics
  result.json                selected epoch, metrics and export path
  checkpoints/
    last.pt                  model, optimizer and CPU RNG for continuation
    best.pt                  selected model weights and selection metadata
  model/
    manifest.json            format, output meaning, provenance and checksums
    config.json              inference model configuration
    weights.pt               plain model state_dict
    features/
      feature-state.npz
      feature-tables.json
```

Recall runs use the same run-level files, `checkpoints/mf.pt`, and
`models/{mf,cosine,popularity}/`; each model directory contains its own manifest,
configuration and fitted state. MF selection remains fixed final epoch.
New training checkpoints carry `training-checkpoint-v1`; pre-refactor checkpoints
require their original training code. Existing inference exports remain readable.

For label-free scoring, use `recommend` with the same `--ranker`, `--recall` and
`--out` arguments plus `--requests requests.json`. Requests are a JSON list of
objects such as `{"query_id":"q1","user_id":1,"items":[10,20]}`. The caller owns
candidate eligibility. This operation reads no event table or label. Both
recommendation and evaluation save label-free traces; evaluation adds per-user
metrics and full-pool denominators. Evaluation uses the existing top10 protocol.

Run EDA independently using the [raw-audit command](raw_audit.md).
For matched comparisons, use the [study guide](kuairand_baselines.md) and explicit
recipe configurations. The runner owns frozen recall reuse, seed groups and
comparison reporting; the package owns preparation, models, training and metrics.

The current package replaces the older execution paths. KuaiRec scalar training,
its feature framework and capped-watch-ratio evaluators, old KuaiRand preparation
and study runners, and their artifact writers/converters are removed. Historical
reports, results and source remain available at the
[pinned pre-refactor revision](https://github.com/giunglee/short-video-feed-ranker/tree/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5).
Use that revision in a separate checkout to reproduce those studies.

Historical KuaiRand v1/v2 ranker and recall readers are also removed. Use the
pinned revision to read those artifacts. The active package maintains Deep, BPR,
sparse cosine and popularity through current fitted artifacts; no unused cross
features or compatibility wrappers remain. The frozen synthetic recall reference
checks original scores, ordering/support and BPR training against current code.
No old training-checkpoint continuation is promised; current named-instance CPU
continuation is tested.

Validation uses public synthetic fixtures, including uninterrupted/resumed CPU
training and actual artifact-only multi-channel scoring. These checks establish
software correctness, not real-data quality, 27K memory readiness, or online
latency. Those measurements require separately scoped runs.

Prepared data now uses `kuairand-prepared-v4`, containing complete example rows.
Regenerate older event-only prepared data into a fresh directory. New exports use
`fitted-scorer-v5`, with configuration and feature state in separate files. The loader also
accepts fitted-scorer-v2/v3/v4 without refitting or constructing training objectives.
Model configuration, weights, feature state and output meaning are still checked.
Historical KuaiRand v1/v2 bundles are a different format and still require the
pinned checkout. Dependent study callers must adopt the new reader/Dataset imports
and materialized-data interface, concrete Trainer arguments and direct export
functions; no forwarding aliases are retained.

Deep models concatenate sparse inputs in sorted-name order followed by configured
dense order. That changed during the larger rearchitecture, so its retraining is
not claimed to reproduce historical initialization trajectories. This ownership
cleanup preserves current feature values, predictions, optimization and continuation;
it changes where operations live and removes objective construction from inference.
