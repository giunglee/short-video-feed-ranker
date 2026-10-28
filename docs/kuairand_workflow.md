# Current KuaiRand workflow

The core interfaces are preprocessing, runtime data access, models, training,
fitted artifacts, evaluation and recommendation. The local job composes them; PR #10 adds the matched-seed study. Prior PRs
supply behavior references, not structural constraints. Standard-tab-1 temporal
`long_view` semantics remain unchanged; May 2–8 is outside this workflow.

## Interface index

These are the current interfaces, in data-flow order. Read each constructor's
state and its public methods before following the private implementation.

- **Preparation functions** — `preprocessing/kuairand.py:prepare_kuairand()` owns
  raw interpretation, cohort/split selection and writing prepared outputs.
  `preprocessing/features.py:fit_features()` fits entity features using training
  events and explicitly supplied snapshots. Neither selects model architectures.
- **`PreparedDataset`** — state: path, manifest, targets, event dtypes, identity.
  Methods: `read_split(name, columns)`, `load_feature_state()`, `validate()`.
- **`FeatureTables` (dataclass)** — state: raw user/item keys, columns, entity
  assignments, vocabulary sizes, descriptions, cached lookup indexes.
  Methods: `gather(users, items, names)`, `save(directory)`, `load(directory)`.
  Preparation constructs this state; training and inference share its runtime
  lookup. `_duration_columns(values)` is a private pure transform in preprocessing.
- **`EventDataset(Dataset)`** — state: shared FeatureTables, selected feature names,
  target tensors and event metadata. Methods: `__len__()`, `__getitem__(index)`.
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
- **`Trainer`** — state: model, TrainConfig, loss callable, optimizer.
  Main method: `fit(train_batches, validate, directory, metadata, ...)`, with
  optional metric recording, frozen-model training evaluation and resume input.
  Epoch history and selection are local to a fit; private methods handle updates,
  checkpoint writing and restoration.
- **`ObservedSampler`** — state: supported-user mask and flattened positive/negative
  histories; `sample(rng)` returns aligned user/positive/negative tensors.
  `training/bpr.py:fit_mf(model, sampler, optimizer, epochs=..., seed=...)` updates
  supplied state and returns epoch history. BPR keeps its fixed-final-epoch policy;
  `training/jobs.py:fit_recall_job()` owns initialization, construction and export.
- **`PredictionEvaluator`** — state: output-to-target and optional mask mappings.
  Methods: `collect(model, batches)` and `evaluate(model, batches)`.
  **`Predictions` (dataclass)** stores aligned labels, logits, users, warm flags and
  validity. `evaluate_predictions(records)` computes metrics without a model.
- **Fitted artifact dataclasses** — `RankerArtifact` holds model plus FeatureTables;
  `MFArtifact` holds model, vocabularies and supported users; `CosineArtifact` and
  `PopularityArtifact` hold fitted scores/state plus vocabularies. Each carries
  identity and provenance. `save_artifact()`/`load_artifact()` own persistence.
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
`TaskLossConfig` (output/target/mask/weight mapping), `TrainConfig` (optimization and
selection), `OptimizerConfig` (optimizer settings), and `TrainingJobConfig` (local
assembly). They configure behavior; they do not execute it. `Batch` is a TypedDict,
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
preprocessing/ kuairand.py, features.py, metadata.py, kuairand_audit.py
data/          prepared.py, feature_tables.py, instances.py
models/        ranking.py, recall.py, components/mlp.py
training/      objectives.py, trainer.py, bpr.py, optimizer.py, jobs.py
artifacts.py   fitted scorer records and validated save/load
persistence.py JSON/checkpoint writes, hashes and run/source provenance
evaluation/    metrics.py, predictions.py, funnel.py
recommendation/ rankers.py, retrievers.py, merged.py, merging.py, pipeline.py
                contracts.py, ordering.py, replay.py, traces.py
```

Preprocessing depends on runtime data types; runtime data access imports no raw
preparation. Models own architecture and scoring, without data or training imports.
Training composes models with supervision and optimization. Artifact loading uses
model definitions and fitted data state without constructing training objectives.
Runtime recommendation consumes those artifacts; evaluation owns held-out labels.

BPR's observed-negative sampling defines training comparisons, so `histories()`,
`ObservedSampler` and `fit_mf()` live together in `training/bpr.py`. Ordinary
example ordering/batching stays with the Dataset/DataLoader in `data/instances.py`.
Jobs own construction and export; `optimizer.py` retains the shared decay policy.

## Responsibilities and state

- **Data:** [prepared datasets](prepared_data.md) own event access and fitted
  feature lookup; preprocessing owns fitting those transformations. `EventDataset`/`DataLoader` produce named feature, target and
  metadata tensors independently of training.
- **Model:** `models/ranking.py:DeepModel` owns its dense/sparse input configuration,
  embeddings and MLP. `forward(features)` returns named logits. Training objectives
  independently resolve labels, masks and weights. It never receives
  a dataset schema. Sparse vocabulary sizes are resolved from ETL once; model
  configuration owns embedding widths and layers. The input mapping is immutable;
  `to_dict()` produces an independent serialization value.
- **Trainer:** `training/trainer.py:Trainer` owns model/optimizer execution state.
  `fit()` consumes an epoch-to-batches function, explicit validation callback,
  checkpoint destination and resume identity. It has no raw-data, preparation,
  feature-selection or fitted-model export logic. An optional `record_metrics`
  callback receives epoch records for another persistence destination; local
  history remains available for continuation. The same Trainer owns optimizer
  updates, epoch history, checkpoint selection and CPU continuation; no second trainer or lifecycle-forwarding class is retained.
  No database service is implemented.
- **Evaluator:** `evaluation/predictions.py:PredictionEvaluator` owns output/label
  interpretation. `collect(model, batches)` produces aligned `Predictions`;
  `evaluate_predictions(records)` calculates metrics without a model or loader.
  `evaluate(model, batches)` combines the two for training-time validation. Full
  prediction AUC, user exclusions and warm/cold cohorts are preserved. Training
  update loss and optional frozen-model training metrics are distinct records.
- **Fitted packages:** `artifacts.py` has concrete `RankerArtifact`,
  `MFArtifact`, `CosineArtifact` and `PopularityArtifact` records. Required state
  is visible in each constructor instead of optional fields with hidden valid
  combinations. `RankerArtifact` alone validates model/ETL input agreement.
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
model = DeepModel(model_config)
objective = Objective(task_configs)
evaluator = PredictionEvaluator({"long_view": "long_view"})
trainer = Trainer(model, train_config, loss=objective)
result = trainer.fit(
    train_batches,  # epoch -> iterable of named tensor batches
    lambda current: evaluator.evaluate(current, validation_loader),
    run_directory,
    {"resume_contract": input_and_model_identity},
    record_metrics=store_epoch_record,
)
```

`training/jobs.py` is the optional local composition, including path handling,
input resolution and both ranking/recall export. The dependent [PR #10](https://github.com/giunglee/short-video-feed-ranker/pull/10) adds comparisons.
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
python -m feed_ranker.cli evaluate --data runs/prepared --ranker runs/deep/model --recall runs/recall/popularity runs/recall/cosine runs/recall/mf --out runs/evaluation
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

For label-free scoring, use `recommend` with the same `--ranker`, `--recall` and
`--out` arguments plus `--requests requests.json`. Requests are a JSON list of
objects such as `{"query_id":"q1","user_id":1,"items":[10,20]}`. The caller owns
candidate eligibility. This operation reads no event table or label. Both
recommendation and evaluation save label-free traces; evaluation adds per-user
metrics and full-pool denominators. Evaluation uses the existing top10 protocol.

Run EDA independently using the [raw-audit command](raw_audit.md).
The study runner, recipe configurations and historical KuaiRand results live in
the dependent PR #10; merge this architecture first.

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

Prepared data remains `kuairand-prepared-v3`. New exports use `fitted-scorer-v3`,
which omits the former top-level training-task definition. The loader also accepts
`fitted-scorer-v2`: those task definitions are historical metadata, unnecessary
for scoring. Model configuration, weights, feature state and output meaning are
still checked. Historical KuaiRand v1/v2 bundles are a different format and still
require the pinned checkout. Earlier unmerged prepared formats require regeneration.

Deep models concatenate sparse inputs in sorted-name order followed by configured
dense order. That changed during the larger rearchitecture, so its retraining is
not claimed to reproduce historical initialization trajectories. This ownership
cleanup preserves current feature values, predictions, optimization and continuation;
it changes where operations live and removes objective construction from inference.
