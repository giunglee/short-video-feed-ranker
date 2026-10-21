# KuaiRand temporal baselines

Start with the [WIP findings and review questions](kuairand_screen.md) for the
completed seed-42 screen. This page covers setup and implementation contracts.
For the reasoning behind each choice and the open decisions, read the
[data and protocol discussion](kuairand_data_review.md).

The KuaiRand study reuses `RecommendationPipeline`, its request/response contracts,
`ModelRanker`, the popularity and MF retrievers, `Trainer`, and the Deep model factory.
The data adapter, strict features and binary window evaluator are dataset-specific.
It does not change the KuaiRec protocol or reuse KuaiRec IDs, weights or profile meanings.

## Prepare and screen

Use the source checkout containing this module (`PYTHONPATH=src` if not installed).
Dataset paths and output paths are explicit; existing outputs are never overwritten.

```bash
python -m feed_ranker.experiments.kuairand_study prepare \
  --raw data/raw/kuairand_1k --out data/extract/kuairand_1k_temporal_v1
python -m feed_ranker.experiments.kuairand_study run \
  --data data/extract/kuairand_1k_temporal_v1 --out artifacts/kuairand_screen \
  --epochs 5 --mf-epochs 5 --seeds 42
```

Preparation defaults to all training users (`--users 0`); a fixed-seed user subset
is available for plumbing checks. Preparation is reused by every recipe and seed.
The seven explicit recipes in `configs/kuairand_study.json` comprise the reference
Deep[64,32], widths[32,16]/[128,64], and four fixed-width feature changes: remove
user ID, remove item ID, remove duration, add duration bands. This is not a Cartesian
hyperparameter grid. `--recipes reference` runs only the reference. `--seeds 42 43 44
45 46` supports matched confirmation after a justified comparison is selected.

## Contract

- Provided `long_view`, standard tab1 traffic, event-weighted binary cross entropy.
- Train before2022-04-29 00:00 UTC+8; validation until2022-05-02 00:00 UTC+8.
  Complete timestamp groups stay together. No test split is scored; dates were
  previously inspected in EDA, so validation remains exploratory.
- Remove identical complete records; preserve repeated events and repeated pairs
  across time. Select users using training traffic only. Users without training
  tab1 activity are outside this cohort; report actual training/validation counts.
- Fitted IDs and positive-duration item medians use training only. Missing duration
  and unknown IDs are explicit. OOV embedding rows are frozen zero. Duration bands
  are fixed at10/30/60 seconds with a separate missing code. No snapshots or
  event-outcome features are admitted. Removing duration leaves a zero constant
  for compatibility with Deep's numeric-input interface; it carries no information.
- Deep receives named `[B]` tensors and emits logits `[B,1]`. Reference inputs have
  18 dimensions: two8-dimensional ID embeddings plus log-duration/missing flag.
  The selected checkpoint minimizes validation logloss, earliest exact tie.
- BPR32 compares each user's positive-ever items with observed never-positive
  items, one uniformly sampled comparison per positive each epoch. Non-long-view
  is task-negative, not dislike. Fixed final MF epoch, no validation selection.
- Popularity counts distinct positive users. Cosine sums item similarities over
  each user's positive history, excluding self-similarity. The sparse factorization
  is algebraically equivalent to the dense model; scores use float64 and round at
  12 decimals for arithmetic-level ties. Historical float32 ties can order differently.
- Queries are user/validation-window pairs. Relevance means any positive outcome
  in that window; repeat items remain eligible. Candidates are warm *observed*
  validation items. These are conditional diagnostics, not full-catalog or next-event
  retrieval. Report warm coverage and all-target recall alongside conditional metrics.

All recipes use identical cached recall scores and M50/K10, with labels outside the
shared inference pipeline. Report candidate Recall50, full-warm-pool binary NDCG10,
its candidate oracle, support shortfalls and all-target recall. AUC, Brier, prevalence
baseline and warm/cold event cohorts diagnose the classifier independently.

## Artifacts and inference

Each study saves the config, preprocessing/source fingerprints, source snapshot,
per-seed recall weights and positive graph, full recall score caches, complete epoch
curves, best/last ranker checkpoints, keyed event predictions, pipeline traces and
per-user metrics. `summary.json` includes per-seed outcomes, mean/sample SD when
multiple seeds exist, and paired logloss differences; it never promotes a recipe.

`inference/kuairand.py:load_ranker` loads a self-contained ranker artifact with fitted
vocabularies, durations and recipe. `load_retrievers` reconstructs all three fitted
channels from a seed directory, checking its state hashes. Both work without raw
logs or evaluation labels. Runtime users supply a `Request` and explicit eligible
`CandidateScope`; catalog availability and seen-item policy remain caller responsibilities.
Score replay is a benchmark optimization, not the serving implementation.

## Scale boundary

Sparse cosine storage is linear in catalog size and positive graph edges; query work
still depends on overlapping histories. No approximate neighbors/history truncation
are introduced. MF remains dense-embedding AdamW. The current preprocessing and
training materialize selected events in RAM. The implementation is validated on1K;
full27K ingestion, optimizer memory and scoring cost require a separate measured
resource plan before migration. More users alone do not solve cold-item support.
