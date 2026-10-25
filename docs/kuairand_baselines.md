# KuaiRand temporal baselines

Start with the [matched-seed merge and feature results](kuairand_confirmation.md).
The [initial five-epoch screen](kuairand_screen.md) remains a historical result. This page covers setup and implementation contracts.
For the reasoning behind each choice and the open decisions, read the
[data and protocol discussion](kuairand_data_review.md).

The KuaiRand study reuses `RecommendationPipeline`, its request/response contracts,
`ModelRanker`, the popularity and MF retrievers, `Trainer`, and the Deep model factory.
The data adapter, strict features and binary window evaluator are dataset-specific.
It does not change the KuaiRec protocol or reuse KuaiRec IDs, weights or profile meanings.

## Prepare and reproduce the initial screen

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
Deep [64,32], widths[32,16]/[128,64], and four fixed-width feature changes: remove
user ID, remove item ID, remove duration, add duration bands. This is not a Cartesian
hyperparameter grid. `--recipes reference` runs only the reference. `--seeds 42 43 44
45 46` supports matched confirmation after a justified comparison is selected.

## Original strict-screen contract

- Provided `long_view`, standard tab 1 traffic, event-weighted binary cross entropy.
- Train before 2022-04-29 00:00 UTC+8; validation until 2022-05-02 00:00 UTC+8.
  Complete timestamp groups stay together. No test split is scored; dates were
  previously inspected in EDA, so validation remains exploratory.
- Remove identical complete records; preserve repeated events and repeated pairs
  across time. Select users using training traffic only. Users without training
  tab 1 activity are outside this cohort; report actual training/validation counts.
- Fitted IDs and positive-duration item medians use training only. Missing duration
  and unknown IDs are explicit. OOV embedding rows are frozen zero. Duration bands
  are fixed at 10/30/60 seconds with a separate missing code. No snapshots or
  event-outcome features are admitted. Removing duration leaves a zero constant
  for compatibility with Deep's numeric-input interface; it carries no information.
- Deep receives named `[B]` tensors and emits logits `[B,1]`. Reference inputs have
  18 dimensions: two 8-dimensional ID embeddings plus log-duration/missing flag.
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

## Current AUC-selected feature and merging study

The current protocol uses a 30-epoch cap and maximum validation global ROC AUC,
with the earliest exact tie. Global AUC, macro within-user AUC, logloss/Brier,
warm/cold cohorts and all funnel metrics are reported at that same checkpoint.
The strict control here includes duration bands, unlike the initial screen's base
reference. Seeds42–46 are matched; the primary stability summary uses43–46 and
keeps the exploration-exposed seed 42 separate.

```bash
PYTHONPATH=src python scripts/prepare_kuairand_metadata.py \
  --data data/extract/kuairand_1k_temporal_v1 \
  --raw data/raw/kuairand_1k --out artifacts/kuairand_metadata.pt
PYTHONPATH=src python -m feed_ranker.experiments.kuairand_study run \
  --data data/extract/kuairand_1k_temporal_v1 --out artifacts/kuairand_side \
  --config configs/kuairand_side_study.json --side-features artifacts/kuairand_metadata.pt \
  --epochs 30 --mf-epochs 5 --seeds 42 43 44 45 46 \
  --selection-metric roc_auc --include-merges
PYTHONPATH=src python -m feed_ranker.experiments.kuairand_study run \
  --data data/extract/kuairand_1k_temporal_v1 --out artifacts/kuairand_without_anonymous \
  --config configs/kuairand_anonymous_ablation.json --side-features artifacts/kuairand_metadata.pt \
  --epochs 30 --mf-epochs 5 --seeds 42 43 44 45 46 \
  --selection-metric roc_auc --include-merges --recall-from artifacts/kuairand_side
```

- **Isolated groups:** author ID; basic metadata duration replacing the train-median
  source; 18 anonymous user categorical fields; all three combined. The follow-up
  author+basic-duration arm removes anonymous fields from that combination.
- **Availability assumptions:** author identity and basic duration are assumed fixed
  and available at scoring. Anonymous user snapshots have unknown meanings and
  unverified historical availability. These are explicit sensitivity arms, not
  additions silently relabeled as strict temporal features.
- **Vocabulary fitting:** authors use only training-item mappings; anonymous
  categories use only training users. Null/-124/unseen values map to 0; legitimate
  raw category 0 remains distinct. Basic duration and author lookup support can
  extend to cold items, but this does not establish catalog eligibility.
- **Merging:** `--include-merges` adds equal-share round-robin and equal-weight RRF
  with constant 60. Each source contributes its top 50; the output budget remains 50.
  Candidate IDs and recall/oracle denominators are fixed across feature arms within
  each seed. Comparing rankers never refits or retunes the recall channels.
- **Reuse:** `--recall-from` checks data/seed/MF-budget and file hashes before reuse.
  `--resume-from` resumes into a new directory, preserving optimizer, RNG, complete
  history and earliest best checkpoint; only the epoch cap may change.
- **Final window:** May 2–8 is accepted for eventual final model evaluation but remains
  unscored. Its EDA exposure is disclosed. Validation-seed stability is not a new
  time-period holdout, and the missing ablation was chosen after prior results.

The metadata preparation command reproduces the fitted lookup tensors/vocabularies
used in this study. Original controller metadata has equivalent assumption wording;
serialized artifact hashes may differ because that descriptive manifest differs.
Do not swap artifacts in a continuing run: resume deliberately checks their hash.

## Artifacts and inference

Each study saves the config, preprocessing/source fingerprints, source snapshot,
per-seed recall weights and positive graph, full recall score caches, complete epoch
curves, best/last ranker checkpoints, keyed event predictions, pipeline traces and
per-user metrics. `summary.json` includes per-seed outcomes, mean/sample SD when
multiple seeds exist, and paired logloss/AUC differences; it never promotes a recipe.

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
training materialize selected events in RAM. The implementation is validated on 1K;
full 27K ingestion, optimizer memory and scoring cost require a separate measured
resource plan before migration. More users alone do not solve cold-item support.
