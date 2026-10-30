# KuaiRand study on the current architecture

Review [the module interfaces](kuairand_workflow.md#interface-index) first, then
[`experiments/kuairand.py`](../experiments/kuairand.py). The runner composes prepared
data, training jobs, fitted artifacts and evaluation. It defines no model,
preprocessor, optimizer, loss or metric implementation.

The shared architecture from #30 is merged into `main` at `3d96c3d`. This study
is rebased onto that revision and uses its materialized-data, explicit training
and structured artifact interfaces. Package source remains unchanged.

## Prepare and run

These commands reproduce the migrated workflow.
Run from the checkout with the package installed. Choose fresh output directories;
preparation and training never overwrite existing runs.

```sh
python -m feed_ranker.cli prepare \
  --raw data/raw/kuairand_1k --out runs/kuairand_prepared --users 0 \
  --basic data/raw/kuairand_1k/video_features_basic.csv \
  --profiles data/raw/kuairand_1k/user_features.csv
python -m feed_ranker.cli validate-data --data runs/kuairand_prepared
OMP_NUM_THREADS=2 python -m experiments.kuairand \
  --config configs/kuairand_side_study.json --out runs/kuairand_side
```

Supply the actual downloaded CSV paths. Snapshot mappings are materialized during
preparation; no separate metadata conversion command or legacy artifact is needed.
The configured `data` path must match the preparation output. Raw snapshots are
never passed into models or inference.

The [six-arm configuration](../configs/kuairand_side_study.json) explicitly lists:

- `reference`: train-median duration, missing flag and duration band, user/item IDs.
- `add_author`: strict reference plus author ID.
- `basic_duration`: replace all three train-duration columns with basic metadata duration.
- `anonymous_user`: strict reference plus the 18 anonymous categorical fields.
- `combined`: basic duration, author, anonymous fields and user/item IDs.
- `author_duration`: the combined recipe without anonymous fields.

All use Deep [64,32], embedding width 8, no dropout, unweighted `long_view` BCE,
AdamW at 0.001 with zero weight decay, batch size 4096 and a 30-epoch cap. Select
maximum validation global AUC, earliest exact tie. Report every metric at that
checkpoint. Seeds 43–46 are primary; 42 is separately reported replication.
Explicit paired contrasts include combined minus author_duration (anonymous
increment) and author_duration minus basic_duration (author increment).

[`configs/kuairand_study.json`](../configs/kuairand_study.json) preserves the initial
seven width/feature questions, a five-epoch cap and minimum-logloss selection.
It uses seed 42 and needs no snapshots. Its `no_duration` arm now has only the two
ID embeddings: the old dummy zero numeric input is removed. This changes its
parameterization and is not historical retraining parity.

## Fixed protocol and outputs

Standard-tab-1 traffic, training before April 29 and validation April 29–May 1
(UTC+8), identical-record deduplication and train-fitted vocabularies are owned by
preparation. Repeated events/pairs remain. May 2–8 is never scored. Snapshot
historical availability remains unverified; metadata arms are sensitivity analyses.

Recall fits popularity, sparse cosine and BPR once per seed (five MF epochs,
fixed-final-epoch selection). Full supported rankings are frozen and hashed.
All recipes reuse those candidates. Single channels, equal-share round-robin and
RRF60 use M50/K10; merged channels take each source's top 50. Candidates are warm
items observed in the validation window, with repeat items eligible. These are
exposure-conditional diagnostics, not full-catalog next-watch performance.

Each study writes:

- `config.json`, `data_manifest.json`, `provenance.json`, `source/`.
- `seedN/recall/models/{popularity,cosine,mf}/`: fitted retrieval scorers.
  `recall/checkpoints/mf.pt` and `recall/history.json` record fixed-budget fitting.
- `seedN/candidates/`, `recall_manifest.json`: frozen source scores and hashes
  covering every recall/cache file.
- `seedN/recipe/`: complete epoch history, `checkpoints/{best,last}.pt`, fitted
  `model/` with separate config/weights/features, keyed `predictions.parquet`,
  and five funnel reports with per-user metrics/traces.
- `runs.json`, `summary.json`: per-seed event/cohort and funnel metrics, coverage,
  training-prevalence baseline, primary/replication summaries, mean/sample SD and
  seed-aligned paired differences with mean/sample SD. Undefined metrics stay null;
  defined-seed counts are explicit.

Event metrics are recomputed from the exported scorer and checked against the
selected checkpoint before funnel evaluation. Event AUC and user AUC are distinct
from candidate recall/NDCG; better AUC alone does not establish a better funnel.
No result automatically promotes a recipe.

## Reuse and continuation

```sh
# New recipe comparison, using exactly the previous fitted recall and candidates:
python -m experiments.kuairand --config followup.json \
  --recall-from runs/kuairand_side --out runs/kuairand_followup

# CPU continuation: configuration is identical except for a larger epoch cap:
python -m experiments.kuairand --config extended.json \
  --resume-from runs/kuairand_side --out runs/kuairand_extended
```

Recall reuse checks prepared-data identity, seed, target, MF budget, fitted scorer
identity/provenance and every saved recall/cache hash. Continuation also checks the
whole study configuration except the epoch cap, then delegates optimizer/RNG and
continuation to the shared Trainer; the job explicitly restores selected weights
before exporting. It reuses original recall;
MF is not retrained. New runs require prepared-v4 data and the current output
layout. Regenerate older prepared-v3 data into a fresh directory; its changed
identity intentionally prevents recall reuse across preparation versions.
Pre-refactor training checkpoints require their original code. Scoring replay
can still load existing fitted-scorer-v2/v3/v4 packages; new exports use v5. The supplied study configurations use CPU; this runner
requires CPU for exact reload/continuation checks.

## Historical evidence

The [matched-seed results](kuairand_confirmation.md), [initial screen](kuairand_screen.md),
[data review](kuairand_data_review.md), [feature inventory](kuairand_feature_inventory.md),
figures and result JSON remain evidence from
[`033fdc9`](https://github.com/giunglee/short-video-feed-ranker/tree/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5).
Use that pinned checkout for the original commands/checkpoints. The architecture
changes Deep input ordering and the new runner does not claim identical historical
weights or metrics. The [refactor regression](kuairand_refactor_regression.md) reruns
all six arms over seeds 43–46 and separates exact input/recall parity from changed
ranking trajectories. The original validation reuse and adaptive-ablation
disclosures still apply.
