# Which features help a fixed Deep ranker?

> Historical evidence and commands refer to the [pre-refactor source](https://github.com/giunglee/short-video-feed-ranker/tree/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5).
> For the active implementation, use the [KuaiRand workflow](kuairand_workflow.md).

**Finding:** removing the discrete duration band, while retaining continuous log
video duration, improved both within-user ordering and the final feed in this
bounded study. Adding features did not necessarily help.

## Question and comparison

- Baseline: Deep `[64, 32]`, 8-dimensional embeddings for user/video IDs, age,
  follow-count range and duration band, plus log video duration.
- Six changes, each tested independently: remove age; remove follow count; remove
  both profiles; remove duration band; add sum-pooled item categories; add activity,
  fan-count and registration-tenure profiles. No combinations or architecture tuning.
- Screen each change with seed 42; confirm at most the two largest positive
  validation user-XAUC gains with seeds 43–46. The baseline reuses five verified
  architecture-study checkpoints. Total: 14 new training runs, 19 scored checkpoints.

## Fixed protocol

- Signature-deduplicated big-matrix events; fixed pair-grouped train/validation
  split 2026. Fit new vocabularies on training entities and preserve existing rows,
  labels, encodings and event weights exactly. No small-matrix access in this study.
- Event-weighted MAE on `min(watch_ratio, 2)`; AdamW `.001`, weight decay 0,
  batch 1024, 120 epochs. No dropout, scheduler, extra epochs or best-seed choice.
- Choose highest validation user-XAUC, earliest exact tie, using 256 pairs/user.
  Report the same checkpoint with 4096 pairs/user and 1M global pairs, fixed seed0.
- Five-seed means, sample SD and same-seed deltas; inspectseeds 43–46 separately because
  seed 42 selected the confirmation candidates. Seed variation is not data uncertainty.
- For the funnel, keep popularity/cooccurrence/MF candidates fixed within each seed.
  Primary M50/K10, M20/100 sensitivities. First-event linear cap2 gain, full-pool
  IDCG, equal-user averaging. The ranking checkpoint is never reselected on NDCG.

## Results

- **Remove duration band:** capped user-XAUC `.735748 → .736576`;
  paired gain `+.000828 ± .000506`, positive in 5/5 seeds.
- **Final feed:** MF → rank cap2 NDCG10 `.737278 → .739541`;
  paired gain `+.002263 ± .000447`, positive in 5/5 seeds. Recall and candidate
  oracle are unchanged, isolating the ranker's contribution on these candidates.
- **Trade-off:** capped MAE `.324153 → .324430`; global XAUC is nearly unchanged
  (`.721216 → .721235`). This is an ordering gain, not an across-metric improvement.
- Excluding screening seed 42, mean user-XAUC gain remains `+.000786` and MF/M50
  NDCG gain `+.002370`.
- Removing age has negligible, inconsistent effects: user-XAUC `+.000063` and
  MF/M50 NDCG `+.000045`. There is no clear quality reason to remove it.
- Removing follow count or both profiles hurt screening performance. Added categories
  and activity profiles did not advance; one screening seed does not prove they
  are generally useless.

[Per-seed ranking scores](results/deep_feature_ranking.csv) and
[matched funnel results](results/deep_feature_funnel.csv) retain the comparison evidence.

## Failure analysis and interpretation

- Duration cohorts are not uniformly better: the >60-second cohort's user-XAUC
  regressed by about `.00195`; shorter cohorts improved. These diagnostics sample
  pairs separately within cohorts and do not sum to the headline effect.
- MAE increased for high-activity users and the highest video-support cohort.
  Raw-ratio ≥5 examples have tied capped labels, so within-tail XAUC is undefined;
  use their error metrics rather than interpreting missing XAUC as zero.
- A redundant discretization can hurt a fixed model's ordering even when duration
  itself is useful. This is an interpretation, not evidence of a causal mechanism.
- **Recommendation:** consider the no-duration-band recipe as the next baseline,
  keeping continuous duration and both original profiles. Publication does not
  automatically change the pipeline's selected model.
- Reused validation, prior small-matrix EDA/evaluation exposure and unverified
  historical profile/category availability remain limitations. These observed-pool
  results are not temporal, causal, full-catalog or online-lift estimates.

## Compatibility with the shared pipeline

- Reused all saved weights and selected epochs; no full retraining or new test access.
- All 19 checkpoints reproduce 8,192 stored validation predictions each and their
  detailed validation metrics. Three fixed minibatch updates across all seven input
  configurations reproduce the original losses and weights exactly.
- All 15 confirmed model/seed combinations reproduce 1,030,046 full-pool predictions
  each, all 135 candidate exports, and per-user/duration replay metrics exactly.
- Request-time MF/M50 inference covers 107,640 queries. One query swaps two tied
  videos at positions 4/5; top-10 membership is unchanged. Maximum prediction
  difference is `1.43e-6`, within the existing `atol=rtol=1e-6` comparison. The largest
  per-run mean NDCG difference is `1.96e-8`; it does not change the conclusion.
  Replay and request-time floating-point arithmetic should not be described as
  universally identical.
- The category artifact reproduces 8,192 stored predictions after conversion.
  **88 tests**, Ruff and formatting checks pass. [Per-seed compatibility evidence](results/deep_feature_compatibility.csv).

## Implementation and reproduction

- [Feature preparation](https://github.com/giunglee/short-video-feed-ranker/blob/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5/src/feed_ranker/data/feature_ablation.py) preserves the
  existing split; [study runner](https://github.com/giunglee/short-video-feed-ranker/blob/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5/src/feed_ranker/experiments/deep_features.py)
  owns the declared arms, bounded confirmation and reports.
- Pooled category IDs use `[batch, length]`, OOV 0 and masked padding −1. The shared
  encoder sums embeddings; inference exports the fitted per-item lookup. This
  supports reproducing the screened category arm without changing scalar models.
- New checkpoints save resolved `FeatureConfig` and `ModelSpec`. Legacy feature
  checkpoints require an explicit matching feature configuration during conversion;
  their weights, vocabulary order and selected epoch are retained.

With prepared baseline data, public KuaiRec CSVs and the frozen architecture study:

```sh
python -m feed_ranker.experiments.deep_features \
  --base PATH/TO/base_data --raw PATH/TO/kuairec \
  --baseline-study PATH/TO/architecture_study --output PATH/TO/new_feature_study
```

Export each chosen checkpoint, then evaluate through the shared recommendation pipeline:

```sh
python -m feed_ranker.experiments.convert_inference --threads 1 ranker \
  --checkpoint PATH/TO/best.pt --checkpoint-sha256 CHECKPOINT_SHA256 \
  --ranking-data PATH/TO/prepared_data --recall-data PATH/TO/recall_data \
  --output PATH/TO/ranker_artifact
python -m feed_ranker.experiments.evaluate_funnel \
  --data PATH/TO/recall_data --ranking-data PATH/TO/prepared_data \
  --ranker PATH/TO/ranker_artifact \
  --retrievers PATH/TO/popularity PATH/TO/cooccurrence PATH/TO/mf \
  --output PATH/TO/funnel_results --execution replay
```

For legacy feature-study checkpoints only, add `--feature-config PATH/TO/features.json`
to conversion. Generate that resolved JSON from the saved study arm and manifest:

```python
from dataclasses import asdict
from pathlib import Path
from feed_ranker.artifacts import write_json
from feed_ranker.experiments.deep_features import make_dataset

config = make_dataset(Path("PATH/TO/prepared_data"), "no_duration_band").config.feature
write_json(Path("features.json"), asdict(config))
```

Use the matching MF seed for each ranker. `--execution model` exercises request-time
inference; replay uses cached model scores with the same recommendation and evaluation
contracts. Neither path trains or reselects a checkpoint. Raw data and weights are
not bundled with the repository.
