# Capped-label ranking: Linear, Wide, Deep, and Wide & Deep

> Historical evidence and commands refer to the [pre-refactor source](https://github.com/giunglee/short-video-feed-ranker/tree/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5).
> For the active implementation, use the [KuaiRand workflow](kuairand_workflow.md).

**Finding:** Deep `[64, 32]` performed best among the tested recipes on both
validation and test. Adding an explicit wide branch did not improve the selected
Deep recipe. This is an offline comparison under a bounded search, not a universal
architecture ranking or evidence of online uplift.

## Question and comparison

- Do explicit crosses and nonlinear representations improve within-user ordering
  beyond an additive baseline?
- All families receive the same information: user ID, video ID, age range,
  follow-count range, duration band, and log-transformed duration.
- **Linear:** categorical coefficients plus a linear duration term; no MLP.
- **Wide:** Linear plus selected age × duration and follow-count × duration indicators.
- **Deep:** five 8-dimensional embeddings plus duration → MLP → scalar prediction.
- **Wide & Deep:** sum the wide and deep predictions; jointly train both branches.
- Missing/unseen categories have fixed zero coefficients or embedding vectors.
  Profiles are snapshots whose historical availability is unverified.

## Protocol

- Remove complete duplicate event signatures; retain distinct observations of a pair.
  Use `signature-dedup-v2`, a fixed 90/10 big-matrix pair split (seed 2026), and
  training-only vocabularies. Reject big/small pair overlap.
- Train on every deduplicated event with MAE against `min(watch_ratio, 2)`.
  Capping defines bounded utility; it neither deletes tail events nor proves they
  are logging errors. Predictions remain unconstrained.
- Primary metric: capped **user-XAUC**, equally averaging eligible users. Secondary:
  capped global XAUC and event-weighted MAE. Equal-label pairs are excluded;
  prediction ties receive half credit.
- Select the maximum validation user-XAUC checkpoint, keeping the earliest exact
  tie. Report all metrics from that checkpoint. This supersedes the earlier
  raw-label experiment's minimum-MAE selector.
- Train with AdamW, batch size 1,024, no dropout or scheduler, and a 120-epoch cap.
  The predeclared extension rule did not trigger: a finalist had to peak in the
  final 10 epochs **and** improve its last-10 mean over the preceding 10 by >0.0002.
  If triggered, all four finalists would continue once to 180 epochs.
- Screen 21 trials with seed 42; confirm selected recipes with matched seeds 42–46
  (reuse seed 42). Total: **37 training runs and 20 frozen finalists**.
- Screen Linear/Wide learning rates `{0.001, 0.003, 0.01}` and Deep/joint hidden
  widths `{[32], [64, 32], [128, 32]}`. Follow with one regularization trial per
  family, cross-family comparisons, and an IDs/duration-only joint diagnostic.
- Freeze recipes and checkpoint hashes before small-matrix scoring. Selection uses
  100,000 global / 256 pairs per user; final scoring uses 1,000,000 / 4,096 with
  metric seed 0. Cohort diagnostics use the smaller sample and do not select models.

## Selected recipes and results

All selected learning rates were 0.001. Linear, Deep, and W&D used no weight decay;
Wide used 0.0001 on eligible matrix parameters (embeddings and biases exempt).
Deep selected `[64, 32]`; W&D selected `[32]`. Both cross families were retained.

Mean capped user-XAUC over five seeds; ± denotes sample standard deviation:

- **Deep:** validation **0.735748 ± 0.000377**; test **0.767499 ± 0.000399**.
- **Wide & Deep:** validation **0.733617 ± 0.000648**; test **0.765647 ± 0.000279**.
- **Wide:** validation **0.728943 ± 0.000029**; test **0.766861 ± 0.000059**.
- **Linear:** validation **0.728742 ± 0.000051**; test **0.766674 ± 0.000065**.

Matched-seed test differences:

- Deep − Wide: **+0.000638 ± 0.000394**, positive in 5/5 seeds.
- Deep − Linear: **+0.000825 ± 0.000427**, positive in 5/5 seeds.
- W&D − Deep: **−0.001852 ± 0.000580**, negative in 5/5 seeds on test;
  also negative in all five validation seeds.
- Deep's test MAE / global XAUC were **0.258127 / 0.749264**, versus W&D's
  **0.261495 / 0.747945**. W&D still beat Wide and Linear on these two metrics.

## Diagnostics and interpretation

- Linear and Wide largely plateau after the first epoch, which already contains
  about 10,163 updates. Their jagged user-XAUC curves span roughly 0.0006;
  independently scaled plot axes make these fluctuations look large.
- Deep and joint models continue improving over more epochs. Some joint seeds
  select late checkpoints; 120 epochs is a practical budget, not proof of convergence.
- W&D beats Wide on validation user-XAUC but loses on test. Aggregate validation
  gains did not transfer uniformly to the denser small-matrix distribution.
- Relative to Deep, W&D improves the shortest-duration cohort but regresses on
  longer videos. These descriptive cohort comparisons locate failures; they do
  not establish that duration features caused them.
- Keep Deep as the strongest tested standalone recipe and retain simpler baselines.
  Its test user-XAUC advantage over Wide is consistent but small: 0.0638 percentage
  points. MAE training does not directly optimize pairwise ordering or top-K utility.

## Limits

- Earlier small-matrix EDA, training, and evaluation informed protocol development;
  this is not an untouched test set. No test-driven retraining followed this study.
- Seed 42 participated in tuning; results also include a seeds 43–46 sensitivity
  summary. All seeds have prior project history. Training-seed variation does not
  measure uncertainty across future users, datasets, or repeated validation searches.
- The search was sequential and bounded, not exhaustive or equal-compute tuning.
  Selected Deep and W&D widths differ, so their comparison is between tuned recipes.
- This split evaluates unseen pairs, not future-time prediction. Snapshot profiles
  are not verified serving-time features. Retrieval and funnel metrics were excluded;
  standalone gains do not establish better retrieved-candidate NDCG.

## Reproduce and inspect

After [installing dependencies and downloading KuaiRec](../data/README.md):

```bash
python -m feed_ranker.data.kuairec_preprocessor \
  --raw-dir data/raw/kuairec \
  --features configs/ranking_architecture_features.json \
  --output data/extract/ranking_architecture --split-seed 2026 --val-ratio 0.1
python -m feed_ranker.experiments.ranking_study \
  --data data/extract/ranking_architecture \
  --output artifacts/ranking_architecture --threads 4
```

- The second command runs the complete search, confirmation, and frozen test scoring;
  it took about 17 hours on the original local CPU. Use a fresh output directory.
- The feature specification preserves the original prepared columns, including
  legacy `main_*` indicators. Architecture construction ignores those indicators
  and supplies categorical coefficients directly, avoiding duplicate main effects.
- Outputs include the protocol, recipes, checkpoint hashes, full histories,
  per-seed and paired summaries, seeds 43–46 sensitivity, cohort traces, and curves.
  Raw data, checkpoints, and generated run artifacts are not committed.
- Start with [feature/model construction](https://github.com/giunglee/short-video-feed-ranker/blob/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5/src/feed_ranker/experiments/ranking_architecture.py)
  and [study orchestration](https://github.com/giunglee/short-video-feed-ranker/blob/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5/src/feed_ranker/experiments/ranking_study.py).
  Public synthetic fixtures exercise all four models and the freeze-before-test flow.

Reference: [Wide & Deep, Cheng et al. (2016)](https://arxiv.org/abs/1606.07792).
