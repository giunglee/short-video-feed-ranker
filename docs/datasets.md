# Data protocol and training decisions

**Current ranking study:** capped-label MAE (`min(watch_ratio, 2)`), maximum
validation user-XAUC checkpoint selection, and a 120-epoch cap. See the
[architecture results and reproduction](ranking_architecture.md).
The raw-label recipe below records the earlier budget investigation.

## Data contract

- **Source:** public [KuaiRec](https://kuairec.com/), locally under `data/raw/kuairec/`.
  Big matrix supplies training/validation; small matrix supplies final test.
- **Cleaning:** remove complete repeated signatures: user, video, timestamp, play
  duration, video duration, watch ratio. Keep the first; retain missing timestamps
  and distinct-time repeats. Record removals as `signature-dedup-v2`.
- **Split:** fixed 90/10 big-only split by user–video pair, seed 2026. Reject
  big/small pair overlap. Shared users/items are expected; retain undated test rows.
- **Interpretation:** evaluates unseen pairs, not future-time prediction or causal
  effects. Historical small-label inspection and test evaluation prevent claiming
  an untouched test set; new model selection uses big validation only.
- **Preprocessing:** fit vocabularies and mean/median baselines on train only.
  Missing/unknown categories use index 0 (fixed zero in the architecture study;
  learned in the earlier reference). Dense missing values become
  zero; model inputs use `log1p(max(x, 0))`.

## Feature scope

- **Reference:** user ID, video ID, and duration; no explicit crosses.
- **Architecture study:** IDs, duration, age range, follow-count range, duration
  bands, and selected explicit crosses. Item categories are outside this comparison.
- **Availability limits:** profiles are snapshots, not verified historical features.
  Daily item attributes need a separate timing audit. Captions and sequence models
  are deferred; daily engagement aggregates are excluded.
- **Serving boundary:** CSV → parquet → tensor batches is an offline adapter.
  Raw IDs remain batch metadata. Online feature lookup and point-in-time serving
  contracts are outside this study.

## Training recipe and budget

- **Reference model:** Wide & Deep; two 8-dimensional ID embeddings plus log1p
  duration → 17 deep inputs → [128, 32] → scalar output. Duration also feeds the
  wide branch. No dropout.
- **Optimizer:** AdamW, learning rate 0.001, weight decay 0, no scheduler.
- **Batches:** training 1,024; evaluation 8,192. Each epoch shuffles every training
  row once using training seed + epoch index; retain the last partial batch.
  10,406,371 rows give 10,163 updates per epoch. `--steps` is debug-only.
- **Loss:** raw-label MAE with event weighting. Optional `--weighting pair` uses
  inverse pair frequency and divides minibatch loss by its weight sum. This
  approximates equal-pair risk; it does not equalize users or correct exposure bias.
- **Budget and selection:** pass `--epochs 120` explicitly; generic defaults remain
  15. Train to the cap, select minimum validation MAE, and keep the earliest exact
  tie. Report every metric from that checkpoint.
- **Confirmation:** matched training seeds 42–46; fixed split, preprocessing policy,
  and selection rule. Report each seed, mean/sample standard deviation, and
  same-seed differences. Change one declared factor per comparison.

### Evidence for the cap

- **Investigation:** seed 42, progressing through 30, 60, and 120 epochs.
  The reference still selected the boundary at epoch 60.
- **Reference result:** extending to 120 selected epoch 99, validation MAE 0.452019.
  At epoch 120, fixed-train MAE was 0.441212 versus validation MAE 0.452320:
  training fit improved while validation improvement flattened.
- **Other budget probes:** pair weighting and added categories selected epochs
  108 and 118. These single-seed runs do not establish comparative gains.
- **Decision:** adopt 120 as a practical cap, not proof of universal convergence.
  A future candidate needing a different budget requires a declared investigation.
  No new small-matrix evaluation informed this decision.

## Metrics and failure diagnostics

- **MAE:** event-weighted across all validation rows, regardless of training weights.
- **Global XAUC:** sampled concordance across unequal-label event pairs; 100,000 pairs.
- **User XAUC:** equal-user average of within-user concordance; 256 sampled pairs per
  user. Prediction ties score 0.5; users without unequal-label pairs are excluded
  and counted. Metric seed is fixed at 0. Seed variation is not data uncertainty.
- **Generalization gap:** score the same 100,000 training rows in evaluation mode,
  sampling seed 2026. Compare with validation from the same checkpoint. Training
  loss accumulated during updates is not directly equivalent. Set
  `--train-eval-rows 0` to disable the diagnostic.
- **Failure review:** inspect neighboring epochs, gradients, prediction distributions,
  and errors by duration, label tail, user activity, and video support. Allowlisted
  traces preserve predictions/labels and features; training counts define activity
  groups. Optional snapshot activity remains descriptive, not point-in-time verified.
- **Runtime:** separate training, diagnostics, validation, and checkpoint time.
  Runs used CPU with four threads. Equal epochs mean equal data passes, not equal
  compute; local wall time is not serving latency.

## Reproducibility

- Save configuration, source/data provenance, full history, best/last checkpoints,
  and train-fitted constant baselines.
- CPU continuation via `run_experiment(..., resume_from=...)` restores model,
  optimizer, and torch RNG into a fresh directory. Preserve history and prior best;
  only the epoch cap may change.
- [Commands and setup](../data/README.md) · [EDA rationale](../data/kuairec_eda_findings.md)
  · [Dataset paper](https://doi.org/10.1145/3511808.3557220).
