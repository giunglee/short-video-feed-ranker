# Milestones

Public build log for this repo — what's done, in the order it was done, with the
headline numbers and the non-obvious things learned along the way. Full experiment
logs (every hyperparameter tried, not just the winners) stay in a local, private
notes file; this page is the public trace of the same work.

---

## Phase 1 — Wide & Deep baseline on KuaiRec ✅

**Task:** predict `watch_ratio` (continuous) for a user/video pair — the classic
memorization-vs-generalization setup Wide & Deep was built for.
**Paper:** Cheng et al., [*Wide & Deep Learning for Recommender Systems*](https://arxiv.org/abs/1606.07792),
DLRS 2016. **Data:** public [KuaiRec](https://kuairec.com/) small matrix.

### 1. Data pipeline (`data/kuairec_preprocessor.py`, `data/kuairec_dataset.py`)

ETL from the raw KuaiRec CSVs (interaction matrix + user/item feature tables) to
train/val/test parquet on a **temporal** split (80/10/10 by timestamp — a random
split would leak future interactions into training). Two correctness bugs were
found and fixed via EDA rather than assumed away:

- A `date` column parse bug (`pd.to_datetime` on a raw `YYYYMMDD` integer reads it
  as nanoseconds-since-epoch) silently collapsed the item-daily-features join to
  one constant snapshot per video, applied to every interaction with that video
  regardless of the interaction's real date.
- Rows with a missing timestamp (≈3.9%) were sorting to the tail on split and
  landing entirely in `test`, handing it a batch of fake all-zero engagement
  features that neither `train` nor `val` ever saw.

Both were caught by routine data-quality checks (a within-ID constancy check and a
train/val/test split audit comparing missing-rate and mean/std per column), not by
design — full writeup in [`data/kuairec_eda_findings.md`](../data/kuairec_eda_findings.md).

### 2. EDA & feature selection

- **Label:** `watch_ratio`, mean 0.91 / std 1.36, heavy right tail (p99 ≈ 3.6×
  mean) — `log1p` used for plotting, MAE still evaluated in the original scale.
- **Redundancy:** Cramér's V screen (cheap, broad) → bijective crosstab check
  (expensive, exact) only on the pairs the screen flagged. Found and dropped 12
  exact-duplicate columns (e.g. several anonymized `onehot_feat*` columns turned
  out to be 1:1 re-encodings of already-present human-readable columns).
- **Cross-features (wide path):** systematically scanned low-cardinality
  categorical pairs by *interaction lift* — `lift(a,b jointly) / (lift(a) ×
  lift(b))`, which isolates true synergy/conflict from a cross that's just one
  strong marginal riding along. Two hand-picked crosses that scored ≈1.0 (no real
  interaction) were dropped in favor of 7 scan-selected crosses, after also
  filtering out near-duplicate candidates that scored well only because they
  100%-overlapped a cross already picked.
- **Leakage:** excluded `play_duration` (label-adjacent) and its soft proxies.

### 3. Model & training infra

Four variants per the paper, built from shared components
(`models/components/feature_encoder.py`, `models/components/mlp.py`):

| Model | Path |
|---|---|
| `LinearRegression` | dense + embeddings → `Linear` |
| `WideNN` | dense + binary crosses → `Linear` |
| `DeepNN` | dense + embeddings → `MLP` |
| `WideDeepNN` | `wide_logit + deep_logit` (paper's joint sum, not a shared head) |

Generic `Trainer` (`training/trainer.py`): true shuffled full-epoch batching
(replacing an earlier per-step resample that under-covered the data at a fixed
step budget), best-validation-checkpoint restore (so a run can't "win" by landing
on a lucky last epoch), AdamW with configurable weight decay. Metrics: **MAE**
(primary) plus **XAUC** — sampled pairwise ranking concordance, reported
alongside MAE because `watch_ratio` is consumed by the ranker to *order*
candidates, not read as an absolute number; a model can have decent MAE while
ranking badly, or vice versa (metric convention follows the KuaiRec benchmark
paper and its watch-time-prediction follow-ups).

**Methodology note:** every result below reports both val and test MAE/XAUC, but
model/hyperparameter selection is driven by **val only** — test is logged purely
as a monitoring signal (and it earned its place: the bug above was first visible
as a test-only MAE blowup, not something selection was tuned to catch).

### 4. Results (best config per model, selected by val MAE)

| Model | val MAE | val XAUC | test MAE | test XAUC |
|---|---|---|---|---|
| predict-mean baseline | 0.488 | – | 0.523 | – |
| `linear_reg` | 0.365 | 0.711 | 0.431 | 0.679 |
| `wide_nn` | 0.364 | 0.732 | 0.411 | 0.721 |
| `deep_nn` | 0.363 | 0.708 | 0.440 | 0.664 |
| **`wide_deep_nn`** (`256→64→8→1`, dropout 0.3) | **0.339** | **0.746** | 0.402 | 0.720 |

All four learned models beat the predict-mean baseline on both val and test.
Single seed — multi-seed variance check is an open follow-up before treating any
one config as *the* answer rather than *a* good one.

### 5. Learnings worth remembering

- **`wide&deep > deep ≈ wide > linear`** here, matching the paper's own ordering
  directionally. Worth being precise about what the paper actually reports
  though (Table 1, Google Play app-store ranker): **offline AUC** wide edges deep
  (0.726 vs 0.722), but **online acquisition gain** deep edges wide (+2.9% vs 0%
  control) — Wide & Deep wins on both. "Deep beats wide" isn't universally true
  even in the source paper; it depends which metric and which regime.
- Architecture size should track data/feature scale, not habit: a narrower/shallower
  `wide_deep_nn` (`256,64,8,1`) outperformed a wider/deeper one (`512,128,32,8,1`)
  on this dataset once training was actually covering full epochs correctly.
- Regularization isn't universally helpful — dropout that clearly helped
  `wide_deep_nn` made `deep_nn` strictly worse on val MAE at every setting tried.
  Same knob, opposite verdicts, on two closely related architectures.
- A training-loop bug (partial epoch coverage from per-step resampling) was
  larger in effect than most of the hyperparameter changes tried on top of it —
  worth fixing infra correctness before reading too much into a hyperparameter
  sweep's results.

**Next:** MMoE multitask (KuaiRand-Pure) — see the model progression table in the
[README](../README.md).

---

## Phase 2 — MMoE multitask on KuaiRand-Pure (not started)

## Phase 3 — GradNorm task balancing (not started)

## Phase 4 — DIN → SIM sequence attention on KuaiRand-1K (not started)

## Additive track — Recall (SASRec → FAISS) + IPS offline eval (not started)
