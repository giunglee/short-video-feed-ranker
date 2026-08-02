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

Experiment entry points:
- `experiments/wide_deep_kuairec.py` — single-seed run; `--emb-mode {all,ids,cats}`
  selects which embedding tables the deep tower uses (wide path is always
  dense∥crosses).
- `experiments/multi_seed.py` — same config across seeds 42–46 (or custom),
  reports mean ± std so a val gap can be checked against seed noise.

**Methodology note:** every result below reports both val and test MAE/XAUC, but
model/hyperparameter selection is driven by **val only** — test is logged purely
as a monitoring signal (and it earned its place: the bug above was first visible
as a test-only MAE blowup, not something selection was tuned to catch). Final
table rows are multi-seed; ablation cells that say “seed 42” are directional.

### 4. Results (best config per model, selected by val MAE)

After the ID-embedding ablation (see learnings): deep / W&D deep-tower use
**profile/context cats + dense only** (no raw `user_id` / `video_id` / …).
`wide_nn` never used IDs. `linear_reg` still includes IDs (not re-tuned).
All rows: mean ± std over seeds 42–46.

| Model | config | val MAE | val XAUC | test MAE | test XAUC |
|---|---|---|---|---|---|
| predict-mean baseline | – | 0.488 | – | 0.523 | – |
| `linear_reg` | emb=all, wd 1e-4 | 0.366 ± 0.001 | 0.711 ± 0.002 | 0.432 ± 0.004 | 0.678 ± 0.004 |
| `wide_nn` | dense + crosses | 0.364 ± 0.000 | 0.732 ± 0.001 | 0.411 ± 0.000 | 0.723 ± 0.001 |
| **`deep_nn`** | `128→32→8→1`, emb=**cats**, drop 0 | **0.321 ± 0.000** | **0.758 ± 0.001** | **0.369 ± 0.001** | **0.745 ± 0.001** |
| **`wide_deep_nn`** | `256→64→8→1`, drop **0**, emb=**cats** | **0.321 ± 0.001** | **0.759 ± 0.001** | **0.370 ± 0.001** | **0.745 ± 0.001** |

(Earlier W&D cats multi-seed used drop 0.3 → val 0.334 / test 0.380 — unfair vs
deep's drop 0. Matching dropout closes the gap.)

Ordering after dropping raw IDs + matched dropout: **deep ≈ W&D ≫ wide > linear**
on val MAE. A one-off wide-path feature ablation (dense∥crosses vs crosses-only
vs dense-only) was within ~0.001 at seed 42 — crosses add little once the deep
tower is healthy; the code keeps the paper-default dense∥crosses stack.

### 5. Learnings worth remembering

- On this split the final ranking is **deep ≈ W&D ≫ wide > linear** (matched
  dropout, no raw IDs). The paper's own ordering is worth citing carefully
  (Table 1, Google Play): **offline AUC** wide edges deep (0.726 vs 0.722), but
  **online acquisition** deep edges wide (+2.9% vs 0% control) — Wide & Deep
  wins on both. "Deep beats wide" isn't universally true even in the source
  paper; it depends which metric and which regime.
- Architecture size should track data/feature scale, not habit: a narrower/shallower
  `wide_deep_nn` (`256,64,8,1`) outperformed a wider/deeper one (`512,128,32,8,1`)
  on this dataset once training was actually covering full epochs correctly.
- Regularization isn't universally helpful — with emb=all (IDs), dropout helped
  `wide_deep_nn` and hurt `deep_nn`. After dropping IDs, matching W&D to deep's
  drop=0 erases nearly all of the remaining MAE gap; the earlier “deep > W&D”
  ranking was partly an unfair hyperparameter comparison.
- A training-loop bug (partial epoch coverage from per-step resampling) was
  larger in effect than most of the hyperparameter changes tried on top of it —
  worth fixing infra correctness before reading too much into a hyperparameter
  sweep's results.
- **Temporal item cold-start kills raw ID embeddings.** Users are 100% shared
  across train/val/test on KuaiRec-small, but ~77% of test rows (and ~45% of val)
  have a `video_id` never seen in train. Deep with IDs: train≪val, weak test
  XAUC; IDs-only → XAUC 0.5 (random). Dropping IDs (profile/context cats + dense)
  fixed deep (test MAE 0.369 / XAUC 0.745) and also improved W&D (val 0.337 /
  test 0.383 vs 0.340 / 0.402 with IDs). W&D still looked strong *with* IDs
  mainly because the **wide** path (dense + crosses) does not use IDs and
  transfers under the temporal split. Emb-dim tweaks would not fix OOV items —
  production systems route around cold items with content/side features, explore,
  and continual ID updates, not larger ID tables.

**Next:** MMoE multitask (KuaiRand-Pure) — see the model progression table in the
[README](../README.md).

---

## Phase 2 — MMoE multitask on KuaiRand-Pure (not started)

## Phase 3 — GradNorm task balancing (not started)

## Phase 4 — DIN → SIM sequence attention on KuaiRand-1K (not started)

## Additive track — Recall (SASRec → FAISS) + IPS offline eval (not started)
