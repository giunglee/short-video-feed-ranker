# short-video-feed-ranker

A public PyTorch study of **recommendation, ranking, and incremental value**.
Implement core modeling patterns from public papers, then explain their tensor
flow, trade-offs, and failure modes through a small number of finished experiments.
Educational only — not affiliated with any employer.

## Current work

Correct the Wide & Deep experiment: EDA → big-matrix train/validation protocol →
trainer diagnostics → multi-seed comparison → small-matrix test evaluation.
The corrected pipeline is implemented and checked on synthetic data.
EDA, feature decisions, and experiment conclusions are separate follow-up work.
Historical small-matrix results are archived locally. Recall code is paused.

## Read the active code in this order

1. `src/feed_ranker/data/schema.py`, `src/feed_ranker/data/config.py` — raw events, tensor batches, and feature contracts.
2. `src/feed_ranker/models/components/feature_encoder.py`, `src/feed_ranker/models/components/mlp.py` — tensors.
3. `src/feed_ranker/models/wide_deep.py` — wide/deep paths and their summed prediction.
4. `src/feed_ranker/data/kuairec_preprocessor.py`, `src/feed_ranker/data/kuairec_dataset.py` — fixed pair split and train-fitted encodings.
5. `src/feed_ranker/training/trainer.py`, `src/feed_ranker/training/evaluation.py`, `src/feed_ranker/training/metrics.py` — updates, evaluation, and metrics.
6. `src/feed_ranker/experiments/wide_deep_kuairec.py`, `src/feed_ranker/experiments/multi_seed.py` — orchestration.

We will revise these in small steps, explaining inputs, outputs, and decisions.

## Learning progression

One experiment at a time; later stages are planned, not implemented.

1. **Feature-based Ranking — protocol correction in progress.** Sparse/dense features, embeddings,
   generalization, pair-held-out evaluation. Reference:
   [Wide & Deep](https://arxiv.org/abs/1606.07792).
2. **Candidate Retrieval — after the ranking correction.** One learned retrieval implementation:
   negative sampling, retrieval metrics, and the Recall → Rank funnel.
   Keep popularity and item-cooccurrence as fixed comparisons. Reference:
   [YouTube DNN](https://research.google/pubs/pub45530/).
3. **Multi-task Ranking — planned.** Shared-bottom vs MMoE: which tasks benefit
   or regress? GradNorm only for a concrete loss-balancing question. References:
   [MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007), optional
   [GradNorm](https://proceedings.mlr.press/v80/chen18a.html).
4. **Behavior-conditioned Ranking — planned.** Pooling vs candidate-aware
   attention, analyzed by history length and user activity. Reference:
   [DIN](https://arxiv.org/abs/1706.06978).
5. **Sequential Representation — planned.** Candidate-independent self-attention
   vs candidate-conditioned behavior under comparable scoring heads. Reference:
   [SASRec](https://arxiv.org/abs/1808.09781).
6. **Relational/Subgraph Ranking — later.** Does public relational context add
   signal beyond behavior history? Validate available relations first. Reference:
   [Neural Graph Pattern Machine](https://arxiv.org/abs/2501.18739).
7. **Incremental-value Modeling — later.** Causal Forest + XNet (exact XNet
   public reference to be resolved before implementation). Define treatment,
   outcome, assumptions, and policy evaluation first. Reference:
   [Causal Forest](https://arxiv.org/abs/1510.04342).

GraphSAGE/PinSage and other uplift methods are references, not extra milestones.
Generative recommendation is a reading stream outside this implementation plan.

## Data and experiment standard

Use public [KuaiRec](https://kuairec.com/) primarily for ranking/retrieval.
For questions requiring randomized exposure, use an appropriate randomized subset
of [KuaiRand](https://kuairand.com/) or another suitable dataset, with explicit
treatment/outcome definitions. Density alone does not establish a causal experiment.

Each experiment states its question, baseline, change, gains/regressions, and
limitations. Explain shapes, attention semantics, loss, and masking/sampling;
check synthetic forward/backward behavior before real runs. Select on validation;
reserve test for final evaluation. Historical practices are recorded in milestones.

## What's implemented

- KuaiRec pair-held-out ETL and batches: `src/feed_ranker/data/kuairec_preprocessor.py`,
  `src/feed_ranker/data/kuairec_dataset.py`.
- Wide & Deep, baselines, training, and multi-seed experiments:
  `src/feed_ranker/models/wide_deep.py`, `src/feed_ranker/training/`, `src/feed_ranker/experiments/wide_deep_kuairec.py`.
- Dataset download helper: `scripts/download_datasets.py`.

## Run the experiment

```bash
# Python 3.10 or newer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m feed_ranker.experiments.eda_kuairec --output artifacts/wide_deep_v2/eda.json
```

Review labels, repeated events, and feature provenance first. The example feature
spec is a plumbing baseline, not an EDA conclusion; choose crosses on big training
rows only. Then prepare once and reuse the same data for every training seed:

```bash
python -m feed_ranker.data.kuairec_preprocessor --features configs/wide_deep_features.json --output data/extract/kuairec_v2
python -m feed_ranker.experiments.wide_deep_kuairec train --data data/extract/kuairec_v2 --output artifacts/wide_deep_v2/debug --model wide_deep_nn --epochs 2 --steps 10
python -m feed_ranker.experiments.multi_seed --data data/extract/kuairec_v2 --output artifacts/wide_deep_v2/comparison
python -m feed_ranker.experiments.plot_run --help
```

Loss, evaluation, and checkpoint selection are independent CLI settings:
`--loss mae|mse|smooth_l1`, `--metrics mae mse xauc user_xauc`,
`--selection-metric mae`, and `--selection-mode min|max`. The selection metric must
be reported and finite; exact ties keep the earliest epoch. Defaults remain MAE
loss, MAE/XAUC/user XAUC reporting, and minimum validation MAE selection.
For an MSE experiment, use `--loss mse --metrics mae mse --selection-metric mse`.
Only scalar regression is supported here; retrieval sampling and candidate-set
metrics will be defined with the recall experiment.

The comparison defaults to Deep and W&D, seeds 42–46. It reports validation
mean/sample std and same-seed W&D-minus-Deep differences. A lower MAE difference
is better; a higher concordance difference is better. `--steps` marks exploratory
runs. Freeze features and settings before evaluating **each** seed's `best.pt`:

```bash
python -m feed_ranker.experiments.wide_deep_kuairec evaluate --data data/extract/kuairec_v2 --checkpoint artifacts/wide_deep_v2/comparison/wide_deep_nn-42/best.pt --output artifacts/wide_deep_v2/comparison/wide_deep_nn-42/test.json
python -m unittest discover -s tests -v
```

Runs preserve config/provenance, epoch history, best/last checkpoints, and
train-fitted mean/median `ConstantRegressor` baselines. By default, best means minimum validation MAE; inspect
neighboring epochs and all seeds before interpreting an isolated gain. Training
loss is measured during updates, not by re-scoring a frozen checkpoint. Plots keep
loss and each validation metric on separate axes.

Data and limitations: [`docs/datasets.md`](docs/datasets.md) ·
Setup: [`data/README.md`](data/README.md) · Progress: [`docs/milestones.md`](docs/milestones.md)

For a fixed full comparison with automatic final evaluation and saved summaries:

```bash
python -m feed_ranker.experiments.run_protocol --output artifacts/wide_deep_v2/full
```

The same command resumes completed runs and archives/restarts an interrupted run
from its seed. It refuses recipe/source changes. Outputs include `status.json`,
`report.md`, `summary.json`, per-run curves and best/last checkpoints. Keep the
computer awake and connected to power for long runs.

Check code with `ruff check src tests scripts`, `ruff format --check src tests scripts`,
and `python -m unittest discover -s tests -v`. Code lives in `src/feed_ranker`;
`data/` holds local dataset files, `configs/` feature specifications, and `scripts/`
download utilities. CSV/Parquet ingestion is an offline modeling adapter; online
feature retrieval and serving contracts are outside the current experiment.

## Stack

PyTorch `nn.Module` for models; `pandas` / `numpy` only in data loaders;
`torch.optim.AdamW` for training.

## License

Code: MIT — see [LICENSE](LICENSE). Datasets retain their original licenses from the
KuaiRec / KuaiRand authors.
