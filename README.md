# short-video-feed-ranker

A public PyTorch study of **recommendation, ranking, and incremental value**.
Implement core modeling patterns from public papers, then explain their tensor
flow, trade-offs, and failure modes through a small number of finished experiments.
Educational only — not affiliated with any employer.

## Current work

Correct the Wide & Deep experiment: EDA → big-matrix train/validation protocol →
trainer diagnostics → multi-seed comparison → small-matrix test evaluation.
The current training code still uses the old protocol; corrected runs have not begun.
Historical small-matrix results are archived locally. Recall code is paused.

## Read the active code in this order

1. `data/config.py` — feature and label contracts.
2. `models/components/feature_encoder.py`, `models/components/mlp.py` — tensors.
3. `models/wide_deep.py` — wide/deep paths and their summed prediction.
4. `data/kuairec_preprocessor.py`, `data/kuairec_dataset.py` — existing data path.
5. `training/trainer.py`, `training/metrics.py` — one training run and evaluation.
6. `experiments/wide_deep_kuairec.py`, `experiments/multi_seed.py` — orchestration.

We will revise these in small steps, explaining inputs, outputs, and decisions.

## Learning progression

One experiment at a time; later stages are planned, not implemented.

1. **Feature-based Ranking — protocol correction in progress.** Sparse/dense features, embeddings,
   generalization, temporal evaluation. Reference:
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

- KuaiRec temporal ETL and batches: `data/kuairec_preprocessor.py`,
  `data/kuairec_dataset.py`.
- Wide & Deep, baselines, training, and multi-seed experiments:
  `models/wide_deep.py`, `training/`, `experiments/wide_deep_kuairec.py`.
- Dataset download helper: `data/scripts/download_datasets.py`.

## Existing runner (legacy protocol)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/scripts/download_datasets.py --all
python -m data.kuairec_preprocessor   # builds data/extract/kuairec/*.parquet
python -m experiments.wide_deep_kuairec --model wide_deep_nn
```

Data setup: [`docs/datasets.md`](docs/datasets.md) · Feature/label EDA +
findings: [`data/kuairec_eda_findings.md`](data/kuairec_eda_findings.md) ·
Build log: [`docs/milestones.md`](docs/milestones.md)

## Stack

PyTorch `nn.Module` for models; `pandas` / `numpy` only in data loaders;
`torch.optim.AdamW` for training.

## License

Code: MIT — see [LICENSE](LICENSE). Datasets retain their original licenses from the
KuaiRec / KuaiRand authors.
