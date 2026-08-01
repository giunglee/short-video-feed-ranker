# short-video-feed-ranker

From-scratch **short-video feed ranking** in PyTorch — paper-based reimplementations
along the classic stack: baseline pointwise ranker → multi-task MMoE → long-sequence
attention (DIN / SIM), with an **additive** sequential recall path (SASRec → FAISS).

Trained on public [KuaiRec](https://kuairec.com/) and [KuaiRand](https://kuairand.com/).
Educational only — not affiliated with any employer.

## Model progression (primary — Rank)

| Status | Stage | Model | Paper | Dataset |
|--------|-------|-------|-------|---------|
| ✅ done | Baseline ranker | Wide & Deep | [Cheng et al., DLRS 2016](https://arxiv.org/abs/1606.07792) | KuaiRec |
| ⬜ next | Multitask | MMoE | [Ma et al., KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3220007) | KuaiRand-Pure |
| ⬜ | Task balancing | GradNorm | [Chen et al., ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | KuaiRand-Pure |
| ⬜ | Sequence attention | DIN | [Zhou et al., KDD 2018](https://arxiv.org/abs/1706.06978) | KuaiRand-1K |
| ⬜ | Long-sequence search | SIM | [Pi et al., KDD 2020](https://arxiv.org/abs/2006.05639) | KuaiRand-1K |

## Additive — Recall + offline eval

| Status | Stage | Method | Paper / tool | Dataset |
|--------|-------|--------|--------------|---------|
| ⬜ | Sequential recall | SASRec → FAISS Top-K | [Kang & McAuley, ICDM 2018](https://arxiv.org/abs/1808.09781) | KuaiRand-1K |
| ⬜ | Bias-aware eval | IPS on random exposure | [Schnabel et al., RecSys 2016](https://arxiv.org/abs/1602.05352) | KuaiRand-Pure |

Rank metrics: **AUC / MAE**. Recall metrics: **Recall@K / NDCG@K**.

Each model is a standalone `nn.Module` with a cited paper docstring, shape-tested
before training on real data. Full build log, per-phase: [`docs/milestones.md`](docs/milestones.md).

## What's implemented

- **`data/kuairec_preprocessor.py`, `data/kuairec_dataset.py`** — KuaiRec ETL
  (temporal split) + parquet reader/batcher
- **`models/wide_deep.py`, `training/`, `experiments/wide_deep_kuairec.py`** —
  Wide & Deep (+ its ablations) on KuaiRec, `watch_ratio` regression
- **`data/scripts/download_datasets.py`** — local download helper for KuaiRec / KuaiRand

Model and training code for later stages (MMoE onward) are added incrementally
as each paper is implemented — see [`docs/milestones.md`](docs/milestones.md).

## Quick start

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
`torch.optim.AdamW` for training; FAISS when the recall stage lands.

## License

Code: MIT — see [LICENSE](LICENSE). Datasets retain their original licenses from the
KuaiRec / KuaiRand authors.
