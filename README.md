# short-video-feed-ranker

From-scratch **short-video feed ranking** in PyTorch — paper-based reimplementations
of the model stack used in production-style feed rankers: baseline pointwise ranker →
multi-task MMoE → long-sequence attention (DIN / SIM).

Trained on public [KuaiRec](https://kuairec.com/) and
[KuaiRand](https://kuairand.com/) data. Educational only — not affiliated with any employer.

## Model progression

| Stage | Model | Paper | Dataset |
|-------|-------|-------|---------|
| Baseline ranker | Wide & Deep | [Cheng et al., DLRS 2016](https://arxiv.org/abs/1606.07792) | KuaiRec |
| Multitask | MMoE | [Ma et al., KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3220007) | KuaiRand-Pure |
| Task balancing | GradNorm | [Chen et al., ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | KuaiRand-Pure |
| Sequence attention | DIN | [Zhou et al., KDD 2018](https://arxiv.org/abs/1706.06978) | KuaiRand-1K |
| Long-sequence search | SIM | [Qi et al., CIKM 2020](https://arxiv.org/abs/2006.14151) | KuaiRand-1K |

Each model is a standalone `nn.Module` with a cited paper docstring, tested first on
`synthetic.py` batch shapes, then on real data.

## What's implemented

- **`data/synthetic.py`** — batch generator for CTR, watch, multitask, sequence, and
  related task shapes (shape tests before real-data training)
- **`data/scripts/download_datasets.py`** — local download helper for KuaiRec / KuaiRand

Model and training code are added incrementally as each paper is implemented.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify batch shapes (no data download needed)
python data/synthetic.py

# Download datasets when training on real data (~5 GB)
python data/scripts/download_datasets.py --all
```

Data setup: [`docs/datasets.md`](docs/datasets.md)

## Stack

PyTorch `nn.Module` for all models; `pandas` / `numpy` only in data loaders;
`torch.optim.AdamW` for training.

## License

Code: MIT — see [LICENSE](LICENSE). Datasets retain their original licenses from the
KuaiRec / KuaiRand authors.
