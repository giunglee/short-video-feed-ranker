# Data setup

Raw CSVs are **not committed**. Download locally before training on real data.

```bash
python scripts/download_datasets.py --all    # ~5 GB: KuaiRec + Pure + 1K
```

Manual links and per-dataset flags: run `python scripts/download_datasets.py --help`.

Extracted files go in `data/raw/kuairec/`, `data/raw/kuairand_pure/`, or
`data/raw/kuairand_1k/` (flat layout — no nested `data/` folder).

Which dataset for which model: [`docs/datasets.md`](../docs/datasets.md).

KuaiRec descriptive audit: `python -m feed_ranker.experiments.eda_kuairec --help`.
Current conclusions: [label and feature findings](kuairec_eda_findings.md).

## Visual EDA

Open `data/kuairec_eda.ipynb` with the project Python environment. Install notebook
support with `pip install -e ".[eda]"`. Run from the repository or a subdirectory.
The notebook reads full matrices and uses the fixed big-only split for candidate
feature analysis. It writes figures and CSVs to ignored `artifacts/eda_review/`.
Label comparisons with small_matrix are descriptive; cross screens use big train
only. Snapshot timing and feature selection require review before model changes.

## Prepare and train

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m feed_ranker.data.kuairec_preprocessor --features configs/wide_deep_features.json --output data/extract/kuairec_v2
# Smoke test only
python -m feed_ranker.experiments.wide_deep_kuairec train --data data/extract/kuairec_v2 --output artifacts/wide_deep_v2/debug --model wide_deep_nn --epochs 2 --steps 10
# Example architecture comparison; agree on candidates before launching
python -m feed_ranker.experiments.multi_seed --data data/extract/kuairec_v2 --output artifacts/wide_deep_v2/comparison --epochs 120
```

- Prepare once and reuse the fixed split. The comparison command defaults to
  Deep and Wide & Deep, seeds 42–46; it is not the entire planned ablation matrix.
- Review [the frozen recipe](../docs/datasets.md#training-recipe-and-budget).
  The CLI defaults to 15 epochs; the agreed budget requires `--epochs 120`.
- Loss, reported metrics, and selection are independent options: `--loss`,
  `--metrics`, `--selection-metric`, `--selection-mode`. See `train --help`.
- Compare per-seed validation results and paired differences. Lower MAE and higher
  concordance are better. Use `python -m feed_ranker.experiments.plot_run --help`
  for curves.

## Final evaluation and checks

Freeze choices on validation before evaluating each seed's selected checkpoint:

```bash
python -m feed_ranker.experiments.wide_deep_kuairec evaluate --data data/extract/kuairec_v2 --checkpoint artifacts/wide_deep_v2/comparison/wide_deep_nn-42/best.pt --output artifacts/wide_deep_v2/comparison/wide_deep_nn-42/test.json
ruff check src tests scripts
ruff format --check src tests scripts
python -m unittest discover -s tests -v
```

- Only scalar regression is currently supported; recall remains paused.
- `run_protocol.py` preserves the earlier full-run recipe and automatically scores
  test. Use the separate commands above for current validation-only ablations.
