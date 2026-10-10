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
Historical EDA and conclusions are archived locally; see the root README for the
current preparation and training commands.

## Visual EDA

Open `data/kuairec_eda.ipynb` with the project Python environment. Install notebook
support with `pip install -e ".[eda]"`. Run from the repository or a subdirectory.
The notebook reads full matrices and uses the fixed big-only split for candidate
feature analysis. It writes figures and CSVs to ignored `artifacts/eda_review/`.
Label comparisons with small_matrix are descriptive; cross screens use big train
only. Snapshot timing and feature selection require review before model changes.
