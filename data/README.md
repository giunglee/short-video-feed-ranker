# Data setup

Raw CSVs are **not committed**. Download locally before training on real data.

```bash
python data/scripts/download_datasets.py --all    # ~5 GB: KuaiRec + Pure + 1K
```

Manual links and per-dataset flags: run `python data/scripts/download_datasets.py --help`.

Extracted files go in `data/raw/kuairec/`, `data/raw/kuairand_pure/`, or
`data/raw/kuairand_1k/` (flat layout — no nested `data/` folder).

Which dataset for which model: [`docs/datasets.md`](../docs/datasets.md).

KuaiRec feature/label EDA: [`kuairec_eda.ipynb`](kuairec_eda.ipynb) (runnable
analysis) and [`kuairec_eda_findings.md`](kuairec_eda_findings.md) (redundancy,
cross-feature selection, known split issue).
