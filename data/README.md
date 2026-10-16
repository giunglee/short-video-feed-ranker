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
