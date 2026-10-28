# Data setup

Raw CSVs are **not committed**. Download locally before training on real data.

```bash
python scripts/download_datasets.py --all    # ~5 GB: KuaiRec + Pure + 1K
```

Manual links and per-dataset flags: run `python scripts/download_datasets.py --help`.

Extracted files go in `data/raw/kuairec/`, `data/raw/kuairand_pure/`, or
`data/raw/kuairand_1k/` (flat layout — no nested `data/` folder).

Which dataset for which model: [`docs/datasets.md`](../docs/datasets.md).

## Preparation, training and checks

Use the [current KuaiRand workflow](../docs/kuairand_workflow.md) and
[raw-source audit](../docs/raw_audit.md). Preparation fits transformations on
training data; current validation commands do not score the final window.

The completed KuaiRec notebooks, reports and their old commands require the
[pinned historical source](https://github.com/giunglee/short-video-feed-ranker/tree/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5). They are preserved as experiment evidence;
the active package no longer ships the old KuaiRec training pipeline.
