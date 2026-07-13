# Datasets

This repo uses two public short-video datasets. Full documentation and file schemas
are on the official homepages — we only note **which dataset pairs with which model
stage** here.

| Dataset | Official site | Used for |
|---------|---------------|----------|
| **KuaiRec** | [kuairec.com](https://kuairec.com/) | Wide & Deep baseline (fully observed watch matrix) |
| **KuaiRand-Pure** | [kuairand.com](https://kuairand.com/) | MMoE multitask (`long_view`, `is_like`, `play_time_ms`) |
| **KuaiRand-1K** | [kuairand.com](https://kuairand.com/) | DIN / SIM (long per-user feed sequences) |

Skip KuaiRand-27K (~46 GB) — same schema as 1K.

## Download

```bash
python data/scripts/download_datasets.py --all
```

Files land in `data/raw/{kuairec,kuairand_pure,kuairand_1k}/`. See [`data/README.md`](../data/README.md)
for options and manual links.

## Citations

KuaiRec (CIKM 2022): https://doi.org/10.1145/3511808.3557220

KuaiRand (CIKM 2022): https://doi.org/10.1145/3511808.3557624
