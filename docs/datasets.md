# Datasets

Public short-video datasets for this study repo. Schemas live on the official sites —
here we only map **dataset → model stage**.

| Dataset | Official site | Used for |
|---------|---------------|----------|
| **KuaiRec** | [kuairec.com](https://kuairec.com/) | Wide & Deep baseline (fully observed watch matrix) |
| **KuaiRand-Pure** | [kuairand.com](https://kuairand.com/) | MMoE multitask · IPS offline eval (random exposure) |
| **KuaiRand-1K** | [kuairand.com](https://kuairand.com/) | DIN / SIM · additive SASRec recall (long sequences) |

Skip KuaiRand-27K (~46 GB) — same schema as 1K.

## Download

```bash
python data/scripts/download_datasets.py --all
```

Files land in `data/raw/{kuairec,kuairand_pure,kuairand_1k}/`. See [`data/README.md`](../data/README.md).

## Citations

KuaiRec (CIKM 2022): https://doi.org/10.1145/3511808.3557220

KuaiRand (CIKM 2022): https://doi.org/10.1145/3511808.3557624
