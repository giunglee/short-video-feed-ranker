"""Sampled XAUC for continuous targets (e.g. watch_ratio).

Used alongside MAE by KuaiRec watch-time work (Gao et al. CIKM 2022;
Zhan et al. KDD 2022). Captures ranking order that MAE can miss.
"""
from __future__ import annotations

import torch


def xauc(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_pairs: int = 100_000,
    seed: int = 0,
) -> float:
    """Sampled pairwise concordance; label ties dropped, pred ties count 0.5."""
    preds = preds.detach().float().reshape(-1)
    labels = labels.detach().float().reshape(-1)
    n = labels.shape[0]
    if n < 2:
        raise ValueError(f"xauc needs >=2 samples, got {n}")

    gen = torch.Generator().manual_seed(seed)
    matched = 0.0
    counted = 0
    while counted < num_pairs:
        remaining = num_pairs - counted
        # Oversample: label-tie pairs are discarded.
        draw = max(remaining * 2, 1024)
        i = torch.randint(0, n, (draw,), generator=gen)
        j = torch.randint(0, n, (draw,), generator=gen)
        keep = labels[i] != labels[j]
        i, j = i[keep][:remaining], j[keep][:remaining]
        if i.numel() == 0:
            continue

        pred_diff = preds[i] - preds[j]
        true_gt = labels[i] > labels[j]
        correct = ((pred_diff > 0) == true_gt).float()
        agree = torch.where(pred_diff == 0, torch.full_like(pred_diff, 0.5), correct)

        matched += agree.sum().item()
        counted += i.numel()

    return matched / counted
