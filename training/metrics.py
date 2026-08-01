"""
Ranking-aware metric for continuous regression targets (e.g. watch_ratio).

Paper: XAUC is used (not introduced) by the KuaiRec benchmark paper
(Gao et al., "KuaiRec: A Fully-Observed Dataset and Insights for Evaluating
Recommender Systems", CIKM 2022) and reported alongside MAE by every public
watch-time-prediction follow-up that benchmarks on KuaiRec (e.g. Zhan et al.,
"Deconfounding Duration Bias in Watch-time Prediction for Video
Recommendation", KDD 2022 [metric origin]; CREAD, arXiv:2401.07521; TPM,
arXiv:2306.03392; FlowTime, arXiv:2606.01352).

Why report this alongside MAE: `watch_ratio` is consumed by the ranker to
*order* candidates for a user, not read as an absolute number. A model with
a systematic scale/offset bias can have mediocre MAE while still ranking
candidates perfectly (XAUC ~1.0); conversely a model can nail the global
mean while getting relative order wrong within a slate, which MAE alone
would not catch. Reporting both avoids over-indexing on either failure mode.
"""
from __future__ import annotations

import torch


def xauc(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_pairs: int = 100_000,
    seed: int = 0,
) -> float:
    """Sampled pairwise concordance rate between `preds` and `labels`.

    Uniformly samples pairs (i, j) with labels[i] != labels[j] (true ties
    carry no ranking signal, so they're excluded -- matching how XAUC is
    defined in the papers above) and returns the fraction where the
    predicted order agrees with the true order. Predicted ties
    (preds[i] == preds[j]) count as 0.5, matching the standard AUC
    tie-handling convention.

    Exact (all-pairs) XAUC is O(n^2) and infeasible at eval-set scale (e.g.
    449K rows -> ~10^11 pairs); sampling `num_pairs` pairs is the standard
    practice in the literature above and gives a low-variance estimate for
    num_pairs in the tens of thousands.
    """
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
        # Oversample 2x since pairs landing on a label tie get discarded.
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
