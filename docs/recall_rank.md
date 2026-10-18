# Two-stage recommendation: retrieve candidates, then rank the feed

**Retained pipeline:** popularity, cosine item cooccurrence or BPR MF → frozen
capped Deep `[64, 32]` → top-10 feed. MF produces the strongest final feed among
these retrieval methods across five matched seeds. **The Deep upgrade itself is
not a demonstrated funnel improvement over historical capped Wide & Deep.**

**Observed held-out pool → Retrieve M unique videos → Fixed Deep ranker → Top 10.**

## What is fixed, and why?

| Boundary | Contract | Rationale |
|---|---|---|
| Candidates | Same saved retrieval scores and candidate identities; M=50 primary, M=20/100 sensitivity | Isolate the downstream ranker change; no recall retraining |
| Recall | First-event duration-adjusted binary labels; clear-negative BPR; validation Recall@50 checkpoint | Retrieve sufficiently viewed pairs without treating unknown exposure as a negative |
| Ranking | Event-weighted MAE on `min(2, watch_ratio)`; frozen architecture-study checkpoints, seeds 42–46 | Preserve the approved utility and independently selected recipe |
| Rank checkpoint | Maximum capped validation user-XAUC, earliest tie, within 120 epochs | No reselection on funnel NDCG or choosing the best seed |
| Candidate features | Saved vocabularies; training-only median video duration and derived band; profiles from prepared training records | No held-out watch outcome or event duration enters inference |
| Final evaluation | First-event **linear cap-2 gain**, full-pool IDCG, equal-user mean, K=10 | Count both retrieval misses and ranker ordering errors |

Deep uses ID, age-range, follow-count-range and duration-band embeddings (width 8),
log-duration input, hidden widths `[64, 32]`, and no dropout. See the
[architecture study](ranking_architecture.md). Profile snapshot timing remains
**unverified**: reconstructing the training inputs does not prove historical
recommendation-time availability.

For each user, `DCG@10 = sum(min(2, watch_ratio_i) / log2(i+1))`, for positions 1–10.
IDCG orders all eligible pool items by that gain, including unretrieved items.
Missing slots contribute zero. All 7,176 users have positive IDCG; binary Recall
excludes and reports the one user with no positive labels. There is no exponential gain.

## Frozen Deep results

Mean ± sample SD over seeds 42–46; the same ranker serves all methods within a seed.
Popularity/cooccurrence candidates are deterministic; MF and Deep use matched seeds.

| Retrieval | Recall@50 | Final cap-2 NDCG@10 | Candidate oracle NDCG@10 |
|---|---:|---:|---:|
| Popularity | .576896 ± .000000 | .717079 ± .000521 | .969125 ± .000000 |
| Cooccurrence | .581345 ± .000000 | .710230 ± .000428 | .965433 ± .000000 |
| **BPR MF** | **.623439 ± .000201** | **.737278 ± .000731** | **.986559 ± .000185** |

![Frozen Deep funnel comparison](figures/funnel_deep_validation.png)

- **MF − popularity:** +.020200 ± .000414 final NDCG; **MF − cooccurrence:**
  +.027048 ± .000580. Both paired differences are positive in all five seeds.
- MF final NDCG at M20/50/100: **.735640 / .737278 / .736687**. More candidate
  coverage does not guarantee a better ranked feed.
- Cooccurrence's higher binary recall than popularity still does not survive
  final ranking. Candidate oracle describes hindsight potential, not wholly learnable headroom.
- MF's advantage over popularity is +.034630 for pools >150 and +.007833 for
  pools 51–150. The ≤50-positive-history cohort (20 users) regresses −.000706;
  these are descriptive cohort means, not independent causal effects.
- The MF feed averages **9.781 of 10 slots at ≤10 seconds**. This pipeline does
  not resolve duration concentration or establish broader satisfaction.

[Per-seed results](results/funnel_deep_per_seed.csv),
[aggregate metrics](results/funnel_deep_means.csv), and
[paired seed differences](results/funnel_deep_paired.csv) contain the numeric evidence.

## Did changing the ranker improve the same candidates?

The historical capped W&D reference exists for seed 42 only. Compare that seed
with frozen Deep on **identical candidate lists and full-pool denominators**:

| Retrieval, M50 | Historical capped W&D | Frozen Deep | Difference |
|---|---:|---:|---:|
| Popularity | .717273 | .717270 | −.000003 |
| Cooccurrence | .710419 | .710204 | −.000215 |
| MF | .738430 | .738048 | −.000383 |

**Better standalone user-XAUC did not improve this top-10 funnel in seed 42.**
This comparison changes architecture, features and checkpoint selector together;
it cannot attribute the effect to Deep alone. The historical W&D checkpoint used
full-pool cap-2 NDCG selection; Deep keeps the architecture study's user-XAUC selection.
Do not compare one historical seed against the new five-seed mean as a matched effect.
[All budgets](results/funnel_deep_ranker_comparison_seed42.csv).

Keep the user-selected frozen Deep recipe as the integration reference, without
claiming a funnel lift. Feature additions/removals and their downstream impact
belong to a **separate experiment and PR**; they do not gate this integration.
Channel merging and its exploratory results are also outside this PR.

## Historical decisions

Capping bounds repeat-watch utility rather than deleting events or proving noise.
The earlier raw-to-cap2 seed-42 pilot accepted .740335 → .738430 for consistent
training/evaluation utility. MAE, user-XAUC and NDCG remain different objectives;
event-weighted training also differs from first-event, equal-user evaluation.

### Curve sensitivity

Historical **capped W&D**, fixed pools and ranker: adding neutral BPR comparisons
changed MF/M50 NDCG .738430 → .737139; changing the positive anchor 10s → 8s changed
it to .738225. Retain clear negatives and the 10s curve; these were exploratory
pilots, not Deep ablations or universal optima. The 8s arm had 9.36% more training
pairs at the same epoch budget. See [recall decisions](recall_decisions.md),
[curve evidence](results/recall_curve_funnel.csv), and [target pilot](results/target_pilot.csv).
Earlier raw-ranker five-seed evidence remains [historical](results/funnel_validation.csv).

## Verification and reproduction

- 64 isolated-tree tests and Ruff pass, including a synthetic Deep funnel with no
  test dataset, profile/OOV handling, duration boundaries, label-independent inputs
  and wrong-checkpoint rejection.
- All five checkpoints reproduce **8,192 saved validation predictions exactly**.
  Train/validation pair membership matches recall exactly despite different feature manifests.
- All 45 method/seed/budget candidate exports match the original recall lists;
  standalone Recall is unchanged. Seed-42 candidate-oracle values match the old
  capped funnel. Source snapshots, checkpoint/input hashes and per-user traces are saved.
- Validation only; no new test access or training. Dense observed pools, repeated
  validation use, historical test exposure and snapshot timing limit generalization.
  Seed SD measures training variation, not data uncertainty or online lift.

For each frozen Deep checkpoint, pair the matching MF seed with the shared baselines:

```sh
python -m feed_ranker.experiments.deep_funnel \
  --data PATH/TO/recall_data \
  --ranking-data PATH/TO/prepared_architecture_data \
  --checkpoint PATH/TO/frozen_deep/best.pt \
  --checkpoint-sha256 SHA256_FROM_FROZEN_FINALISTS \
  --reference-trace PATH/TO/saved_validation_trace.parquet \
  --runs PATH/TO/popularity PATH/TO/cooccurrence PATH/TO/matching_mf \
  --output artifacts/deep_funnel_seed42 --threads 1
```

The output includes keyed rank predictions and a provenance manifest, per-user
metrics, raw-ID candidates, duration diagnostics and a full-pool ranker reference.
The legacy `recall_rank` scorer remains restricted to its historical IDs/duration
models; do not pass profile-based Deep weights through that adapter.
