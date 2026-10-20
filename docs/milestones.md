# Progress

1. **EDA and data protocol:** label/feature audit, signature deduplication,
   fixed big-only pair split, and train-fitted preprocessing.
   [Findings](../data/kuairec_eda_findings.md).
2. **Training settings and diagnostics:** completed exploratory budget investigation;
   adopted a 120-epoch cap, fixed-checkpoint diagnostics, and best-validation-MAE
   selection. [Recipe and rationale](datasets.md#training-recipe-and-budget).
3. **Ranking architecture study — complete:** capped-label Linear/Wide/Deep/W&D
   comparison with matched seeds and frozen small-matrix evaluation.
   Deep was strongest among the tested recipes. [Results](ranking_architecture.md).

4. **Retrieval and integration — complete:** popularity, cooccurrence and BPR MF
   feed the frozen capped Deep ranker. Five matched seeds preserve recall candidates
   and evaluate full-pool cap-2 NDCG@10. [Results and limitations](recall_rank.md).

5. **Deep feature study — complete:** six feature ablations with bounded confirmation;
   [standalone and fixed-recall results](ranking_features.md). Removal of duration
   band improved ordering and feed NDCG, with a small MAE cost.
6. **Channel merging — exploratory:** [optional composition and fixed-budget results](recall_channels.md).
   Round-robin, rank fusion and MF-heavy allocation share the existing pipeline;
   MF remains the single-retriever reference.

Historical small-matrix temporal-split conclusions are archived and do not
establish big-to-small performance.
