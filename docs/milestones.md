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

Feature exploration and its downstream evaluation are a separate follow-up.
Historical small-matrix temporal-split conclusions are archived and do not
establish big-to-small performance.
