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

Historical small-matrix temporal-split conclusions are archived and do not
establish big-to-small performance. Retrieval and Recall → Rank evaluation remain
a separate milestone.
