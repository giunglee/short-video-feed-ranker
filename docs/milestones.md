# Progress

1. **EDA and data protocol:** label/feature audit, signature deduplication,
   fixed big-only pair split, and train-fitted preprocessing.
   [Findings](../data/kuairec_eda_findings.md).
2. **Training settings and diagnostics:** completed exploratory budget investigation;
   adopted a 120-epoch cap, fixed-checkpoint diagnostics, and best-validation-MAE
   selection. [Recipe and rationale](datasets.md#training-recipe-and-budget).
3. **Next — controlled ablations:** weighting, features, and architectures with
   matched seeds, followed by final small-matrix evaluation and interpretation.

Historical small-matrix temporal-split conclusions are archived and do not
establish big-to-small performance. Recall remains paused.
