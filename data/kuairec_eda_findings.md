# KuaiRec label and feature analysis

**Question:** can a model trained on sparse interactions generalize to held-out
user–video pairs?

- **Outcome:** retain raw labels, remove complete duplicate events, and start with
  IDs/duration plus MAE. Treat additional features as hypotheses to test.
- **Evidence:** [visual notebook](kuairec_eda.ipynb).
- **Boundary:** small-matrix label inspection describes distribution shift;
  feature selection uses big training/validation only.

## Findings and decisions

- **Distribution shift:** big and small matrices have different label distributions.
  Report validation and final-test performance separately; dense observations do
  not establish randomized exposure or causal effects.
- **Heavy-tailed labels:** a large ratio can arise from a very short recorded video
  or long play duration. Logs cannot resolve session/replay intent. Retain raw
  labels and use MAE as the initial loss: large residuals have linear rather than
  quadratic cost. This does not make extreme labels correct or unimportant.
- **Repeated observations:** remove complete duplicate signatures (user, video,
  timestamp, play duration, video duration, watch ratio), not all repeated pairs.
  Keep distinct-time events and incomplete signatures. Event weighting is the
  reference; inverse-pair-frequency weighting is a separate training ablation.
- **Dates and duration:** daily event counts describe released dated-row coverage,
  not platform traffic. Undated rows remain in other analyses. Duration composition
  and label means can co-vary without establishing causation. Short-duration
  cohorts need explicit error analysis because duration is the ratio denominator.
- **Features:** use IDs and duration as the reference. Marginal lift, categorical
  redundancy, and departures from additive group means screen candidates; they
  do not prove incremental predictive value. Validate additions and removals
  through controlled ablations on big validation.
- **Availability:** snapshot profiles lack verified historical availability.
  Evaluate them separately and disclose this limitation. Daily item attributes
  require an availability audit; captions and sequence modeling are deferred.

## What follows from this analysis

- Keep user–video pairs together in the fixed big-only split; fit encodings on train.
- Compare event versus pair weighting, then feature additions, under the
  [declared training and selection rules](../docs/datasets.md).
- Inspect user activity, item support, duration, and label-tail cohorts alongside
  aggregate results. Associations do not establish their causes.
- Confirm gains across matched seeds before final test evaluation. Historical
  small-matrix temporal-split conclusions do not establish performance here.

Generated reports remain local; this page records the public findings and rationale.
