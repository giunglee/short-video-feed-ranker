# KuaiRec EDA status

The previous analysis used temporal splits of `small_matrix` and is historical.
Its conclusions about ID embeddings, feature redundancy, and wide crosses are
not established for training on `big_matrix`.

Current work will audit both matrices, then make supervised feature decisions
using big-matrix training/validation only. Small-matrix label summaries describe
distribution shift; they must not drive model selection.

The current preprocessing and feature configuration still implement the old
protocol. Do not regenerate extracts or report them as corrected benchmark results.
New findings will be recorded here after the audit. Historical notes and notebook
are preserved locally; existing result artifacts are unchanged.
