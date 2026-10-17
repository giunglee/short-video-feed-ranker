# short-video-feed-ranker

A PyTorch study of short-video recommendation using public KuaiRec data.
The focus is **modeling judgment: data quality, valid comparisons, failure analysis,
and reproducible conclusions**. Educational; not affiliated with any employer.

## The question

Can Wide & Deep trained on sparse interactions generalize to unseen user–video
pairs, and which weighting, feature, and architecture choices improve it?

## What we have learned

- **Clean observations before comparing models.** Remove complete duplicate events,
  retain distinct-time repeats, and keep each user–video pair in one split.
- **Inspect the label before choosing the loss.** Extreme watch ratios mix short
  video durations with long play times; retain raw labels and start with MAE.
- **Measure generalization at the same checkpoint.** Compare validation with a
  fixed training sample in evaluation mode, rather than training-time loss.
- **Declare a practical budget.** Single-seed diagnostics support a 120-epoch cap
  with best-validation-MAE selection. Multi-seed model gains are not yet established.

## Evidence and next steps

1. **EDA — complete:** [label/feature findings](data/kuairec_eda_findings.md)
   and [visual notebook](data/kuairec_eda.ipynb).
2. **Training diagnostics — complete:** [settings, evidence, and budget rationale](docs/datasets.md#training-recipe-and-budget).
3. **Ablations — next:** compare weighting, features, and architectures with matched
   seeds, then evaluate frozen choices on small-matrix test data.

**Limits:** the split tests pair generalization, not future-time prediction.
Snapshot feature timing is unverified. Historical test-label inspection is
explicitly disclosed. This repository studies offline modeling, not online serving.

## Review the implementation

- **Data contracts and preparation:** [schema](src/feed_ranker/data/schema.py),
  [configuration](src/feed_ranker/data/config.py),
  [preprocessor](src/feed_ranker/data/kuairec_preprocessor.py),
  [loader](src/feed_ranker/data/kuairec_dataset.py).
- **Model:** [Wide & Deep](src/feed_ranker/models/wide_deep.py) and its
  [feature encoders](src/feed_ranker/models/components/feature_encoder.py).
- **Training and diagnosis:** [trainer](src/feed_ranker/training/trainer.py),
  [evaluation](src/feed_ranker/training/evaluation.py),
  [metrics](src/feed_ranker/training/metrics.py),
  [cohort traces](src/feed_ranker/experiments/diagnostics.py).
- **Experiments:** [single run](src/feed_ranker/experiments/wide_deep_kuairec.py)
  and [matched-seed comparison](src/feed_ranker/experiments/multi_seed.py).

## Future directions

- **Next:** one learned retrieval (recall) experiment after ranking analysis.
- **Later:** multi-task ranking, DIN/SASRec-style behavior modeling, graph-based
  recommendation, and incremental-value modeling.
- These are research directions, not implemented capabilities. Each becomes an
  experiment only after defining its question, baseline, and evaluation protocol.

## Run and reproduce

- [Setup, training commands, and checks](data/README.md).
- [Data protocol and frozen training recipe](docs/datasets.md).
- [Progress](docs/milestones.md). Recall remains paused until ranking comparisons
  and their interpretation are complete.

Reference: [Wide & Deep, Cheng et al.](https://arxiv.org/abs/1606.07792).
Code: [MIT](LICENSE); datasets retain their authors' licenses.
