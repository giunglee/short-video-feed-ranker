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
- **Inspect the label before choosing the objective.** Extreme watch ratios mix
  short video durations with long play times. Start with raw-label MAE, then
  explicitly test bounded utility using capped-label MAE.
- **Measure generalization at the same checkpoint.** Compare validation with a
  fixed training sample in evaluation mode, rather than training-time loss.
- **Compare architectures without requiring a preferred winner.** In a bounded
  five-seed study, Deep `[64, 32]` beat the selected Wide & Deep recipe on capped
  user-XAUC. Its test advantage over Wide was consistent but small.

## Evidence and next steps

1. **EDA — complete:** [label/feature findings](data/kuairec_eda_findings.md)
   and [visual notebook](data/kuairec_eda.ipynb).
2. **Training diagnostics — complete:** [settings, evidence, and budget rationale](docs/datasets.md#training-recipe-and-budget).
3. **Ranking architecture comparison — complete:** [protocol, results, limitations,
   and reproduction](docs/ranking_architecture.md). Four model families, 37 training
   runs, and 20 frozen finalists; capped validation user-XAUC selects checkpoints.

4. **Retrieval and final-feed integration — complete:** [recall methods](docs/recall_decisions.md)
   and [frozen Deep funnel evaluation](docs/recall_rank.md). MF leads the retrieval
   baselines across five matched seeds; standalone ranking gains do not guarantee
   higher final-feed NDCG.
5. **Deep feature ablations — complete:** [feature results and reproduction](docs/ranking_features.md).
   Removing duration band while retaining log duration improved user-XAUC and
   fixed-candidate feed NDCG across five seeds, with a small MAE regression.
6. **Channel merging — exploratory:** [reusable composition and findings](docs/recall_channels.md).
   MF-heavy allocation reduces merging losses; MF alone remains the quality reference.

**Limits:** the split tests pair generalization, not future-time prediction.
Snapshot feature timing is unverified. Historical test-label inspection is
explicitly disclosed. This repository studies offline modeling, not online serving.

## Review the implementation

- **Data contracts and preparation:** [schema](src/feed_ranker/data/schema.py),
  [feature configuration](src/feed_ranker/features/config.py),
  [preprocessor](src/feed_ranker/data/kuairec_preprocessor.py),
  [loader](src/feed_ranker/data/kuairec_dataset.py).
- **Model:** [Wide & Deep](src/feed_ranker/models/wide_deep.py) and its
  [feature encoders](src/feed_ranker/models/components/feature_encoder.py).
- **Training and diagnosis:** [trainer](src/feed_ranker/training/trainer.py),
  [evaluation](src/feed_ranker/evaluation/ranking.py),
  [metrics](src/feed_ranker/evaluation/metrics.py),
  [cohort traces](src/feed_ranker/experiments/diagnostics.py).
- **Recommendation:** [pipeline and contracts](src/feed_ranker/recommendation),
  [inference adapters](src/feed_ranker/inference), [candidate features](src/feed_ranker/features/kuairec.py),
  [funnel evaluator](src/feed_ranker/evaluation/funnel.py).
- **Experiments:** [architecture construction](src/feed_ranker/experiments/ranking_architecture.py)
  and [bounded study](src/feed_ranker/experiments/ranking_study.py).

## Future directions

- **Next:** channel-merging experiments, separate from the frozen ranking comparisons.
- **Later:** multi-task ranking, DIN/SASRec-style behavior modeling, graph-based
  recommendation, and incremental-value modeling.
- These are research directions, not implemented capabilities. Each becomes an
  experiment only after defining its question, baseline, and evaluation protocol.

## Run and reproduce

- [Setup, training commands, and checks](data/README.md).
- [Data protocol and frozen training recipe](docs/datasets.md).
- [Progress](docs/milestones.md). Retrieval and funnel evaluation are a separate milestone.

Reference: [Wide & Deep, Cheng et al.](https://arxiv.org/abs/1606.07792).
Code: [MIT](LICENSE); datasets retain their authors' licenses.

KuaiRand WIP: [matched-seed recall merging and feature ablations](docs/kuairand_confirmation.md),
with [temporal recall and Deep setup](docs/kuairand_baselines.md).
