# short-video-feed-ranker

A PyTorch study of short-video recommendation using public KuaiRec and KuaiRand
data. The focus is modeling judgment: valid comparisons, failure analysis and
reproducible conclusions. Educational; not affiliated with any employer.

## Current implementation

[KuaiRand workflow and commands](docs/kuairand_workflow.md) is the entry point.
Standard-tab-1 events train a Deep `long_view` ranker; popularity, sparse cosine
and BPR MF provide candidates for retrieval, merging and ranking.

- **Preprocessing:** [raw validation, splits and preparation](src/feed_ranker/preprocessing/kuairand.py),
  [train-fitted features](src/feed_ranker/preprocessing/features.py).
- **Data:** [Parquet split reading and validation](src/feed_ranker/data/parquet.py)
  and [tensor datasets/loaders](src/feed_ranker/data/dataset.py).
- **Models:** [Deep ranking](src/feed_ranker/models/ranking.py) and
  [BPR / sparse cosine recall](src/feed_ranker/models/recall.py).
- **Training:** [named objectives](src/feed_ranker/training/objectives.py) and
  [updates, metrics and checkpoints](src/feed_ranker/training/trainer.py).
- **Evaluator:** [prediction records and independent metrics](src/feed_ranker/evaluation/predictions.py).
- **Recommendation:** [retrieval → ranking → trace](src/feed_ranker/recommendation/pipeline.py)
  from [self-contained fitted artifacts](src/feed_ranker/artifacts.py).

[Raw EDA](docs/raw_audit.md) runs independently. The external
[study runner](experiments/kuairand.py) composes these interfaces;
[study configurations and commands](docs/kuairand_baselines.md) explain the comparison.

## Evidence

- [KuaiRec data findings](data/kuairec_eda_findings.md),
  [ranking comparisons](docs/ranking_architecture.md), and
  [feature ablations](docs/ranking_features.md).
- [KuaiRand matched-seed recall and feature study](docs/kuairand_confirmation.md).
- [Retrieval findings](docs/recall_decisions.md) and
  [retrieval/ranking trade-offs](docs/recall_rank.md).

The rearchitecture retires superseded loaders, training paths, feature frameworks
and experiment executables. Those historical studies remain reproducible from
[their source revision](https://github.com/giunglee/short-video-feed-ranker/tree/033fdc90d89a1c8b9dc5a6dbb6940290762af3a5); their reports and results remain here.
Historical artifact readers also live in that pinned checkout; the active package
loads current fitted artifacts. See the workflow guide for format and resume limits.

## Run and check

```sh
python -m pip install -e '.[dev]'
python -m feed_ranker.cli --help
python -m ruff check src tests scripts experiments
python -m ruff format --check src tests scripts experiments
python -m unittest discover -s tests -v
```

[Dataset downloads](data/README.md). No real-data training or final-window scoring
is required by the software checks. Snapshot feature timing remains unverified;
software correctness does not establish model quality, 27K scale or online latency.
Future model families require their own experiment question and protocol.

Code: [MIT](LICENSE); datasets retain their authors' licenses.
