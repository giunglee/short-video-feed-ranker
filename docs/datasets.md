# Data

Current experiment: [KuaiRec](https://kuairec.com/), stored in `data/raw/kuairec/`.
Use `big_matrix.csv` for training/validation and `small_matrix.csv` for final test.
Preparation groups big user–video pairs into a fixed 90/10 train/validation split
(seed 2026), keeping repeated events together. It rejects overlap with small and
retains undated test rows. This tests pair generalization, not future prediction.
Shared users/items are expected; encodings and mean/median baselines fit train only.

The explicit feature spec supports IDs, duration, selected user profiles and binary
equality crosses. Daily engagement aggregates are excluded. Review profile timing
before use. Unknown/missing categories map to index 0 (a normal embedding row);
dense missing values become zero and dense model inputs use log1p(max(x, 0)).
The loader caches selected training columns for shuffling and streams evaluation.
Batch metadata retains `raw_user_id` and `raw_video_id`; encoded IDs remain model
features. `ConstantRegressor` fits mean/median predictions from prepared training
labels; runs save those predictions for final evaluation without refitting.

This is an offline modeling pipeline: public CSV snapshots → prepared parquet →
tensor batches. CSV parsing happens during preparation, not on every training step.
It does not implement online feature lookup, event ingestion, or serving infrastructure.
A production adaptation would need point-in-time feature availability and consistent
training/serving transformations; those are outside this experiment's scope.

MAE is event-weighted on raw watch ratio. Global XAUC samples unequal-label pairs;
within-user concordance averages eligible users equally. Prediction ties score 0.5;
users without unequal-label pairs are excluded and counted. Metric sampling uses a
fixed seed, independent of training seeds. Seed variation is not test-population uncertainty.
Historical small-label use prevents claiming a never-inspected test set.

Download/setup: [data README](../data/README.md).
Dataset paper: [KuaiRec (CIKM 2022)](https://doi.org/10.1145/3511808.3557220).

[KuaiRand](https://kuairand.com/) is a later option when an experiment requires it;
no additional dataset preparation is needed now.
