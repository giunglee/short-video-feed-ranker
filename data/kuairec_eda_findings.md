# KuaiRec EDA findings

Results from the feature/label analysis behind this repo's Wide & Deep
baseline on KuaiRec (`watch_ratio` regression). The runnable analysis is
[`kuairec_eda.ipynb`](kuairec_eda.ipynb) in this directory.

## Label — `watch_ratio` (`small_matrix.csv`, pre-split)

| | value |
|---|---|
| rows (after dropping missing timestamp) | 4,494,578 (3.89% dropped) |
| mean / std | 0.911 / 1.356 |
| median / p90 / p99 | 0.775 / 1.563 / 3.585 |
| frac > 1 (finished the video at least once) | 32.7% |
| predict-mean MAE baseline | 0.487 |

Heavy right tail (max 571, p99 already 3.6x the mean) → `log1p` on the label
distribution plot for the deep path; still evaluate MAE in the original
scale, not log-space.

## Categorical redundancy (Cramér's V + bijective crosstab)

Cheap broad screen (Cramér's V over all pairs) → expensive exact check
(does level `a` determine level `b`, or vice versa) only on the pairs the
screen flags. 46 column pairs cleared Cramér's V > 0.9 out of the full
`CAT_COLS` set; the bijective check split them into:

- **Exact duplicates (dropped):**
  - `is_live_streamer` == `is_live_author` (identical on every row)
  - `user_onehot_feat{0,1,2,3,5,6,7,8,9,10,11}` are anonymized re-encodings
    of `gender`, `age_range`, `phone_brand`, `phone_model`, `fre_country`,
    `fre_country_region`, `fre_province`, `fre_city`, `fre_city_level`,
    `fre_community_type`, `platform` respectively (>=99.6% exact 1:1
    mapping each way) — kept the human-readable column, dropped the
    anonymized twin.
- **Flagged but kept:** `is_video_author` / `is_photo_author` are also
  bijective in this sample, but it's only 1,411 unique users and the two
  are semantically distinct concepts — re-check on `big_matrix` or KuaiRand
  before assuming true duplication. `user_onehot_feat4` (price-tier bucket)
  and `feat12–17` are weak but not proven redundant — kept.
- **Constant (dropped):** `video_type`, `is_lowactive_period` (`nunique == 1`
  in this sample).

Net: `CAT_COLS` went from the full raw categorical set down to 34 columns
(see `experiments/wide_deep_kuairec.py`).

## Cross features — interaction lift

`interaction_lift = lift(a, b jointly) / (lift(a) * lift(b))` separates true
synergy/conflict from lift that's just one strong marginal riding along.
`≈ 1` → no cross term needed, a linear model already gets it from `a` and
`b` separately.

**Dropped** (hand-picked originally, failed the interaction bar):

| cross | interaction_lift |
|---|---|
| `cross_new_teenager_female_users` (gender=F, age 12-17, register 31-60d) | 1.017 |
| `cross_young_male_active_users` (gender=M, age 24-30, full_active) | 1.007 |

**Kept** (final `WIDE_CROSS_SPECS`, mined via a systematic scan over
low-cardinality categoricals, `kuairec_eda.ipynb` §4b):

| cross | interaction_lift | support |
|---|---|---|
| `cross_high_friend_tier2_city_users` | 1.63 | 0.14% (n=6,380) |
| `cross_unknown_active_recent_users` | 1.46 | 0.07% (n=3,183) |
| `cross_socially_active_new_users` | 1.46 | 0.14% (n=6,346) |
| `cross_high_follow_high_friend_users` | 1.46 | 0.14% (n=6,443) |
| `cross_midage_high_friend_users` | 1.39 | 0.14% (n=6,430) |
| `cross_old_middle_active_users` | 0.75 | (kept from original hand-picked set) |
| `cross_isolated_users` | 0.72 | 0.07% (n=3,146) |

The two below-1.0 crosses are conflict segments — the joint group
under-performs what its marginals predict — which is as useful for the wide
path to memorize as an over-performing one.

**Candidates rejected despite a strong scan score** — the scan ranks pairs
independently, so it can surface near-duplicates of what's already picked:

- `follow_user_num_range=0` combined with `register_days_range=31-60`,
  `onehot_feat13`, `onehot_feat14`, or `user_active_degree=high_active` all
  scored well, but every one of them is the *exact same* 3,146 rows as
  `cross_isolated_users` (100% overlap) — one tiny, internally-homogeneous
  user cohort surfacing 5 times, not 5 independent signals.
- `gender=F` × `friend_user_num_range=[30,60)` (interaction_lift=1.34,
  n=9,581) fully contains `cross_high_follow_high_friend_users`'s 6,443 rows
  — adding it would just duplicate an existing cross's memorization capacity
  over a broader population, not add new information.
- Anything crossing on an anonymized `onehot_feat{12,13,14,16,17}` level
  (e.g. `follow_user_num_range=(50,100]` × `onehot_feat16=1.0`,
  interaction_lift=1.52) was skipped on principle — we don't know what those
  levels mean, so a cross on them isn't a checkable, interpretable memorized
  concept.

## Known data-quality issues — found and fixed

### 1. Item-daily join silently collapsed to one snapshot per video

`KuaiRecPreprocessor._extract` parsed `date` with plain `pd.to_datetime(...)`
on a raw numeric `YYYYMMDD` column (e.g. `20200705.0`). Pandas reads a bare
number as nanoseconds since the Unix epoch, so *every* row — in both
`small_matrix.csv` and `item_daily_features.csv` — silently collapsed to
`1970-01-01`. `items.drop_duplicates(subset=["video_id", "date"], keep=
"last")` then had only one "date" per video, so it kept a single arbitrary
daily snapshot per video and the merge applied that one snapshot to every
interaction with that video, regardless of the interaction's real date.
Verified by checking `video_daily_show_cnt` was literally constant within
`video_id` across the whole extract (0/2,993 videos had >1 distinct value).

This is a uniform bug — every split gets the same wrong-but-constant value —
so the split audit (which compares splits to each other) couldn't see it;
only a within-split sanity check (group by ID, count distinct values) could.
**Fix:** parse both `date` columns with the explicit `format="%Y%m%d"`. Post-fix,
2,907/2,922 videos show >1 distinct daily value (mean 19.4 distinct days).

### 2. NaN-timestamp rows all landed in `test`

The split audit (`kuairec_eda.ipynb` §5) compares missing rate and mean/std
across `train`/`val`/`test.parquet`. On the pre-fix extract, 52 columns —
almost all `video_daily_*` engagement counts plus
`author_id`/`music_id`/`video_tag_id` — showed a large missing-rate jump in
`test` only.

**Root cause:** `KuaiRecPreprocessor._load` sorts by `timestamp` before the
80/10/10 split; pandas' default `na_position="last"` pushed every row with a
missing `timestamp` (≈3.9% of rows) to the tail — straight into `test`. Those
rows then got `fillna(0.0)` downstream, so `test` saw a batch of fake
all-zero "engagement" features that neither `train` nor `val` ever saw.
**Fix:** drop rows with a missing `date`/`timestamp` in `_extract`, before
any join or sort. Re-running the audit post-fix: missing-rate drift >2% is
now 0 columns (down from 52).

### 3. `collect_cnt` family — real dataset characteristic, not a bug

`video_daily_collect_cnt` / `cancel_collect_{cnt,user_num}` / `collect_user_num`
are ~49% null in `train` vs 0% in `val`/`test` even after the fixes above.
KuaiRec didn't track "collect" (bookmark) actions for the first 23 days of
its 63-day window (`item_daily_features.csv` is 100% null for `collect_cnt`
from 2020-07-05 to 2020-07-27); `train` is the earliest 80% chronologically,
so it inherits that gap. Excluded from `DENSE_COLS` for now rather than
imputed with a fake 0.