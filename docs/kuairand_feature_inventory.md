# KuaiRand-1K feature inventory

[Back to data/protocol review](kuairand_data_review.md)

Full-file census: 19 event fields, 31 user fields, 12 basic-video fields and 52 monthly-statistics fields (114 total, including keys and labels). Cardinality is exact non-null distinct values, including unknown sentinels. Missing means SQL null; unknown means `-124` or literal `UNKNOWN`. Zero remains a value unless a field-specific rule says otherwise. Invalid counts use only the listed checks; zero invalid does not prove semantic validity.

Numeric quantiles are exact over finite, non-null, non-sentinel values, including zeros. Identity numbers have no meaningful magnitude; their quantiles are retained only in the JSON audit. Frequencies show every value for cardinality ≤100 and the top ten otherwise. Metadata counts are per entity, never event-weighted. No field is approved for model use merely by appearing here.

[Machine-readable census](results/kuairand_deep_eda.json). The live companion supports selecting a table and field.

## Event log: descriptive; outcomes are targets, not inputs

| Field | Cardinality | Missing n (%) | Unknown n (%) | Invalid n (%) |
|---|---|---|---|---|
| user_id | 1,000 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| video_id | 4,371,868 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| date | 31 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| hourmin | 24 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| time_ms | 2,031,274 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_click | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_like | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_follow | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_comment | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_forward | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_hate | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| long_view | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| play_time_ms | 248,083 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| duration_ms | 212,199 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| profile_stay_time | 2,020 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| comment_stay_time | 52,110 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_profile_enter | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_rand | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| tab | 15 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

<details>
<summary>user_id: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 879 | 127,028 | 1.09% |
| 793 | 101,916 | 0.87% |
| 909 | 82,036 | 0.70% |
| 725 | 76,840 | 0.66% |
| 71 | 72,347 | 0.62% |
| 75 | 68,119 | 0.58% |
| 205 | 66,295 | 0.57% |
| 718 | 65,748 | 0.56% |
| 238 | 63,813 | 0.55% |
| 613 | 63,609 | 0.55% |

</details>

<details>
<summary>video_id: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 3892658 | 431 | 0.00% |
| 160718 | 414 | 0.00% |
| 111390 | 408 | 0.00% |
| 26075 | 403 | 0.00% |
| 4066465 | 390 | 0.00% |
| 371172 | 387 | 0.00% |
| 1628483 | 366 | 0.00% |
| 6921 | 362 | 0.00% |
| 1758348 | 361 | 0.00% |
| 689882 | 361 | 0.00% |

</details>

<details>
<summary>date: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 20220501 | 439,865 | 3.77% |
| 20220503 | 436,477 | 3.74% |
| 20220430 | 435,855 | 3.74% |
| 20220502 | 425,622 | 3.65% |
| 20220423 | 419,178 | 3.59% |
| 20220508 | 415,127 | 3.56% |
| 20220504 | 409,568 | 3.51% |
| 20220410 | 402,656 | 3.45% |
| 20220506 | 398,141 | 3.41% |
| 20220409 | 397,734 | 3.41% |
| 20220417 | 394,489 | 3.38% |
| 20220429 | 391,103 | 3.35% |
| 20220507 | 389,900 | 3.34% |
| 20220416 | 386,228 | 3.31% |
| 20220505 | 385,267 | 3.30% |
| 20220411 | 368,100 | 3.16% |
| 20220422 | 365,950 | 3.14% |
| 20220415 | 359,155 | 3.08% |
| 20220408 | 355,732 | 3.05% |
| 20220426 | 349,204 | 2.99% |
| 20220424 | 347,899 | 2.98% |
| 20220427 | 347,667 | 2.98% |
| 20220413 | 344,189 | 2.95% |
| 20220428 | 343,165 | 2.94% |
| 20220412 | 343,069 | 2.94% |
| 20220421 | 342,527 | 2.94% |
| 20220425 | 340,438 | 2.92% |
| 20220420 | 336,447 | 2.89% |
| 20220419 | 335,532 | 2.88% |
| 20220418 | 331,629 | 2.84% |
| 20220414 | 323,935 | 2.78% |

</details>

<details>
<summary>hourmin: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 2100 | 791,160 | 6.78% |
| 2200 | 778,362 | 6.67% |
| 2000 | 775,125 | 6.65% |
| 1900 | 723,010 | 6.20% |
| 1800 | 692,974 | 5.94% |
| 1700 | 637,930 | 5.47% |
| 1200 | 618,339 | 5.30% |
| 1300 | 615,565 | 5.28% |
| 1600 | 608,138 | 5.21% |
| 2300 | 583,655 | 5.00% |
| 1500 | 582,403 | 4.99% |
| 1400 | 558,095 | 4.79% |
| 1100 | 550,732 | 4.72% |
| 1000 | 514,922 | 4.42% |
| 900 | 471,924 | 4.05% |
| 800 | 430,512 | 3.69% |
| 0 | 366,421 | 3.14% |
| 700 | 363,060 | 3.11% |
| 600 | 270,305 | 2.32% |
| 100 | 232,290 | 1.99% |
| 500 | 151,430 | 1.30% |
| 200 | 139,851 | 1.20% |
| 400 | 102,963 | 0.88% |
| 300 | 102,682 | 0.88% |

</details>

<details>
<summary>time_ms: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1650541726939 | 23 | 0.00% |
| 1651154663374 | 23 | 0.00% |
| 1651244350495 | 23 | 0.00% |
| 1651658250587 | 23 | 0.00% |
| 1649399512911 | 22 | 0.00% |
| 1649741821085 | 22 | 0.00% |
| 1649844207974 | 22 | 0.00% |
| 1650117796848 | 22 | 0.00% |
| 1651598996063 | 22 | 0.00% |
| 1651231173177 | 21 | 0.00% |

</details>

<details>
<summary>is_click: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 7,241,382 | 62.09% |
| 1 | 4,420,466 | 37.91% |

Finite known numeric values: min=0, median=0, p90=1, p99=1, max=1, mean=0.3791. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_like: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,479,378 | 98.44% |
| 1 | 182,470 | 1.56% |

Finite known numeric values: min=0, median=0, p90=0, p99=1, max=1, mean=0.01565. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_follow: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,650,536 | 99.90% |
| 1 | 11,312 | 0.10% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=1, mean=0.00097. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_comment: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,630,757 | 99.73% |
| 1 | 31,091 | 0.27% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=1, mean=0.002666. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_forward: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,652,655 | 99.92% |
| 1 | 9,193 | 0.08% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=1, mean=0.0007883. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_hate: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,650,982 | 99.91% |
| 1 | 10,866 | 0.09% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=1, mean=0.0009318. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>long_view: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 8,596,811 | 73.72% |
| 1 | 3,065,037 | 26.28% |

Finite known numeric values: min=0, median=0, p90=1, p99=1, max=1, mean=0.2628. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>play_time_ms: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 2,336,338 | 20.03% |
| 1342 | 2,359 | 0.02% |
| 1198 | 2,355 | 0.02% |
| 1248 | 2,344 | 0.02% |
| 1312 | 2,343 | 0.02% |
| 1341 | 2,326 | 0.02% |
| 1195 | 2,322 | 0.02% |
| 1294 | 2,316 | 0.02% |
| 1259 | 2,312 | 0.02% |
| 1378 | 2,310 | 0.02% |

Finite known numeric values: min=0, median=3032, p90=3.882e+04, p99=1.689e+05, max=1529347, mean=1.483e+04. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>duration_ms: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 936,226 | 8.03% |
| 11500 | 42,947 | 0.37% |
| 6000 | 31,154 | 0.27% |
| 11533 | 28,154 | 0.24% |
| 60000 | 27,915 | 0.24% |
| 7000 | 25,527 | 0.22% |
| 8000 | 24,026 | 0.21% |
| 9000 | 22,257 | 0.19% |
| 7100 | 17,676 | 0.15% |
| 7033 | 17,322 | 0.15% |

Finite known numeric values: min=0, median=2.907e+04, p90=1.947e+05, p99=4.797e+05, max=16513080, mean=7.227e+04. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>profile_stay_time: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,659,655 | 99.98% |
| 300000 | 76 | 0.00% |
| 1059 | 3 | 0.00% |
| 2527 | 3 | 0.00% |
| 2578 | 3 | 0.00% |
| 10309 | 2 | 0.00% |
| 11006 | 2 | 0.00% |
| 1129 | 2 | 0.00% |
| 1191 | 2 | 0.00% |
| 12085 | 2 | 0.00% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=300000, mean=6.853. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>comment_stay_time: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,095,748 | 95.15% |
| 300000 | 114 | 0.00% |
| 2417 | 102 | 0.00% |
| 2178 | 99 | 0.00% |
| 2285 | 97 | 0.00% |
| 2004 | 96 | 0.00% |
| 2115 | 96 | 0.00% |
| 2969 | 96 | 0.00% |
| 1885 | 95 | 0.00% |
| 2037 | 95 | 0.00% |

Finite known numeric values: min=0, median=0, p90=0, p99=1.352e+04, max=300000, mean=493.2. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_profile_enter: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,452,528 | 98.21% |
| 1 | 209,320 | 1.79% |

Finite known numeric values: min=0, median=0, p90=0, p99=1, max=1, mean=0.01795. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_rand: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 11,618,821 | 99.63% |
| 1 | 43,027 | 0.37% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=1, mean=0.00369. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>tab: value counts and distribution</summary>

Rows: 11,661,848; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 1 | 7,760,232 | 66.54% |
| 0 | 2,404,886 | 20.62% |
| 4 | 895,385 | 7.68% |
| 2 | 402,366 | 3.45% |
| 6 | 99,087 | 0.85% |
| 3 | 37,382 | 0.32% |
| 11 | 17,872 | 0.15% |
| 8 | 17,430 | 0.15% |
| 12 | 11,007 | 0.09% |
| 5 | 9,028 | 0.08% |
| 7 | 3,833 | 0.03% |
| 9 | 1,911 | 0.02% |
| 10 | 952 | 0.01% |
| 14 | 460 | 0.00% |
| 13 | 17 | 0.00% |

Finite known numeric values: min=0, median=1, p90=2, p99=6, max=14, mean=1.151. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

## User snapshot: availability unverified, excluded from strict baseline

| Field | Cardinality | Missing n (%) | Unknown n (%) | Invalid n (%) |
|---|---|---|---|---|
| user_id | 1,000 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| user_active_degree | 7 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_lowactive_period | 1 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| is_live_streamer | 2 | 0 (0.00%) | 782 (78.20%) | 0 (0.00%) |
| is_video_author | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| follow_user_num | 532 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| follow_user_num_range | 8 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| fans_user_num | 379 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| fans_user_num_range | 8 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| friend_user_num | 264 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| friend_user_num_range | 7 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| register_days | 798 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| register_days_range | 7 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat0 | 2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat1 | 7 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat2 | 23 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat3 | 394 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat4 | 13 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat5 | 5 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat6 | 3 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat7 | 37 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat8 | 283 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat9 | 7 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat10 | 5 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat11 | 3 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat12 | 2 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat13 | 2 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat14 | 2 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat15 | 2 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat16 | 2 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |
| onehot_feat17 | 2 | 33 (3.30%) | 0 (0.00%) | 0 (0.00%) |

<details>
<summary>user_id: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 1 | 0.10% |
| 1 | 1 | 0.10% |
| 10 | 1 | 0.10% |
| 100 | 1 | 0.10% |
| 101 | 1 | 0.10% |
| 102 | 1 | 0.10% |
| 103 | 1 | 0.10% |
| 104 | 1 | 0.10% |
| 105 | 1 | 0.10% |
| 106 | 1 | 0.10% |

</details>

<details>
<summary>user_active_degree: value counts and distribution</summary>

Rows: 1,000; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| full_active | 640 | 64.00% |
| high_active | 233 | 23.30% |
| middle_active | 83 | 8.30% |
| 2_14_day_new | 23 | 2.30% |
| low_active | 17 | 1.70% |
| 30day_retention | 2 | 0.20% |
| single_low_active | 2 | 0.20% |

</details>

<details>
<summary>is_lowactive_period: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 1,000 | 100.00% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=0, mean=0. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_live_streamer: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| -124 | 782 | 78.20% |
| 1 | 218 | 21.80% |

Finite known numeric values: min=1, median=1, p90=1, p99=1, max=1, mean=1. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>is_video_author: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: known binary outside {0,1}. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 1 | 808 | 80.80% |
| 0 | 192 | 19.20% |

Finite known numeric values: min=0, median=1, p90=1, p99=1, max=1, mean=0.808. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>follow_user_num: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 5 | 11 | 1.10% |
| 14 | 10 | 1.00% |
| 0 | 9 | 0.90% |
| 28 | 9 | 0.90% |
| 11 | 8 | 0.80% |
| 12 | 8 | 0.80% |
| 17 | 8 | 0.80% |
| 24 | 8 | 0.80% |
| 3 | 8 | 0.80% |
| 8 | 8 | 0.80% |

Finite known numeric values: min=0, median=160.5, p90=1000, p99=3521, max=5000, mean=395.4. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>follow_user_num_range: value counts and distribution</summary>

Rows: 1,000; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 500+ | 209 | 20.90% |
| (10,50] | 194 | 19.40% |
| (250,500] | 175 | 17.50% |
| (50,100] | 134 | 13.40% |
| (150,250] | 129 | 12.90% |
| (100,150] | 88 | 8.80% |
| (0,10] | 62 | 6.20% |
| 0 | 9 | 0.90% |

</details>

<details>
<summary>fans_user_num: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 46 | 4.60% |
| 1 | 42 | 4.20% |
| 6 | 26 | 2.60% |
| 2 | 25 | 2.50% |
| 8 | 25 | 2.50% |
| 4 | 24 | 2.40% |
| 9 | 24 | 2.40% |
| 5 | 23 | 2.30% |
| 3 | 22 | 2.20% |
| 10 | 18 | 1.80% |

Finite known numeric values: min=0, median=43, p90=642.7, p99=7516, max=156158, mean=718.1. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>fans_user_num_range: value counts and distribution</summary>

Rows: 1,000; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| [10,100) | 372 | 37.20% |
| [100,1k) | 297 | 29.70% |
| [1,10) | 223 | 22.30% |
| 0 | 46 | 4.60% |
| [1k,5k) | 45 | 4.50% |
| [1w,10w) | 8 | 0.80% |
| [5k,1w) | 8 | 0.80% |
| [10w,100w) | 1 | 0.10% |

</details>

<details>
<summary>friend_user_num: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 129 | 12.90% |
| 1 | 81 | 8.10% |
| 2 | 47 | 4.70% |
| 4 | 44 | 4.40% |
| 3 | 38 | 3.80% |
| 5 | 29 | 2.90% |
| 6 | 21 | 2.10% |
| 7 | 21 | 2.10% |
| 13 | 18 | 1.80% |
| 9 | 18 | 1.80% |

Finite known numeric values: min=0, median=14, p90=235.2, p99=1536, max=4690, mean=111.4. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>friend_user_num_range: value counts and distribution</summary>

Rows: 1,000; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| [5,30) | 278 | 27.80% |
| [1,5) | 210 | 21.00% |
| 0 | 129 | 12.90% |
| [30,60) | 100 | 10.00% |
| [60,120) | 98 | 9.80% |
| 250+ | 93 | 9.30% |
| [120,250) | 92 | 9.20% |

</details>

<details>
<summary>register_days: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 835 | 6 | 0.60% |
| 451 | 4 | 0.40% |
| 73 | 4 | 0.40% |
| 1156 | 3 | 0.30% |
| 1190 | 3 | 0.30% |
| 1267 | 3 | 0.30% |
| 1302 | 3 | 0.30% |
| 1396 | 3 | 0.30% |
| 1493 | 3 | 0.30% |
| 1568 | 3 | 0.30% |

Finite known numeric values: min=22, median=1234, p90=2149, p99=2700, max=3187, mean=1231. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>register_days_range: value counts and distribution</summary>

Rows: 1,000; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 730+ | 741 | 74.10% |
| 366-730 | 115 | 11.50% |
| 181-365 | 70 | 7.00% |
| 91-180 | 39 | 3.90% |
| 31-60 | 16 | 1.60% |
| 61-90 | 15 | 1.50% |
| 15-30 | 4 | 0.40% |

</details>

<details>
<summary>onehot_feat0: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 1 | 570 | 57.00% |
| 0 | 430 | 43.00% |

Finite known numeric values: min=0, median=1, p90=1, p99=1, max=1, mean=0.57. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat1: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 4 | 267 | 26.70% |
| 1 | 202 | 20.20% |
| 2 | 191 | 19.10% |
| 3 | 135 | 13.50% |
| 5 | 98 | 9.80% |
| 6 | 69 | 6.90% |
| 0 | 38 | 3.80% |

Finite known numeric values: min=0, median=3, p90=5, p99=6, max=6, mean=2.961. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat2: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 2 | 265 | 26.50% |
| 19 | 202 | 20.20% |
| 10 | 197 | 19.70% |
| 29 | 192 | 19.20% |
| 31 | 44 | 4.40% |
| 28 | 29 | 2.90% |
| 9 | 26 | 2.60% |
| 48 | 14 | 1.40% |
| 33 | 8 | 0.80% |
| 18 | 3 | 0.30% |
| 20 | 3 | 0.30% |
| 0 | 2 | 0.20% |
| 13 | 2 | 0.20% |
| 22 | 2 | 0.20% |
| 24 | 2 | 0.20% |
| 34 | 2 | 0.20% |
| 11 | 1 | 0.10% |
| 15 | 1 | 0.10% |
| 16 | 1 | 0.10% |
| 32 | 1 | 0.10% |
| 35 | 1 | 0.10% |
| 38 | 1 | 0.10% |
| 49 | 1 | 0.10% |

Finite known numeric values: min=0, median=19, p90=29, p99=48, max=49, mean=15.75. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat3: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1160 | 38 | 3.80% |
| 1159 | 29 | 2.90% |
| 1153 | 28 | 2.80% |
| 1165 | 23 | 2.30% |
| 1152 | 18 | 1.80% |
| 1171 | 18 | 1.80% |
| 620 | 13 | 1.30% |
| 567 | 12 | 1.20% |
| 906 | 12 | 1.20% |
| 1180 | 11 | 1.10% |

Finite known numeric values: min=8, median=889, p90=1171, p99=1372, max=1460, mean=786. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat4: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 369 | 36.90% |
| 2.0 | 181 | 18.10% |
| 3.0 | 118 | 11.80% |
| 0.0 | 109 | 10.90% |
| 4.0 | 78 | 7.80% |
| 5.0 | 43 | 4.30% |
| (missing) | 33 | 3.30% |
| 6.0 | 32 | 3.20% |
| 7.0 | 12 | 1.20% |
| 8.0 | 11 | 1.10% |
| 9.0 | 10 | 1.00% |
| 12.0 | 2 | 0.20% |
| 11.0 | 1 | 0.10% |
| 13.0 | 1 | 0.10% |

Finite known numeric values: min=0.0, median=2, p90=5, p99=9, max=13.0, mean=2.186. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat5: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 995 | 99.50% |
| 2 | 2 | 0.20% |
| 27 | 1 | 0.10% |
| 3 | 1 | 0.10% |
| 5 | 1 | 0.10% |

Finite known numeric values: min=0, median=0, p90=0, p99=0, max=27, mean=0.039. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat6: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 564 | 56.40% |
| 1 | 431 | 43.10% |
| 2 | 5 | 0.50% |

Finite known numeric values: min=0, median=0, p90=1, p99=1, max=2, mean=0.441. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat7: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 31 | 97 | 9.70% |
| 13 | 84 | 8.40% |
| 8 | 74 | 7.40% |
| 15 | 61 | 6.10% |
| 20 | 52 | 5.20% |
| 24 | 52 | 5.20% |
| 43 | 48 | 4.80% |
| 7 | 45 | 4.50% |
| 44 | 39 | 3.90% |
| 14 | 30 | 3.00% |
| 22 | 30 | 3.00% |
| 34 | 29 | 2.90% |
| 10 | 27 | 2.70% |
| 30 | 27 | 2.70% |
| 9 | 26 | 2.60% |
| 18 | 25 | 2.50% |
| 33 | 25 | 2.50% |
| 0 | 23 | 2.30% |
| 19 | 23 | 2.30% |
| 1 | 22 | 2.20% |
| 21 | 21 | 2.10% |
| 42 | 21 | 2.10% |
| 5 | 19 | 1.90% |
| 28 | 18 | 1.80% |
| 2 | 15 | 1.50% |
| 26 | 15 | 1.50% |
| 38 | 14 | 1.40% |
| 32 | 13 | 1.30% |
| 12 | 9 | 0.90% |
| 17 | 9 | 0.90% |
| 103 | 1 | 0.10% |
| 114 | 1 | 0.10% |
| 16 | 1 | 0.10% |
| 25 | 1 | 0.10% |
| 54 | 1 | 0.10% |
| 73 | 1 | 0.10% |
| 85 | 1 | 0.10% |

Finite known numeric values: min=0, median=20, p90=42, p99=44, max=114, mean=21.12. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat8: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 82 | 22 | 2.20% |
| 330 | 17 | 1.70% |
| 191 | 16 | 1.60% |
| 399 | 15 | 1.50% |
| 412 | 15 | 1.50% |
| 139 | 14 | 1.40% |
| 42 | 14 | 1.40% |
| 116 | 13 | 1.30% |
| 264 | 13 | 1.30% |
| 33 | 13 | 1.30% |

Finite known numeric values: min=8, median=248, p90=399, p99=450, max=453, mean=235.8. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat9: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 1 | 227 | 22.70% |
| 6 | 217 | 21.70% |
| 3 | 210 | 21.00% |
| 5 | 163 | 16.30% |
| 4 | 121 | 12.10% |
| 2 | 53 | 5.30% |
| 0 | 9 | 0.90% |

Finite known numeric values: min=0, median=4, p90=6, p99=6, max=6, mean=3.564. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat10: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 3 | 387 | 38.70% |
| 2 | 289 | 28.90% |
| 0 | 167 | 16.70% |
| 4 | 156 | 15.60% |
| 1 | 1 | 0.10% |

Finite known numeric values: min=0, median=3, p90=4, p99=4, max=4, mean=2.364. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat11: value counts and distribution</summary>

Rows: 1,000; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 734 | 73.40% |
| 2 | 255 | 25.50% |
| 1 | 11 | 1.10% |

Finite known numeric values: min=0, median=0, p90=2, p99=2, max=2, mean=0.521. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat12: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 701 | 70.10% |
| 0.0 | 266 | 26.60% |
| (missing) | 33 | 3.30% |

Finite known numeric values: min=0.0, median=1, p90=1, p99=1, max=1.0, mean=0.7249. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat13: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 907 | 90.70% |
| 1.0 | 60 | 6.00% |
| (missing) | 33 | 3.30% |

Finite known numeric values: min=0.0, median=0, p90=0, p99=1, max=1.0, mean=0.06205. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat14: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 854 | 85.40% |
| 1.0 | 113 | 11.30% |
| (missing) | 33 | 3.30% |

Finite known numeric values: min=0.0, median=0, p90=1, p99=1, max=1.0, mean=0.1169. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat15: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 939 | 93.90% |
| (missing) | 33 | 3.30% |
| 1.0 | 28 | 2.80% |

Finite known numeric values: min=0.0, median=0, p90=0, p99=1, max=1.0, mean=0.02896. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat16: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 929 | 92.90% |
| 1.0 | 38 | 3.80% |
| (missing) | 33 | 3.30% |

Finite known numeric values: min=0.0, median=0, p90=0, p99=1, max=1.0, mean=0.0393. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>onehot_feat17: value counts and distribution</summary>

Rows: 1,000; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 958 | 95.80% |
| (missing) | 33 | 3.30% |
| 1.0 | 9 | 0.90% |

Finite known numeric values: min=0.0, median=0, p90=0, p99=0, max=1.0, mean=0.009307. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

## Video snapshot: availability unverified, excluded from strict baseline

| Field | Cardinality | Missing n (%) | Unknown n (%) | Invalid n (%) |
|---|---|---|---|---|
| video_id | 4,371,868 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| author_id | 1,407,475 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| video_type | 3 | 0 (0.00%) | 245 (0.01%) | 0 (0.00%) |
| upload_dt | 235 | 58 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| upload_type | 32 | 0 (0.00%) | 16,698 (0.38%) | 0 (0.00%) |
| visible_status | 3 | 5 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| video_duration | 212,211 | 573,057 (13.11%) | 0 (0.00%) | 0 (0.00%) |
| server_width | 3,109 | 7 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| server_height | 3,997 | 7 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| music_id | 2,622,610 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| music_type | 6 | 59,952 (1.37%) | 0 (0.00%) | 0 (0.00%) |
| tag | 1,188 | 137,484 (3.14%) | 0 (0.00%) | 0 (0.00%) |

<details>
<summary>video_id: value counts and distribution</summary>

Rows: 4,371,868; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 1 | 0.00% |
| 1 | 1 | 0.00% |
| 10 | 1 | 0.00% |
| 100 | 1 | 0.00% |
| 1000 | 1 | 0.00% |
| 10000 | 1 | 0.00% |
| 100000 | 1 | 0.00% |
| 1000000 | 1 | 0.00% |
| 1000001 | 1 | 0.00% |
| 1000002 | 1 | 0.00% |

</details>

<details>
<summary>author_id: value counts and distribution</summary>

Rows: 4,371,868; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 3967151 | 975 | 0.02% |
| 5294188 | 857 | 0.02% |
| 5921989 | 786 | 0.02% |
| 7136643 | 749 | 0.02% |
| 6710390 | 712 | 0.02% |
| 6004355 | 684 | 0.02% |
| 5281979 | 586 | 0.01% |
| 5683007 | 563 | 0.01% |
| 5308250 | 547 | 0.01% |
| 5195566 | 545 | 0.01% |

</details>

<details>
<summary>video_type: value counts and distribution</summary>

Rows: 4,371,868; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| NORMAL | 4,361,638 | 99.77% |
| AD | 9,985 | 0.23% |
| UNKNOWN | 245 | 0.01% |

</details>

<details>
<summary>upload_dt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DATE. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 2022-05-01 | 116,364 | 2.66% |
| 2022-05-02 | 114,899 | 2.63% |
| 2022-05-03 | 114,710 | 2.62% |
| 2022-04-29 | 113,696 | 2.60% |
| 2022-04-30 | 113,622 | 2.60% |
| 2022-04-09 | 113,172 | 2.59% |
| 2022-04-08 | 109,701 | 2.51% |
| 2022-04-22 | 109,528 | 2.51% |
| 2022-04-16 | 108,373 | 2.48% |
| 2022-05-04 | 107,509 | 2.46% |

</details>

<details>
<summary>upload_type: value counts and distribution</summary>

Rows: 4,371,868; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| LongImport | 1,108,258 | 25.35% |
| ShortImport | 872,196 | 19.95% |
| Kmovie | 777,186 | 17.78% |
| Web | 428,111 | 9.79% |
| LongPicture | 323,304 | 7.40% |
| LongCamera | 263,384 | 6.02% |
| ShortCamera | 250,519 | 5.73% |
| PictureCopy | 128,750 | 2.94% |
| PictureSet | 90,383 | 2.07% |
| PhotoCopy | 25,953 | 0.59% |
| FollowShoot | 20,857 | 0.48% |
| UNKNOWN | 16,698 | 0.38% |
| Recreation | 16,163 | 0.37% |
| AiCutVideo | 14,020 | 0.32% |
| ShareFromOtherApp | 12,227 | 0.28% |
| FlashPhoto | 7,029 | 0.16% |
| SameFrame | 5,881 | 0.13% |
| OriginPicture | 5,345 | 0.12% |
| LipsSync | 1,673 | 0.04% |
| LocalIntelligenceAlbum | 1,639 | 0.04% |
| LiveClip | 1,289 | 0.03% |
| Karaoke | 358 | 0.01% |
| Status | 270 | 0.01% |
| StoryMoodTemplate | 248 | 0.01% |
| LocalCollection | 78 | 0.00% |
| ALBUM2021 | 21 | 0.00% |
| Solitaire | 15 | 0.00% |
| CommonAvatar | 4 | 0.00% |
| Copy | 3 | 0.00% |
| ShortOriginImport | 3 | 0.00% |
| ResidentChange | 2 | 0.00% |
| ShootRecognition | 1 | 0.00% |

</details>

<details>
<summary>visible_status: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 4,347,473 | 99.44% |
| 1.0 | 24,207 | 0.55% |
| 3.0 | 183 | 0.00% |
| (missing) | 5 | 0.00% |

Finite known numeric values: min=0.0, median=0, p90=0, p99=0, max=3.0, mean=0.005663. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>video_duration: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| (missing) | 573,057 | 13.11% |
| 11500.0 | 25,609 | 0.59% |
| 60000.0 | 21,235 | 0.49% |
| 11533.0 | 14,247 | 0.33% |
| 6000.0 | 9,913 | 0.23% |
| 7000.0 | 9,695 | 0.22% |
| 8000.0 | 7,946 | 0.18% |
| 11511.0 | 7,233 | 0.17% |
| 7033.0 | 6,280 | 0.14% |
| 9000.0 | 6,198 | 0.14% |

Finite known numeric values: min=40.0, median=3.51e+04, p90=1.93e+05, p99=5.08e+05, max=16513080.0, mean=7.571e+04. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>server_width: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 720.0 | 3,197,445 | 73.14% |
| 1280.0 | 274,252 | 6.27% |
| 1080.0 | 158,281 | 3.62% |
| 960.0 | 79,407 | 1.82% |
| 540.0 | 64,935 | 1.49% |
| 1440.0 | 50,205 | 1.15% |
| 810.0 | 38,031 | 0.87% |
| 576.0 | 36,138 | 0.83% |
| 686.0 | 14,921 | 0.34% |
| 3024.0 | 9,981 | 0.23% |

Finite known numeric values: min=48.0, median=720, p90=1162, p99=1701, max=8394.0, mean=813.3. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>server_height: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1280.0 | 2,873,174 | 65.72% |
| 720.0 | 456,393 | 10.44% |
| 1440.0 | 272,123 | 6.22% |
| 960.0 | 150,942 | 3.45% |
| 1282.0 | 80,193 | 1.83% |
| 1920.0 | 45,669 | 1.04% |
| 1600.0 | 36,520 | 0.84% |
| 1080.0 | 27,541 | 0.63% |
| 1024.0 | 24,071 | 0.55% |
| 1556.0 | 20,316 | 0.46% |

Finite known numeric values: min=33.0, median=1280, p90=1440, p99=2208, max=11826.0, mean=1235. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>music_id: value counts and distribution</summary>

Rows: 4,371,868; dtype: BIGINT. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 60,123 | 1.38% |
| 6201083918 | 6,850 | 0.16% |
| 9107419068 | 4,963 | 0.11% |
| 7097343593 | 4,790 | 0.11% |
| 12279391 | 3,623 | 0.08% |
| 7700840082 | 2,796 | 0.06% |
| 6859762599 | 2,616 | 0.06% |
| 9321912835 | 2,588 | 0.06% |
| 7655044588 | 2,428 | 0.06% |
| 8775845872 | 2,413 | 0.06% |

</details>

<details>
<summary>music_type: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: no domain rule asserted. Frequency scope: all values.

| Value | Count | Share of rows |
|---|---|---|
| 9.0 | 3,166,538 | 72.43% |
| 4.0 | 943,659 | 21.58% |
| 8.0 | 116,075 | 2.66% |
| 7.0 | 82,808 | 1.89% |
| (missing) | 59,952 | 1.37% |
| 11.0 | 2,832 | 0.06% |
| 1.0 | 4 | 0.00% |

Finite known numeric values: min=1.0, median=9, p90=9, p99=9, max=11.0, mean=7.842. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>tag: value counts and distribution</summary>

Rows: 4,371,868; dtype: VARCHAR. Invalid rule: no domain rule asserted. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 34 | 459,481 | 10.51% |
| 3 | 216,336 | 4.95% |
| 9 | 184,808 | 4.23% |
| 2 | 178,404 | 4.08% |
| 12 | 175,502 | 4.01% |
| 11 | 174,805 | 4.00% |
| 39 | 170,592 | 3.90% |
| 28 | 149,176 | 3.41% |
| 39,68 | 148,060 | 3.39% |
| (missing) | 137,484 | 3.14% |

</details>

## Monthly video aggregates: excluded for temporal leakage

| Field | Cardinality | Missing n (%) | Unknown n (%) | Invalid n (%) |
|---|---|---|---|---|
| video_id | 4,371,868 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| counts | 189 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| show_cnt | 3,175,236 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| show_user_num | 3,119,637 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| play_cnt | 2,872,004 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| play_user_num | 2,825,383 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| play_duration | 4,369,949 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| complete_play_cnt | 2,011,395 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| complete_play_user_num | 1,995,880 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| valid_play_cnt | 2,499,223 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| valid_play_user_num | 2,479,305 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| long_time_play_cnt | 2,323,404 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| long_time_play_user_num | 2,307,101 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| short_time_play_cnt | 2,282,998 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| short_time_play_user_num | 2,241,422 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| play_progress | 4,370,275 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| comment_stay_duration | 4,307,430 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| like_cnt | 1,004,521 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| like_user_num | 998,361 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| click_like_cnt | 760,892 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| double_click_cnt | 776,474 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| cancel_like_cnt | 354,819 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| cancel_like_user_num | 331,940 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| comment_cnt | 320,636 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| comment_user_num | 297,554 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| direct_comment_cnt | 275,307 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| reply_comment_cnt | 165,801 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| delete_comment_cnt | 71,885 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| delete_comment_user_num | 63,992 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| comment_like_cnt | 432,621 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| comment_like_user_num | 331,265 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| follow_cnt | 357,217 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| follow_user_num | 355,553 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| cancel_follow_cnt | 86,474 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| cancel_follow_user_num | 85,895 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| share_cnt | 189,399 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| share_user_num | 179,017 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| download_cnt | 161,747 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| download_user_num | 155,614 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| report_cnt | 8,614 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| report_user_num | 8,153 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| reduce_similar_cnt | 199,011 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| reduce_similar_user_num | 191,616 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| collect_cnt | 437,489 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| collect_user_num | 428,964 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| cancel_collect_cnt | 221,299 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| cancel_collect_user_num | 213,397 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| direct_comment_user_num | 266,986 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| reply_comment_user_num | 144,321 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| share_all_cnt | 244,097 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| share_all_user_num | 227,160 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| outsite_share_all_cnt | 192,578 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

<details>
<summary>video_id: value counts and distribution</summary>

Rows: 4,371,868; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0 | 1 | 0.00% |
| 1 | 1 | 0.00% |
| 10 | 1 | 0.00% |
| 100 | 1 | 0.00% |
| 1000 | 1 | 0.00% |
| 10000 | 1 | 0.00% |
| 100000 | 1 | 0.00% |
| 1000000 | 1 | 0.00% |
| 1000001 | 1 | 0.00% |
| 1000002 | 1 | 0.00% |

</details>

<details>
<summary>counts: value counts and distribution</summary>

Rows: 4,371,868; dtype: BIGINT. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 12 | 52,827 | 1.21% |
| 18 | 52,444 | 1.20% |
| 24 | 49,440 | 1.13% |
| 16 | 46,998 | 1.08% |
| 30 | 46,992 | 1.07% |
| 17 | 46,341 | 1.06% |
| 22 | 45,772 | 1.05% |
| 20 | 45,266 | 1.04% |
| 23 | 45,180 | 1.03% |
| 21 | 45,085 | 1.03% |

Finite known numeric values: min=1, median=58, p90=135, p99=185, max=189, mean=66.23. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>show_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 25.0 | 217 | 0.00% |
| 27.0 | 217 | 0.00% |
| 8.0 | 217 | 0.00% |
| 16.0 | 216 | 0.00% |
| 30.0 | 215 | 0.00% |
| 11.0 | 214 | 0.00% |
| 19.0 | 211 | 0.00% |
| 20.0 | 211 | 0.00% |
| 10.0 | 208 | 0.00% |
| 13.0 | 208 | 0.00% |

Finite known numeric values: min=0.4, median=1556, p90=1.262e+04, p99=7.435e+04, max=5446968.833333333, mean=6090. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>show_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 17.0 | 308 | 0.01% |
| 12.0 | 304 | 0.01% |
| 20.0 | 301 | 0.01% |
| 11.0 | 294 | 0.01% |
| 8.0 | 294 | 0.01% |
| 18.0 | 293 | 0.01% |
| 16.0 | 292 | 0.01% |
| 14.0 | 288 | 0.01% |
| 21.0 | 285 | 0.01% |
| 30.0 | 283 | 0.01% |

Finite known numeric values: min=0.4, median=1384, p90=1.127e+04, p99=6.687e+04, max=5358021.666666667, mean=5460. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>play_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 7.0 | 568 | 0.01% |
| 14.0 | 565 | 0.01% |
| 5.0 | 563 | 0.01% |
| 6.0 | 558 | 0.01% |
| 12.0 | 554 | 0.01% |
| 8.0 | 554 | 0.01% |
| 11.0 | 549 | 0.01% |
| 4.0 | 540 | 0.01% |
| 9.0 | 534 | 0.01% |
| 2.0 | 530 | 0.01% |

Finite known numeric values: min=0.0, median=798.3, p90=9245, p99=6.43e+04, max=5360840.833333333, mean=4674. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>play_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 4.0 | 736 | 0.02% |
| 7.0 | 726 | 0.02% |
| 2.0 | 715 | 0.02% |
| 6.0 | 713 | 0.02% |
| 5.0 | 701 | 0.02% |
| 8.0 | 695 | 0.02% |
| 9.0 | 685 | 0.02% |
| 11.0 | 683 | 0.02% |
| 10.0 | 678 | 0.02% |
| 3.0 | 669 | 0.02% |

Finite known numeric values: min=0.0, median=722, p90=8365, p99=5.85e+04, max=4955046.166666667, mean=4239. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>play_duration: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 7 | 0.00% |
| 125487.0 | 4 | 0.00% |
| 1109131.0 | 3 | 0.00% |
| 116799.0 | 3 | 0.00% |
| 150035.66666666666 | 3 | 0.00% |
| 213213.0 | 3 | 0.00% |
| 3037537.0 | 3 | 0.00% |
| 314434.5 | 3 | 0.00% |
| 321352.0 | 3 | 0.00% |
| 33773.5 | 3 | 0.00% |

Finite known numeric values: min=0.0, median=1.372e+07, p90=2.421e+08, p99=2.342e+09, max=230992904724.6111, mean=1.465e+08. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>complete_play_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 7,328 | 0.17% |
| 0.0 | 6,842 | 0.16% |
| 0.5 | 5,838 | 0.13% |
| 2.0 | 4,918 | 0.11% |
| 3.0 | 3,802 | 0.09% |
| 0.3333333333333333 | 3,534 | 0.08% |
| 1.5 | 3,300 | 0.08% |
| 0.25 | 3,230 | 0.07% |
| 4.0 | 3,127 | 0.07% |
| 0.6666666666666666 | 3,009 | 0.07% |

Finite known numeric values: min=0.0, median=116.2, p90=2294, p99=2.232e+04, max=3468119.5, mean=1400. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>complete_play_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 8,052 | 0.18% |
| 0.0 | 6,842 | 0.16% |
| 0.5 | 6,658 | 0.15% |
| 2.0 | 5,018 | 0.11% |
| 0.3333333333333333 | 4,074 | 0.09% |
| 3.0 | 4,001 | 0.09% |
| 0.25 | 3,737 | 0.09% |
| 1.5 | 3,442 | 0.08% |
| 0.6666666666666666 | 3,257 | 0.07% |
| 4.0 | 3,059 | 0.07% |

Finite known numeric values: min=0.0, median=111.9, p90=2214, p99=2.168e+04, max=3389049.5, mean=1357. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>valid_play_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 2.0 | 1,739 | 0.04% |
| 1.0 | 1,721 | 0.04% |
| 3.0 | 1,679 | 0.04% |
| 4.0 | 1,593 | 0.04% |
| 5.0 | 1,537 | 0.04% |
| 7.0 | 1,441 | 0.03% |
| 6.0 | 1,427 | 0.03% |
| 8.0 | 1,282 | 0.03% |
| 9.0 | 1,206 | 0.03% |
| 10.0 | 1,114 | 0.03% |

Finite known numeric values: min=0.0, median=370.8, p90=5097, p99=3.926e+04, max=3537738.833333333, mean=2692. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>valid_play_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 2,218 | 0.05% |
| 2.0 | 2,201 | 0.05% |
| 3.0 | 2,079 | 0.05% |
| 4.0 | 1,930 | 0.04% |
| 5.0 | 1,659 | 0.04% |
| 6.0 | 1,566 | 0.04% |
| 7.0 | 1,448 | 0.03% |
| 8.0 | 1,303 | 0.03% |
| 1.5 | 1,284 | 0.03% |
| 9.0 | 1,218 | 0.03% |

Finite known numeric values: min=0.0, median=355.4, p90=4868, p99=3.766e+04, max=3454915.6666666665, mean=2578. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>long_time_play_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 4,001 | 0.09% |
| 2.0 | 3,572 | 0.08% |
| 0.0 | 3,397 | 0.08% |
| 3.0 | 2,988 | 0.07% |
| 4.0 | 2,509 | 0.06% |
| 0.5 | 2,278 | 0.05% |
| 1.5 | 2,182 | 0.05% |
| 5.0 | 2,139 | 0.05% |
| 6.0 | 1,877 | 0.04% |
| 2.5 | 1,857 | 0.04% |

Finite known numeric values: min=0.0, median=250.2, p90=3963, p99=3.239e+04, max=3459012.833333333, mean=2152. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>long_time_play_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 5,255 | 0.12% |
| 2.0 | 3,941 | 0.09% |
| 0.0 | 3,397 | 0.08% |
| 3.0 | 3,048 | 0.07% |
| 0.5 | 3,041 | 0.07% |
| 1.5 | 2,567 | 0.06% |
| 4.0 | 2,486 | 0.06% |
| 5.0 | 2,095 | 0.05% |
| 2.5 | 1,985 | 0.05% |
| 6.0 | 1,778 | 0.04% |

Finite known numeric values: min=0.0, median=241.2, p90=3809, p99=3.128e+04, max=3381990.9444444445, mean=2074. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>short_time_play_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,695 | 0.04% |
| 2.0 | 1,447 | 0.03% |
| 4.0 | 1,336 | 0.03% |
| 3.0 | 1,329 | 0.03% |
| 5.0 | 1,312 | 0.03% |
| 6.0 | 1,296 | 0.03% |
| 1.0 | 1,268 | 0.03% |
| 7.0 | 1,250 | 0.03% |
| 8.0 | 1,210 | 0.03% |
| 9.0 | 1,142 | 0.03% |

Finite known numeric values: min=0.0, median=252.4, p90=2646, p99=1.711e+04, max=2214924.833333333, mean=1302. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>short_time_play_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,695 | 0.04% |
| 2.0 | 1,569 | 0.04% |
| 3.0 | 1,514 | 0.03% |
| 4.0 | 1,489 | 0.03% |
| 1.0 | 1,475 | 0.03% |
| 5.0 | 1,466 | 0.03% |
| 6.0 | 1,405 | 0.03% |
| 7.0 | 1,347 | 0.03% |
| 8.0 | 1,296 | 0.03% |
| 9.0 | 1,271 | 0.03% |

Finite known numeric values: min=0.0, median=232.6, p90=2439, p99=1.581e+04, max=2050803.0, mean=1201. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>play_progress: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,592 | 0.04% |
| 0.5 | 3 | 0.00% |
| 0.00013713295123420147 | 1 | 0.00% |
| 0.0001925925925925926 | 1 | 0.00% |
| 0.00020164534738008707 | 1 | 0.00% |
| 0.00020722784999178023 | 1 | 0.00% |
| 0.0002780592526809694 | 1 | 0.00% |
| 0.0002884251652370874 | 1 | 0.00% |
| 0.0003421937938520327 | 1 | 0.00% |
| 0.0004899540957721431 | 1 | 0.00% |

Finite known numeric values: min=0.0, median=0.1387, p90=0.3465, p99=0.544, max=0.9045627702211512, mean=0.168. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>comment_stay_duration: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 12,308 | 0.28% |
| 1078.0 | 8 | 0.00% |
| 159.0 | 8 | 0.00% |
| 317.0 | 8 | 0.00% |
| 603.0 | 8 | 0.00% |
| 737.0 | 8 | 0.00% |
| 10920.0 | 7 | 0.00% |
| 126.5 | 7 | 0.00% |
| 1846.0 | 7 | 0.00% |
| 2127.0 | 7 | 0.00% |

Finite known numeric values: min=0.0, median=2.795e+05, p90=6.255e+06, p99=6.683e+07, max=10639544829.166666, mean=4.072e+06. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>like_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 11,487 | 0.26% |
| 2.0 | 9,091 | 0.21% |
| 3.0 | 7,594 | 0.17% |
| 0.5 | 6,841 | 0.16% |
| 0.0 | 6,550 | 0.15% |
| 4.0 | 6,518 | 0.15% |
| 5.0 | 5,557 | 0.13% |
| 1.5 | 5,453 | 0.12% |
| 6.0 | 4,844 | 0.11% |
| 2.5 | 4,469 | 0.10% |

Finite known numeric values: min=0.0, median=13.65, p90=190.7, p99=1679, max=218748.63636363635, mean=109.9. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>like_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 11,631 | 0.27% |
| 2.0 | 9,213 | 0.21% |
| 3.0 | 7,676 | 0.18% |
| 0.5 | 6,852 | 0.16% |
| 0.0 | 6,550 | 0.15% |
| 4.0 | 6,514 | 0.15% |
| 5.0 | 5,750 | 0.13% |
| 1.5 | 5,560 | 0.13% |
| 6.0 | 4,897 | 0.11% |
| 2.5 | 4,463 | 0.10% |

Finite known numeric values: min=0.0, median=13.41, p90=187, p99=1652, max=152098.83333333334, mean=107.9. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>click_like_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 1.0 | 20,869 | 0.48% |
| 0.0 | 16,504 | 0.38% |
| 2.0 | 13,891 | 0.32% |
| 0.5 | 13,855 | 0.32% |
| 3.0 | 10,156 | 0.23% |
| 0.3333333333333333 | 8,927 | 0.20% |
| 1.5 | 8,842 | 0.20% |
| 0.6666666666666666 | 7,903 | 0.18% |
| 4.0 | 7,597 | 0.17% |
| 0.25 | 7,410 | 0.17% |

Finite known numeric values: min=0.0, median=6.012, p90=89.39, p99=761, max=215355.93181818185, mean=50.38. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>double_click_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 28,778 | 0.66% |
| 1.0 | 18,478 | 0.42% |
| 0.5 | 12,894 | 0.29% |
| 2.0 | 12,253 | 0.28% |
| 3.0 | 9,284 | 0.21% |
| 0.3333333333333333 | 8,856 | 0.20% |
| 1.5 | 7,908 | 0.18% |
| 0.25 | 7,785 | 0.18% |
| 4.0 | 7,292 | 0.17% |
| 0.6666666666666666 | 7,261 | 0.17% |

Finite known numeric values: min=0.0, median=6.44, p90=96, p99=938.1, max=89178.5, mean=58.88. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>cancel_like_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 210,178 | 4.81% |
| 0.5 | 27,082 | 0.62% |
| 1.0 | 26,420 | 0.60% |
| 0.3333333333333333 | 25,147 | 0.58% |
| 0.25 | 22,392 | 0.51% |
| 0.2 | 19,773 | 0.45% |
| 0.1666666666666666 | 18,716 | 0.43% |
| 0.1428571428571428 | 15,894 | 0.36% |
| 0.125 | 15,025 | 0.34% |
| 0.6666666666666666 | 14,178 | 0.32% |

Finite known numeric values: min=0.0, median=1.152, p90=16.02, p99=132.3, max=216622.9772727273, mean=9.146. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>cancel_like_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 210,178 | 4.81% |
| 0.5 | 27,679 | 0.63% |
| 1.0 | 26,792 | 0.61% |
| 0.3333333333333333 | 26,473 | 0.61% |
| 0.25 | 23,669 | 0.54% |
| 0.2 | 20,817 | 0.48% |
| 0.1666666666666666 | 19,857 | 0.45% |
| 0.1428571428571428 | 16,912 | 0.39% |
| 0.125 | 15,759 | 0.36% |
| 0.6666666666666666 | 14,599 | 0.33% |

Finite known numeric values: min=0.0, median=1.028, p90=14.08, p99=115.6, max=17654.735294117647, mean=7.826. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>comment_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 132,579 | 3.03% |
| 1.0 | 28,821 | 0.66% |
| 0.5 | 24,024 | 0.55% |
| 0.3333333333333333 | 19,800 | 0.45% |
| 0.25 | 17,245 | 0.39% |
| 2.0 | 14,815 | 0.34% |
| 0.2 | 14,254 | 0.33% |
| 0.6666666666666666 | 13,288 | 0.30% |
| 0.1666666666666666 | 13,277 | 0.30% |
| 0.1428571428571428 | 11,547 | 0.26% |

Finite known numeric values: min=0.0, median=1.182, p90=14.46, p99=149.5, max=66844.91666666667, mean=10.19. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>comment_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 132,579 | 3.03% |
| 1.0 | 30,662 | 0.70% |
| 0.5 | 26,488 | 0.61% |
| 0.3333333333333333 | 22,074 | 0.50% |
| 0.25 | 19,353 | 0.44% |
| 0.2 | 16,104 | 0.37% |
| 0.1666666666666666 | 14,882 | 0.34% |
| 0.6666666666666666 | 14,517 | 0.33% |
| 2.0 | 14,497 | 0.33% |
| 0.1428571428571428 | 12,821 | 0.29% |

Finite known numeric values: min=0.0, median=1, p90=12.18, p99=125.6, max=53764.25, mean=8.509. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>direct_comment_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 140,279 | 3.21% |
| 1.0 | 30,132 | 0.69% |
| 0.5 | 26,979 | 0.62% |
| 0.3333333333333333 | 22,770 | 0.52% |
| 0.25 | 19,866 | 0.45% |
| 0.2 | 16,743 | 0.38% |
| 0.1666666666666666 | 15,430 | 0.35% |
| 0.6666666666666666 | 14,535 | 0.33% |
| 2.0 | 14,428 | 0.33% |
| 0.1428571428571428 | 13,545 | 0.31% |

Finite known numeric values: min=0.0, median=0.9178, p90=10.91, p99=106.5, max=33603.75, mean=7.309. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>reply_comment_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 932,107 | 21.32% |
| 0.3333333333333333 | 21,314 | 0.49% |
| 0.5 | 21,046 | 0.48% |
| 0.25 | 20,698 | 0.47% |
| 0.1666666666666666 | 19,312 | 0.44% |
| 0.2 | 19,215 | 0.44% |
| 1.0 | 18,006 | 0.41% |
| 0.1428571428571428 | 17,665 | 0.40% |
| 0.125 | 17,662 | 0.40% |
| 0.1111111111111111 | 16,555 | 0.38% |

Finite known numeric values: min=0.0, median=0.15, p90=3.145, p99=44.45, max=33241.166666666664, mean=2.882. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>delete_comment_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,427,001 | 32.64% |
| 0.1666666666666666 | 20,952 | 0.48% |
| 0.125 | 19,703 | 0.45% |
| 0.1111111111111111 | 19,661 | 0.45% |
| 0.1 | 19,480 | 0.45% |
| 0.0833333333333333 | 19,230 | 0.44% |
| 0.0909090909090909 | 18,988 | 0.43% |
| 0.1428571428571428 | 18,870 | 0.43% |
| 0.2 | 18,700 | 0.43% |
| 0.25 | 18,161 | 0.42% |

Finite known numeric values: min=0.0, median=0.04167, p90=0.6875, p99=7.5, max=4012.1666666666665, mean=0.4912. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>delete_comment_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,427,001 | 32.64% |
| 0.1666666666666666 | 22,068 | 0.50% |
| 0.125 | 20,623 | 0.47% |
| 0.1111111111111111 | 20,561 | 0.47% |
| 0.1 | 20,504 | 0.47% |
| 0.0833333333333333 | 20,301 | 0.46% |
| 0.0909090909090909 | 20,072 | 0.46% |
| 0.1428571428571428 | 19,517 | 0.45% |
| 0.2 | 19,110 | 0.44% |
| 0.0714285714285714 | 18,962 | 0.43% |

Finite known numeric values: min=0.0, median=0.03846, p90=0.5806, p99=6.094, max=1236.4166666666667, mean=0.3934. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>comment_like_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 683,135 | 15.63% |
| 0.5 | 18,566 | 0.42% |
| 1.0 | 18,329 | 0.42% |
| 0.3333333333333333 | 17,937 | 0.41% |
| 0.25 | 17,216 | 0.39% |
| 0.1666666666666666 | 15,895 | 0.36% |
| 0.2 | 15,867 | 0.36% |
| 0.1428571428571428 | 14,123 | 0.32% |
| 0.125 | 13,840 | 0.32% |
| 0.1111111111111111 | 12,975 | 0.30% |

Finite known numeric values: min=0.0, median=0.55, p90=20.57, p99=358.6, max=82437.66666666667, mean=20.41. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>comment_like_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 683,135 | 15.63% |
| 0.3333333333333333 | 22,527 | 0.52% |
| 0.25 | 21,832 | 0.50% |
| 0.1666666666666666 | 21,087 | 0.48% |
| 0.2 | 20,850 | 0.48% |
| 0.5 | 20,795 | 0.48% |
| 0.1428571428571428 | 19,335 | 0.44% |
| 0.125 | 19,020 | 0.44% |
| 0.1111111111111111 | 17,798 | 0.41% |
| 0.1 | 17,332 | 0.40% |

Finite known numeric values: min=0.0, median=0.2857, p90=10.79, p99=179, max=35342.0, mean=10.13. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>follow_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 268,345 | 6.14% |
| 1.0 | 23,887 | 0.55% |
| 0.5 | 23,021 | 0.53% |
| 0.3333333333333333 | 21,938 | 0.50% |
| 0.25 | 19,780 | 0.45% |
| 0.2 | 17,691 | 0.40% |
| 0.1666666666666666 | 17,126 | 0.39% |
| 0.1428571428571428 | 14,844 | 0.34% |
| 0.125 | 14,531 | 0.33% |
| 0.1111111111111111 | 12,844 | 0.29% |

Finite known numeric values: min=0.0, median=1.115, p90=16.11, p99=139.1, max=55359.91666666666, mean=9.319. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>follow_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 268,345 | 6.14% |
| 1.0 | 23,936 | 0.55% |
| 0.5 | 23,134 | 0.53% |
| 0.3333333333333333 | 22,088 | 0.51% |
| 0.25 | 19,959 | 0.46% |
| 0.2 | 17,932 | 0.41% |
| 0.1666666666666666 | 17,383 | 0.40% |
| 0.1428571428571428 | 15,129 | 0.35% |
| 0.125 | 14,739 | 0.34% |
| 0.1111111111111111 | 13,117 | 0.30% |

Finite known numeric values: min=0.0, median=1.097, p90=15.91, p99=137.7, max=55347.25, mean=9.21. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>cancel_follow_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 742,481 | 16.98% |
| 0.1666666666666666 | 26,829 | 0.61% |
| 0.3333333333333333 | 26,471 | 0.61% |
| 0.25 | 26,332 | 0.60% |
| 0.2 | 26,157 | 0.60% |
| 0.1428571428571428 | 24,929 | 0.57% |
| 0.125 | 24,747 | 0.57% |
| 0.1111111111111111 | 23,098 | 0.53% |
| 0.1 | 22,691 | 0.52% |
| 0.5 | 22,675 | 0.52% |

Finite known numeric values: min=0.0, median=0.1217, p90=1.406, p99=11.55, max=10219.113636363636, mean=0.8344. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>cancel_follow_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 742,481 | 16.98% |
| 0.1666666666666666 | 26,937 | 0.62% |
| 0.25 | 26,494 | 0.61% |
| 0.3333333333333333 | 26,358 | 0.60% |
| 0.2 | 26,133 | 0.60% |
| 0.1428571428571428 | 25,049 | 0.57% |
| 0.125 | 24,820 | 0.57% |
| 0.1111111111111111 | 23,145 | 0.53% |
| 0.1 | 22,829 | 0.52% |
| 0.5 | 22,616 | 0.52% |

Finite known numeric values: min=0.0, median=0.12, p90=1.387, p99=11.44, max=2768.7727272727275, mean=0.8169. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>share_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 642,236 | 14.69% |
| 0.3333333333333333 | 24,004 | 0.55% |
| 0.25 | 22,794 | 0.52% |
| 0.2 | 22,041 | 0.50% |
| 0.1666666666666666 | 21,877 | 0.50% |
| 0.5 | 21,819 | 0.50% |
| 0.1428571428571428 | 20,205 | 0.46% |
| 0.125 | 19,712 | 0.45% |
| 0.1111111111111111 | 18,779 | 0.43% |
| 1.0 | 18,280 | 0.42% |

Finite known numeric values: min=0.0, median=0.25, p90=4.318, p99=48.94, max=12348.0, mean=3.146. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>share_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 642,236 | 14.69% |
| 0.3333333333333333 | 24,628 | 0.56% |
| 0.25 | 23,494 | 0.54% |
| 0.1666666666666666 | 23,189 | 0.53% |
| 0.2 | 22,739 | 0.52% |
| 0.5 | 21,580 | 0.49% |
| 0.1428571428571428 | 21,262 | 0.49% |
| 0.125 | 20,964 | 0.48% |
| 0.1111111111111111 | 20,044 | 0.46% |
| 0.1 | 19,427 | 0.44% |

Finite known numeric values: min=0.0, median=0.2216, p90=3.824, p99=43.62, max=9737.083333333334, mean=2.794. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>download_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,041,583 | 23.82% |
| 0.1666666666666666 | 22,309 | 0.51% |
| 0.25 | 21,199 | 0.48% |
| 0.3333333333333333 | 21,047 | 0.48% |
| 0.2 | 21,022 | 0.48% |
| 0.125 | 20,918 | 0.48% |
| 0.1428571428571428 | 20,553 | 0.47% |
| 0.1 | 19,772 | 0.45% |
| 0.1111111111111111 | 19,725 | 0.45% |
| 0.0909090909090909 | 18,808 | 0.43% |

Finite known numeric values: min=0.0, median=0.125, p90=2.914, p99=37.19, max=10020.166666666666, mean=2.335. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>download_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 1,041,583 | 23.82% |
| 0.1666666666666666 | 22,589 | 0.52% |
| 0.25 | 21,244 | 0.49% |
| 0.3333333333333333 | 21,231 | 0.49% |
| 0.2 | 21,190 | 0.48% |
| 0.125 | 21,114 | 0.48% |
| 0.1428571428571428 | 20,600 | 0.47% |
| 0.1 | 19,919 | 0.46% |
| 0.1111111111111111 | 19,873 | 0.45% |
| 0.0833333333333333 | 19,013 | 0.43% |

Finite known numeric values: min=0.0, median=0.125, p90=2.776, p99=33.56, max=9964.416666666666, mean=2.08. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>report_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 3,827,275 | 87.54% |
| 0.0053763440860215 | 5,611 | 0.13% |
| 0.0833333333333333 | 5,046 | 0.12% |
| 0.0555555555555555 | 4,848 | 0.11% |
| 0.0416666666666666 | 4,505 | 0.10% |
| 0.0333333333333333 | 4,336 | 0.10% |
| 0.0277777777777777 | 4,088 | 0.09% |
| 0.010752688172043 | 4,056 | 0.09% |
| 0.1666666666666666 | 3,998 | 0.09% |
| 0.0238095238095238 | 3,801 | 0.09% |

Finite known numeric values: min=0.0, median=0, p90=0.01031, p99=0.1791, max=332.75, mean=0.01279. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>report_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 3,827,275 | 87.54% |
| 0.0053763440860215 | 5,919 | 0.14% |
| 0.0833333333333333 | 5,132 | 0.12% |
| 0.0555555555555555 | 5,013 | 0.11% |
| 0.0416666666666666 | 4,638 | 0.11% |
| 0.0333333333333333 | 4,511 | 0.10% |
| 0.0277777777777777 | 4,275 | 0.10% |
| 0.010752688172043 | 4,260 | 0.10% |
| 0.0238095238095238 | 3,967 | 0.09% |
| 0.1666666666666666 | 3,910 | 0.09% |

Finite known numeric values: min=0.0, median=0, p90=0.009901, p99=0.1579, max=320.0833333333333, mean=0.01135. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>reduce_similar_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 344,609 | 7.88% |
| 0.5 | 28,460 | 0.65% |
| 0.3333333333333333 | 27,345 | 0.63% |
| 1.0 | 26,701 | 0.61% |
| 0.25 | 24,347 | 0.56% |
| 0.2 | 22,302 | 0.51% |
| 0.1666666666666666 | 21,091 | 0.48% |
| 0.1428571428571428 | 18,316 | 0.42% |
| 0.125 | 17,444 | 0.40% |
| 0.1111111111111111 | 15,709 | 0.36% |

Finite known numeric values: min=0.0, median=0.5806, p90=5.843, p99=42.43, max=4013.203703703704, mean=3.134. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>reduce_similar_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 344,609 | 7.88% |
| 0.5 | 28,502 | 0.65% |
| 0.3333333333333333 | 27,722 | 0.63% |
| 1.0 | 26,717 | 0.61% |
| 0.25 | 24,519 | 0.56% |
| 0.2 | 22,469 | 0.51% |
| 0.1666666666666666 | 21,336 | 0.49% |
| 0.1428571428571428 | 18,604 | 0.43% |
| 0.125 | 17,642 | 0.40% |
| 0.1111111111111111 | 15,862 | 0.36% |

Finite known numeric values: min=0.0, median=0.5625, p90=5.518, p99=39.97, max=3987.796296296296, mean=2.968. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>collect_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 283,234 | 6.48% |
| 1.0 | 23,649 | 0.54% |
| 0.5 | 21,683 | 0.50% |
| 0.3333333333333333 | 19,457 | 0.45% |
| 0.25 | 17,578 | 0.40% |
| 0.2 | 15,448 | 0.35% |
| 0.1666666666666666 | 15,028 | 0.34% |
| 0.1428571428571428 | 12,943 | 0.30% |
| 0.125 | 12,388 | 0.28% |
| 0.6666666666666666 | 12,080 | 0.28% |

Finite known numeric values: min=0.0, median=1.457, p90=23.83, p99=212.3, max=28560.75, mean=13.69. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>collect_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 283,234 | 6.48% |
| 1.0 | 23,886 | 0.55% |
| 0.5 | 21,935 | 0.50% |
| 0.3333333333333333 | 19,699 | 0.45% |
| 0.25 | 17,793 | 0.41% |
| 0.2 | 15,704 | 0.36% |
| 0.1666666666666666 | 15,315 | 0.35% |
| 0.1428571428571428 | 13,224 | 0.30% |
| 0.125 | 12,683 | 0.29% |
| 0.6666666666666666 | 12,087 | 0.28% |

Finite known numeric values: min=0.0, median=1.4, p90=22.87, p99=204.2, max=28233.416666666668, mean=13.17. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>cancel_collect_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 560,189 | 12.81% |
| 0.3333333333333333 | 24,950 | 0.57% |
| 0.25 | 23,682 | 0.54% |
| 0.1666666666666666 | 23,076 | 0.53% |
| 0.2 | 22,556 | 0.52% |
| 0.5 | 22,391 | 0.51% |
| 0.1428571428571428 | 21,416 | 0.49% |
| 0.125 | 20,580 | 0.47% |
| 0.1111111111111111 | 19,501 | 0.45% |
| 1.0 | 18,636 | 0.43% |

Finite known numeric values: min=0.0, median=0.3846, p90=6.076, p99=50.48, max=10741.5, mean=3.381. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>cancel_collect_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 560,189 | 12.81% |
| 0.3333333333333333 | 25,336 | 0.58% |
| 0.25 | 24,151 | 0.55% |
| 0.1666666666666666 | 23,939 | 0.55% |
| 0.2 | 23,219 | 0.53% |
| 0.5 | 22,394 | 0.51% |
| 0.1428571428571428 | 22,161 | 0.51% |
| 0.125 | 21,315 | 0.49% |
| 0.1111111111111111 | 20,268 | 0.46% |
| 0.1 | 19,349 | 0.44% |

Finite known numeric values: min=0.0, median=0.36, p90=5.654, p99=47.61, max=10595.5, mean=3.175. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>direct_comment_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 140,279 | 3.21% |
| 1.0 | 30,376 | 0.69% |
| 0.5 | 27,765 | 0.64% |
| 0.3333333333333333 | 23,340 | 0.53% |
| 0.25 | 20,794 | 0.48% |
| 0.2 | 17,544 | 0.40% |
| 0.1666666666666666 | 16,204 | 0.37% |
| 0.6666666666666666 | 14,757 | 0.34% |
| 0.1428571428571428 | 14,349 | 0.33% |
| 2.0 | 14,094 | 0.32% |

Finite known numeric values: min=0.0, median=0.8621, p90=10.17, p99=99.14, max=32614.83333333333, mean=6.764. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>reply_comment_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 932,107 | 21.32% |
| 0.1666666666666666 | 25,189 | 0.58% |
| 0.25 | 24,209 | 0.55% |
| 0.2 | 24,083 | 0.55% |
| 0.125 | 24,081 | 0.55% |
| 0.1428571428571428 | 23,631 | 0.54% |
| 0.3333333333333333 | 23,628 | 0.54% |
| 0.1 | 22,738 | 0.52% |
| 0.1111111111111111 | 22,617 | 0.52% |
| 0.0909090909090909 | 21,139 | 0.48% |

Finite known numeric values: min=0.0, median=0.1, p90=2.125, p99=32.79, max=27074.0, mean=2.122. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>share_all_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 486,669 | 11.13% |
| 0.3333333333333333 | 23,697 | 0.54% |
| 0.5 | 23,444 | 0.54% |
| 0.25 | 21,952 | 0.50% |
| 1.0 | 21,881 | 0.50% |
| 0.2 | 20,657 | 0.47% |
| 0.1666666666666666 | 19,951 | 0.46% |
| 0.1428571428571428 | 18,205 | 0.42% |
| 0.125 | 17,550 | 0.40% |
| 0.1111111111111111 | 16,108 | 0.37% |

Finite known numeric values: min=0.0, median=0.4595, p90=7.477, p99=76.58, max=20171.916666666668, mean=4.985. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>share_all_user_num: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 486,669 | 11.13% |
| 0.3333333333333333 | 24,773 | 0.57% |
| 0.5 | 23,275 | 0.53% |
| 0.25 | 22,933 | 0.52% |
| 0.2 | 21,863 | 0.50% |
| 0.1666666666666666 | 21,465 | 0.49% |
| 1.0 | 20,780 | 0.48% |
| 0.1428571428571428 | 19,520 | 0.45% |
| 0.125 | 19,222 | 0.44% |
| 0.1111111111111111 | 17,617 | 0.40% |

Finite known numeric values: min=0.0, median=0.3939, p90=6.446, p99=66.4, max=18585.166666666668, mean=4.319. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

<details>
<summary>outsite_share_all_cnt: value counts and distribution</summary>

Rows: 4,371,868; dtype: DOUBLE. Invalid rule: negative known numeric. Frequency scope: top 10 values; includes null if frequent.

| Value | Count | Share of rows |
|---|---|---|
| 0.0 | 654,303 | 14.97% |
| 0.3333333333333333 | 22,984 | 0.53% |
| 0.25 | 21,808 | 0.50% |
| 0.5 | 21,500 | 0.49% |
| 0.2 | 21,221 | 0.49% |
| 0.1666666666666666 | 21,016 | 0.48% |
| 0.1428571428571428 | 19,376 | 0.44% |
| 0.125 | 18,750 | 0.43% |
| 1.0 | 18,725 | 0.43% |
| 0.1111111111111111 | 17,734 | 0.41% |

Finite known numeric values: min=0.0, median=0.2727, p90=4.63, p99=49.57, max=13504.833333333334, mean=3.214. Units follow the source field: duration/play fields in milliseconds, register_days in days, count fields in counts. Monthly values are averages, not event-time observations.

</details>

## Tag token frequencies

The 1,188 non-null tag strings contain 58 distinct comma-separated tokens. Token frequency counts occurrences across video rows; multi-tag videos contribute to several tokens.

| Token | Occurrences |
|---|---|
| 34 | 576,886 |
| 67 | 397,438 |
| 39 | 365,543 |
| 12 | 255,350 |
| 3 | 235,909 |
| 19 | 190,374 |
| 9 | 189,619 |
| 11 | 188,904 |
| 2 | 182,312 |
| 65 | 179,060 |
| 5 | 177,114 |
| 68 | 163,838 |
| 6 | 160,103 |
| 20 | 158,126 |
| 28 | 157,792 |
| 8 | 157,338 |
| 25 | 148,064 |
| 13 | 136,089 |
| 62 | 124,057 |
| 18 | 123,726 |
| 1 | 107,717 |
| 7 | 106,955 |
| 15 | 81,700 |
| 4 | 73,914 |
| 16 | 61,954 |
| 17 | 60,551 |
| 43 | 52,247 |
| 36 | 44,416 |
| 26 | 41,484 |
| 38 | 33,826 |
| 10 | 32,480 |
| 60 | 29,956 |
| 35 | 28,675 |
| 24 | 26,352 |
| 27 | 24,771 |
| 22 | 22,407 |
| 14 | 20,342 |
| 42 | 18,283 |
| 21 | 13,523 |
| 30 | 13,361 |
| 37 | 11,440 |
| 29 | 10,461 |
| 23 | 8,511 |
| 53 | 6,578 |
| 64 | 2,619 |
| 54 | 2,545 |
| 40 | 1,242 |
| 56 | 1,007 |
| 52 | 145 |
| 49 | 142 |
| 45 | 136 |
| 41 | 124 |
| 50 | 104 |
| 48 | 99 |
| 51 | 53 |
| 46 | 34 |
| 44 | 25 |
| 47 | 1 |
