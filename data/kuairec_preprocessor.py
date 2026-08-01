from __future__ import annotations

import json

import pandas as pd
from pandas.errors import ParserError, EmptyDataError
from pathlib import Path

# Wide-path equality crosses (Cheng et al.). Written as float 0/1 before
# sparse factorization. Selected by interaction_lift = lift(joint) /
# (lift(a) * lift(b))
WIDE_CROSS_SPECS: list[dict] = [
    {
        "name": "cross_old_middle_active_users",
        "input_features": {
            "register_days_range": "181-365",
            "user_active_degree": "middle_active",
        },
    },  # interaction_lift=0.75
    {
        "name": "cross_high_friend_tier2_city_users",
        "input_features": {
            "friend_user_num_range": "[30,60)",
            "fre_city_level": "Tier-2 city",
        },
    },  # interaction_lift=1.63
    {
        "name": "cross_socially_active_new_users",
        "input_features": {
            "friend_user_num_range": "[5,30)",
            "register_days_range": "61-90",
        },
    },  # interaction_lift=1.46
    {
        "name": "cross_high_follow_high_friend_users",
        "input_features": {
            "follow_user_num_range": "(250,500]",
            "friend_user_num_range": "[30,60)",
        },
    },  # interaction_lift=1.46
    {
        "name": "cross_isolated_users",
        "input_features": {
            "follow_user_num_range": "0",
            "fans_user_num_range": "0",
        },
    },  # interaction_lift=0.72
    {
        "name": "cross_unknown_active_recent_users",
        "input_features": {
            "user_active_degree": "UNKNOWN",
            "register_days_range": "61-90",
        },
    },  # interaction_lift=1.46, support=0.07% (n=3183)
    {
        "name": "cross_midage_high_friend_users",
        "input_features": {
            "age_range": "41-49",
            "friend_user_num_range": "[30,60)",
        },
    },  # interaction_lift=1.39, support=0.14% (n=6430)
]

# All categoricals that go through nn.Embedding (factorized in ETL).
# Dense numerics (durations, counts, …) stay numeric.
SPARSE_FEATURE_COLS: list[str] = [
    # IDs
    "user_id",
    "video_id",
    "author_id",
    "music_id",
    "video_tag_id",
    # user profile categoricals
    "gender",
    "age_range",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "is_live_author",
    "is_photo_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *[f"user_onehot_feat{i}" for i in range(18)],
    # context
    "phone_brand",
    "phone_model",
    "mod_price", 
    "fre_country",
    "fre_country_region",
    "fre_province",
    "fre_city",
    "fre_city_level",
    "fre_community_type",
    "platform",
    "os_version",
    "app_version",
    "app_download_channel",
    "isp",
    # video static categoricals
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
]


class KuaiRecPreprocessor:
    def __init__(
        self,
        use_small: bool = True,
        seed: int = 42
    ) -> None:
        self.use_small = use_small
        self.seed = seed
        self.raw_data_dir = Path("data/raw/kuairec")
        self.extract_data_base_dir = Path("data/extract/kuairec")
        self.sparse_cols = list(SPARSE_FEATURE_COLS)

    def run(self) -> None:
        df = self._extract()
        df, vocab_size = self._transform(df)
        self._load(df, vocab_size)
        
    def _extract(self):
        base_path = self.raw_data_dir / "small_matrix.csv" if self.use_small else self.raw_data_dir / "big_matrix.csv"
        user_features_path = self.raw_data_dir / "user_features.csv"
        user_features_raw_path = self.raw_data_dir / "user_features_raw.csv"
        item_daily_feature_path = self.raw_data_dir / "item_daily_features.csv"

        try:
            base = pd.read_csv(base_path)
            users = pd.read_csv(user_features_path)
            users_raw = pd.read_csv(user_features_raw_path)
            items = pd.read_csv(item_daily_feature_path)            
            print(f"Info: CSV {self.raw_data_dir} read and parsed successfully")
        except FileNotFoundError:
            print(f"Error: Raw data could not be found in {self.raw_data_dir}. Check the directory path.")
            raise
        except EmptyDataError:
            print(f"Error: Empty Data found in {self.raw_data_dir}")
            raise
        except ParserError:
            print(f"Error: The CSV {self.raw_data_dir} is unparsable due to data formatting flaws.")
            raise
    
        # ~3.9% of rows have a genuinely missing date/timestamp in the raw
        # file. Drop them before anything else: left in, they (a) can't be
        # date-joined to items below and (b) all land in the `test` split
        # later (df.sort_values(by="timestamp") pushes NaN to the tail via
        # the pandas default na_position="last"), corrupting test-only.
        n_before = len(base)
        base = base.dropna(subset=["date", "timestamp"])
        n_dropped = n_before - len(base)
        print(f"Info: dropped {n_dropped} rows ({n_dropped / n_before:.2%}) with missing date/timestamp")

        # Align dtypes on join keys before merge
        for col in ("user_id", "video_id"):
            base[col] = base[col].astype("int64")
        users["user_id"] = users["user_id"].astype("int64")
        users_raw["user_id"] = users_raw["user_id"].astype("int64")
        items["video_id"] = items["video_id"].astype("int64")
        # date is YYYYMMDD encoded as a number (e.g. 20200705.0 / 20200705),
        # not a date string -- pd.to_datetime() on a raw numeric column reads
        # it as ns-since-epoch and silently collapses every row to
        # 1970-01-01. Force the real format instead.
        base["date"] = pd.to_datetime(base["date"].astype("int64").astype(str), format="%Y%m%d")
        items["date"] = pd.to_datetime(items["date"].astype("int64").astype(str), format="%Y%m%d")
        
        # Deduplicate right tables on the join key
        users = users.drop_duplicates(subset=["user_id"], keep="last")
        users_raw = users_raw.drop_duplicates(subset=["user_id"], keep="last")
        items = items.drop_duplicates(subset=["video_id", "date"], keep="last")

        users = users.rename(columns={f"onehot_feat{i}": f"user_onehot_feat{i}" for i in range(18)})
        users_raw = users_raw[[
            "user_id", "gender", "age_range", "is_live_author", "is_photo_author",
            "phone_brand", "phone_model", "mod_price",
            "fre_country", "fre_country_region", "fre_province", "fre_city",
            "fre_city_level", "fre_community_type",
            "platform", "os_version", "app_version", "app_download_channel", "isp",
        ]]
        # `video_duration` also exists on `base` (the interaction matrix, kept
        # as the label-adjacent dense feature in schema.py) -- drop the item
        # table's copy so the merge below doesn't produce a duplicate/suffixed
        # column. `video_tag_name` is a free-text label of `video_tag_id`
        # (e.g. "JianZhu" for tag_id 841); the ID is already kept as the
        # sparse/embedding feature, so the redundant string is dropped rather
        # than adding an unencoded high-cardinality text column.
        items = items.drop(columns=["video_duration", "video_tag_name"])
        items = items.rename(columns={
            "show_cnt": "video_daily_show_cnt",
            "show_user_num": "video_daily_show_user_num",
            "play_cnt": "video_daily_play_cnt",
            "play_user_num": "video_daily_play_user_num",
            "play_duration": "video_daily_play_duration",
            "complete_play_cnt": "video_daily_complete_play_cnt",
            "complete_play_user_num": "video_daily_complete_play_user_num",
            "valid_play_cnt": "video_daily_valid_play_cnt",
            "valid_play_user_num": "video_daily_valid_play_user_num",
            "long_time_play_cnt": "video_daily_long_time_play_cnt",
            "long_time_play_user_num": "video_daily_long_time_play_user_num",
            "short_time_play_cnt": "video_daily_short_time_play_cnt",
            "short_time_play_user_num": "video_daily_short_time_play_user_num",
            "play_progress": "video_daily_play_progress",
            "comment_stay_duration": "video_daily_comment_stay_duration",
            "like_cnt": "video_daily_like_cnt",
            "like_user_num": "video_daily_like_user_num",
            "click_like_cnt": "video_daily_click_like_cnt",
            "double_click_cnt": "video_daily_double_click_cnt",
            "cancel_like_cnt": "video_daily_cancel_like_cnt",
            "cancel_like_user_num": "video_daily_cancel_like_user_num",
            "comment_cnt": "video_daily_comment_cnt",
            "comment_user_num": "video_daily_comment_user_num",
            "direct_comment_cnt": "video_daily_direct_comment_cnt",
            "reply_comment_cnt": "video_daily_reply_comment_cnt",
            "delete_comment_cnt": "video_daily_delete_comment_cnt",
            "delete_comment_user_num": "video_daily_delete_comment_user_num",
            "comment_like_cnt": "video_daily_comment_like_cnt",
            "comment_like_user_num": "video_daily_comment_like_user_num",
            "follow_cnt": "video_daily_follow_cnt",
            "follow_user_num": "video_daily_follow_user_num",
            "cancel_follow_cnt": "video_daily_cancel_follow_cnt",
            "cancel_follow_user_num": "video_daily_cancel_follow_user_num",
            "share_cnt": "video_daily_share_cnt",
            "share_user_num": "video_daily_share_user_num",
            "download_cnt": "video_daily_download_cnt",
            "download_user_num": "video_daily_download_user_num",
            "report_cnt": "video_daily_report_cnt",
            "report_user_num": "video_daily_report_user_num",
            "reduce_similar_cnt": "video_daily_reduce_similar_cnt",
            "reduce_similar_user_num": "video_daily_reduce_similar_user_num",
            "collect_cnt": "video_daily_collect_cnt",
            "collect_user_num": "video_daily_collect_user_num",
            "cancel_collect_cnt": "video_daily_cancel_collect_cnt",
            "cancel_collect_user_num": "video_daily_cancel_collect_user_num",
        })
        
        df = base.merge(users, on="user_id", how="left", validate="many_to_one")
        df = df.merge(users_raw, on="user_id", how="left", validate="many_to_one")
        df = df.merge(items, on=["video_id", "date"], how="left", validate="many_to_one")
        assert len(df) == len(base)

        return df

    def _transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        cols = [c for c in self.sparse_cols if c in df.columns]
        missing = [c for c in self.sparse_cols if c not in df.columns]
        if missing:
            print(f"Warning: sparse cols not in frame (skipped): {missing}")

        # Wide crosses on raw string/categorical values (before factorization).
        df = self._add_cross_features(df)

        # Categorical / ID columns -> contiguous embedding indices (0 = missing)
        vocab_size = self._id_to_embedding_idx(df, cols=cols)

        return df, vocab_size

    def _add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equality-AND crosses → float 0/1 columns (wide path)."""
        for spec in WIDE_CROSS_SPECS:
            name = spec["name"]
            pairs: dict[str, str] = spec["input_features"]
            mask = pd.Series(True, index=df.index)
            for field, value in pairs.items():
                if field not in df.columns:
                    raise KeyError(
                        f"WIDE_CROSS_SPECS '{name}' needs column '{field}' "
                        "before factorize"
                    )
                mask &= df[field].astype(str) == str(value)
            df[name] = mask.astype("float32")
            print(f"Info: cross {name} positive_rate={float(df[name].mean()):.6f}")
        return df

    def _load(self, df: pd.DataFrame, vocab_size: dict[str, int]) -> None:
        self.extract_data_base_dir.mkdir(parents=True, exist_ok=True)

        meta_path = self.extract_data_base_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            # vocab size per id col (includes reserved index 0 for missing)
            json.dump(vocab_size, f, indent=2)

        train_path = self.extract_data_base_dir / "train.parquet"
        val_path = self.extract_data_base_dir / "val.parquet"
        test_path = self.extract_data_base_dir / "test.parquet"

        train_ratio, val_ratio = 0.8, 0.1
        df_sorted = df.sort_values(by="timestamp", ascending=True)
        train_end = int(len(df_sorted) * train_ratio)
        val_end = int(len(df_sorted) * (train_ratio + val_ratio))

        try:
            df_sorted.iloc[:train_end].to_parquet(train_path, engine="pyarrow")
            df_sorted.iloc[train_end:val_end].to_parquet(val_path, engine="pyarrow")
            df_sorted.iloc[val_end:].to_parquet(test_path, engine="pyarrow")
            print(
                f"Info: wrote {train_path}, {val_path}, {test_path}, {meta_path}"
            )
            print(f"Info: vocab_size={vocab_size}")
        except Exception as e:
            print(f"Error: failed to write parquet/meta: {e}")
            raise

    def _id_to_embedding_idx(
        self, df: pd.DataFrame, cols: list[str]
    ) -> dict[str, int]:
        """
        Map raw categorical IDs -> contiguous embedding indices.

        Index 0 is reserved for missing (NaN after left joins).
        Real IDs map to 1..K. Returned value is nn.Embedding vocab size (= K + 1).
        """
        vocab_size: dict[str, int] = {}
        for col in cols:
            codes, _ = pd.factorize(df[col], sort=True)
            df[col] = (codes + 1).astype("int64")
            vocab_size[col] = int(df[col].max()) + 1
        return vocab_size


if __name__ == "__main__":
    # Regenerates data/extract/kuairec/{train,val,test}.parquet + meta.json
    # from data/raw/kuairec/ (see data/scripts/download_datasets.py).
    KuaiRecPreprocessor(use_small=True).run()
