"""
KuaiRec training-row schema (documentation / lookup).

Not used at runtime by Dataset/Trainer — parquet columns + FeatureConfig drive I/O.

Types after ETL:
  - sparse / ID fields → int embedding indices (0 = missing)
  - dense counts / durations → numeric
  - cross_* → float 0/1 (engineered in preprocessor for the wide path)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KuaiRecInstance:
    # keys / time
    user_id: int = 0
    video_id: int = 0
    date: str = ""
    timestamp: int = 0

    # interaction (label for M1 watch regression: watch_ratio)
    play_duration: float = 0.0
    video_duration: float = 0.0
    watch_ratio: float = 0.0

    # engineered wide crosses (see KuaiRecPreprocessor.WIDE_CROSS_SPECS)
    cross_new_teenager_female_users: float = 0.0
    cross_young_male_active_users: float = 0.0
    cross_old_middle_active_users: float = 0.0

    # user profile (factorized ints in parquet except where noted)
    gender: int = 0
    age_range: int = 0
    user_active_degree: int = 0
    is_lowactive_period: int = 0
    is_live_streamer: int = 0
    is_video_author: int = 0
    is_live_author: int = 0
    is_photo_author: int = 0
    follow_user_num: int = 0
    follow_user_num_range: int = 0
    fans_user_num: int = 0
    fans_user_num_range: int = 0
    friend_user_num: int = 0
    friend_user_num_range: int = 0
    register_days: int = 0
    register_days_range: int = 0
    user_onehot_feat0: int = 0
    user_onehot_feat1: int = 0
    user_onehot_feat2: int = 0
    user_onehot_feat3: int = 0
    user_onehot_feat4: int = 0
    user_onehot_feat5: int = 0
    user_onehot_feat6: int = 0
    user_onehot_feat7: int = 0
    user_onehot_feat8: int = 0
    user_onehot_feat9: int = 0
    user_onehot_feat10: int = 0
    user_onehot_feat11: int = 0
    user_onehot_feat12: int = 0
    user_onehot_feat13: int = 0
    user_onehot_feat14: int = 0
    user_onehot_feat15: int = 0
    user_onehot_feat16: int = 0
    user_onehot_feat17: int = 0

    # context
    phone_brand: int = 0
    phone_model: int = 0
    mod_price: int = 0
    fre_country: int = 0
    fre_country_region: int = 0
    fre_province: int = 0
    fre_city: int = 0
    fre_city_level: int = 0
    fre_community_type: int = 0
    platform: int = 0
    os_version: int = 0
    app_version: int = 0
    app_download_channel: int = 0
    isp: int = 0

    # video static
    author_id: int = 0
    video_type: int = 0
    upload_dt: int = 0
    upload_type: int = 0
    visible_status: int = 0
    video_width: int = 0
    video_height: int = 0
    music_id: int = 0
    video_tag_id: int = 0

    # video daily stats
    video_daily_show_cnt: int = 0
    video_daily_show_user_num: int = 0
    video_daily_play_cnt: int = 0
    video_daily_play_user_num: int = 0
    video_daily_play_duration: int = 0
    video_daily_complete_play_cnt: int = 0
    video_daily_complete_play_user_num: int = 0
    video_daily_valid_play_cnt: int = 0
    video_daily_valid_play_user_num: int = 0
    video_daily_long_time_play_cnt: int = 0
    video_daily_long_time_play_user_num: int = 0
    video_daily_short_time_play_cnt: int = 0
    video_daily_short_time_play_user_num: int = 0
    video_daily_play_progress: float = 0.0
    video_daily_comment_stay_duration: int = 0
    video_daily_like_cnt: int = 0
    video_daily_like_user_num: int = 0
    video_daily_click_like_cnt: int = 0
    video_daily_double_click_cnt: int = 0
    video_daily_cancel_like_cnt: int = 0
    video_daily_cancel_like_user_num: int = 0
    video_daily_comment_cnt: int = 0
    video_daily_comment_user_num: int = 0
    video_daily_direct_comment_cnt: int = 0
    video_daily_reply_comment_cnt: int = 0
    video_daily_delete_comment_cnt: int = 0
    video_daily_delete_comment_user_num: int = 0
    video_daily_comment_like_cnt: int = 0
    video_daily_comment_like_user_num: int = 0
    video_daily_follow_cnt: int = 0
    video_daily_follow_user_num: int = 0
    video_daily_cancel_follow_cnt: int = 0
    video_daily_cancel_follow_user_num: int = 0
    video_daily_share_cnt: int = 0
    video_daily_share_user_num: int = 0
    video_daily_download_cnt: int = 0
    video_daily_download_user_num: int = 0
    video_daily_report_cnt: int = 0
    video_daily_report_user_num: int = 0
    video_daily_reduce_similar_cnt: int = 0
    video_daily_reduce_similar_user_num: int = 0
    video_daily_collect_cnt: int = 0
    video_daily_collect_user_num: int = 0
    video_daily_cancel_collect_cnt: int = 0
    video_daily_cancel_collect_user_num: int = 0
