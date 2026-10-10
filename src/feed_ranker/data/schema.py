"""Raw recommendation observations and prepared tensor-batch contracts.

Raw IDs retain dataset identity. Missing optional attributes are None, never an
invented zero. Fields listed here are not automatically available model inputs:
feature selection and prediction-time availability are separate decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch

KEYS = ["user_id", "video_id"]
LABEL = "watch_ratio"
MATRIX_COLUMNS = [*KEYS, LABEL, "video_duration"]
PROTOCOL = "big-to-small-v1"


class _OptionalBatchFields(TypedDict, total=False):
    pair_weight: torch.Tensor


class RecommendationBatch(_OptionalBatchFields):
    """Features/labels are [B].

    Raw IDs identify rows independently of encoded IDs; optional pair weights are [B]."""

    feature: dict[str, torch.Tensor]
    label: torch.Tensor
    raw_user_id: torch.Tensor
    raw_video_id: torch.Tensor


@dataclass
class KuaiRecInstance:
    """One raw observed user-video event, before categorical encoding.

    Optional attributes preserve the public dataset vocabulary for inspection.
    Daily engagement fields are outcomes and excluded by the current preprocessor.
    """

    user_id: int
    video_id: int
    watch_ratio: float
    video_duration: float
    date: str | None = None
    timestamp: int | None = None
    play_duration: float | None = None
    gender: str | int | None = None
    age_range: str | int | None = None
    user_active_degree: str | int | None = None
    is_lowactive_period: str | int | None = None
    is_live_streamer: str | int | None = None
    is_video_author: str | int | None = None
    is_live_author: str | int | None = None
    is_photo_author: str | int | None = None
    follow_user_num: str | int | None = None
    follow_user_num_range: str | int | None = None
    fans_user_num: str | int | None = None
    fans_user_num_range: str | int | None = None
    friend_user_num: str | int | None = None
    friend_user_num_range: str | int | None = None
    register_days: str | int | None = None
    register_days_range: str | int | None = None
    user_onehot_feat0: str | int | None = None
    user_onehot_feat1: str | int | None = None
    user_onehot_feat2: str | int | None = None
    user_onehot_feat3: str | int | None = None
    user_onehot_feat4: str | int | None = None
    user_onehot_feat5: str | int | None = None
    user_onehot_feat6: str | int | None = None
    user_onehot_feat7: str | int | None = None
    user_onehot_feat8: str | int | None = None
    user_onehot_feat9: str | int | None = None
    user_onehot_feat10: str | int | None = None
    user_onehot_feat11: str | int | None = None
    user_onehot_feat12: str | int | None = None
    user_onehot_feat13: str | int | None = None
    user_onehot_feat14: str | int | None = None
    user_onehot_feat15: str | int | None = None
    user_onehot_feat16: str | int | None = None
    user_onehot_feat17: str | int | None = None
    phone_brand: str | int | None = None
    phone_model: str | int | None = None
    mod_price: str | int | None = None
    fre_country: str | int | None = None
    fre_country_region: str | int | None = None
    fre_province: str | int | None = None
    fre_city: str | int | None = None
    fre_city_level: str | int | None = None
    fre_community_type: str | int | None = None
    platform: str | int | None = None
    os_version: str | int | None = None
    app_version: str | int | None = None
    app_download_channel: str | int | None = None
    isp: str | int | None = None
    author_id: str | int | None = None
    video_type: str | int | None = None
    upload_dt: str | int | None = None
    upload_type: str | int | None = None
    visible_status: str | int | None = None
    video_width: str | int | None = None
    video_height: str | int | None = None
    music_id: str | int | None = None
    video_tag_id: str | int | None = None
    video_daily_show_cnt: str | int | None = None
    video_daily_show_user_num: str | int | None = None
    video_daily_play_cnt: str | int | None = None
    video_daily_play_user_num: str | int | None = None
    video_daily_play_duration: str | int | None = None
    video_daily_complete_play_cnt: str | int | None = None
    video_daily_complete_play_user_num: str | int | None = None
    video_daily_valid_play_cnt: str | int | None = None
    video_daily_valid_play_user_num: str | int | None = None
    video_daily_long_time_play_cnt: str | int | None = None
    video_daily_long_time_play_user_num: str | int | None = None
    video_daily_short_time_play_cnt: str | int | None = None
    video_daily_short_time_play_user_num: str | int | None = None
    video_daily_play_progress: float | None = None
    video_daily_comment_stay_duration: str | int | None = None
    video_daily_like_cnt: str | int | None = None
    video_daily_like_user_num: str | int | None = None
    video_daily_click_like_cnt: str | int | None = None
    video_daily_double_click_cnt: str | int | None = None
    video_daily_cancel_like_cnt: str | int | None = None
    video_daily_cancel_like_user_num: str | int | None = None
    video_daily_comment_cnt: str | int | None = None
    video_daily_comment_user_num: str | int | None = None
    video_daily_direct_comment_cnt: str | int | None = None
    video_daily_reply_comment_cnt: str | int | None = None
    video_daily_delete_comment_cnt: str | int | None = None
    video_daily_delete_comment_user_num: str | int | None = None
    video_daily_comment_like_cnt: str | int | None = None
    video_daily_comment_like_user_num: str | int | None = None
    video_daily_follow_cnt: str | int | None = None
    video_daily_follow_user_num: str | int | None = None
    video_daily_cancel_follow_cnt: str | int | None = None
    video_daily_cancel_follow_user_num: str | int | None = None
    video_daily_share_cnt: str | int | None = None
    video_daily_share_user_num: str | int | None = None
    video_daily_download_cnt: str | int | None = None
    video_daily_download_user_num: str | int | None = None
    video_daily_report_cnt: str | int | None = None
    video_daily_report_user_num: str | int | None = None
    video_daily_reduce_similar_cnt: str | int | None = None
    video_daily_reduce_similar_user_num: str | int | None = None
    video_daily_collect_cnt: str | int | None = None
    video_daily_collect_user_num: str | int | None = None
    video_daily_cancel_collect_cnt: str | int | None = None
    video_daily_cancel_collect_user_num: str | int | None = None

    def __post_init__(self) -> None:
        for name in KEYS:
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be a nonnegative int64 ID")
        if not math.isfinite(self.watch_ratio):
            raise ValueError("watch_ratio must be finite")
        if not math.isfinite(self.video_duration):
            raise ValueError("video_duration must be finite")

    @classmethod
    def validate_frame(cls, frame: pd.DataFrame, source: Path) -> None:
        """Validate columnwise without allocating one Python object per event."""
        missing = set(MATRIX_COLUMNS) - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f"{source.name}: empty matrix or missing columns {sorted(missing)}")
        validate_ids(frame, KEYS, source)
        labels = frame[LABEL]
        if not pd.api.types.is_numeric_dtype(labels):
            raise ValueError(f"{source.name}: watch_ratio must be numeric")
        if not np.isfinite(labels).all() or (labels.abs() > np.finfo(np.float32).max).any():
            raise ValueError(f"{source.name}: labels must be finite float32 values")


def validate_ids(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    for name in columns:
        values = frame[name]
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{source.name}: {name} must contain numeric IDs")
        invalid = (
            ~np.isfinite(values) | (values < 0) | (values != np.floor(values)) | (values >= 2**63)
        )
        if invalid.any():
            raise ValueError(f"{source.name}: {name} must contain nonnegative int64 IDs")
