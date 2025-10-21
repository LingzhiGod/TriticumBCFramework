"""
TriticumBCFramework - Core Features (Stateful Unified Version)
--------------------------------------------------------------
提供基础的日期类特征构建模块。

接口标准：
    def feature_func(df, mode, params, shared_state, logger):
        return df, shared_state
"""

import pandas as pd
from datetime import datetime
from .feature_registry import register_feature
from utils.logger import DummyLogger


@register_feature("date_features")
def apply_date_features(df, mode, params=None, shared_state=None, logger=DummyLogger()):
    """
    日期类特征：将日期列转换为注册年、注册月、距参考日期天数。

    参数：
        params : dict
            {
              "date_col": "registration_date",
              "reference_date": "2020-12-31"
            }
        mode : str
            "train" 或 "test"
        shared_state : dict | None
            - train 阶段：可返回 {"ref_date": ...}
            - test 阶段：复用相同 reference_date
    """
    params = params or {}
    date_col = params.get("date_col", "registration_date")

    # 获取参考日期（优先使用 shared_state 中保存的）
    ref_date_str = None
    if shared_state and "ref_date" in shared_state:
        ref_date_str = shared_state["ref_date"]
    else:
        ref_date_str = params.get("reference_date", "2020-12-31")

    ref_date = pd.to_datetime(ref_date_str)

    if date_col not in df.columns:
        if logger:
            logger.warning(f"[Feature:date_features] Missing column: {date_col}")
        return df, shared_state

    # 转换为日期类型
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 衍生特征
    df["reg_year"] = df[date_col].dt.year
    df["reg_month"] = df[date_col].dt.month
    df["reg_days_since_ref"] = (df[date_col] - ref_date).dt.days

    if logger:
        logger.info(f"[Feature:date_features] Applied to column '{date_col}' (mode={mode})")

    # 返回更新后的状态（仅训练阶段需要保存 reference_date）
    if mode == "train":
        shared_state = shared_state or {}
        shared_state["ref_date"] = ref_date_str

    return df, shared_state
