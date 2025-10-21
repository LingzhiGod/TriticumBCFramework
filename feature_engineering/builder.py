
"""
Feature Builder
----------------------------------------------------------------
- 单 DataFrame 输入：支持 train/test 统一调用
- shared_state：在训练与推理阶段共享统计参数（如编码器）
- 参数化：从 config["feature_engineering"]["params"] 中读取模块参数
- 输出模式：typed/raw 两种；typed 会自动区分数值&类别列并统一编码
- 动态扩展：支持 external_modules 动态导入并注册特征
返回：df, used_cols, num_cols, cat_cols, shared_state
"""

from __future__ import annotations
import importlib
import os
from typing import Dict, List, Tuple, Optional
from utils.logger import DummyLogger

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .feature_registry import REGISTRY, list_features


# -------------------------------
# Utilities
# -------------------------------
def import_external_modules(modules: List[str], logger=DummyLogger()) -> None:
    """动态导入外部特征模块；若模块内实现 register_features(registry) 会被调用。"""
    for mod in modules or []:
        try:
            imported = importlib.import_module(mod)
            if hasattr(imported, "register_features"):
                imported.register_features(REGISTRY)
            if logger:
                logger.info(f"[Feature] External module loaded: {mod}")
        except Exception as e:
            if logger:
                logger.warning(f"[Feature] Failed to import {mod}: {e}")


def detect_feature_types(df: pd.DataFrame, ignore: Optional[List[str]] = None, logger=DummyLogger()) -> Tuple[List[str], List[str]]:
    """自动区分数值列与类别列（基于 dtype）；ignore 中的列将被排除。"""
    ignore = set(ignore or [])
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in ignore:
            continue
        cname = c.lower()
        dtype = str(df[c].dtype)
        if cname.endswith("_cat") or cname.endswith("_bin") or cname.endswith("_le"):
            cat_cols.append(c)
        elif cname.endswith("_num"):
            num_cols.append(c)
        elif dtype.startswith(("int", "float")):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    if logger:
        logger.debug(f"[Feature] Detected {len(num_cols)} numeric, {len(cat_cols)} categorical")
    return num_cols, cat_cols


def _encode_categoricals(
    df: pd.DataFrame,
    mode: str,
    shared_state: Optional[Dict] = None,
    include_cols: Optional[List[str]] = None,
    ignore_cols: Optional[List[str]] = None,
    logger=DummyLogger()
) -> Tuple[pd.DataFrame, Dict]:
    """
    统一的类别编码器：训练阶段拟合，推理阶段复用。
    - 训练：拟合 LabelEncoder 并保存到 shared_state["_encoders"]
    - 推理：使用已保存的 encoder 进行 transform；未见值标记为 'UNK' 并扩展 classes_
    include_cols：若为 None，则对非数值/非目标/非id列进行编码；否则仅对指定列编码。
    返回：编码后的 df 与更新后的 shared_state
    """
    if shared_state is None:
        shared_state = {}

    enc_store: Dict[str, LabelEncoder] = shared_state.get("_encoders", {})

    # 推断需要编码的列
    if include_cols is None:
        # 统计 dtype 为 object/category 的列
        cand = [c for c in df.columns if (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
    else:
        cand = list(include_cols)

    for col in cand:
        if col in ignore_cols:
            continue
        series = df[col].astype(str).fillna("nan")
        if mode == "train":
            le = LabelEncoder()
            # 预留一个 'UNK' 类别以容纳未见值
            values = pd.Index(series.unique()).astype(str)
            if "UNK" not in values:
                values = values.append(pd.Index(["UNK"]))
            le.fit(values)
            df[col + "_le"] = le.transform(series.where(series.isin(le.classes_), "UNK"))
            enc_store[col] = le
            if logger:
                logger.debug(f"[Encode] [train] {col} -> {col}_le (classes={len(le.classes_)})")
        else:
            le = enc_store.get(col)
            if le is None:
                if logger:
                    logger.warning(f"[Encode] [test] encoder for '{col}' not found; skip encoding.")
                continue
            # 对未见值进行扩展至 'UNK'
            cls = pd.Index(le.classes_)
            if "UNK" not in cls:
                cls = cls.append(pd.Index(["UNK"]))
                le.classes_ = cls.values
            mapped = series.where(series.isin(le.classes_), "UNK")
            df[col + "_le"] = le.transform(mapped)
            if logger:
                logger.debug(f"[Encode] [test] {col} -> {col}_le")

    shared_state["_encoders"] = enc_store
    return df, shared_state


# -------------------------------
# Main Builder
# -------------------------------
def build_features(
    df: pd.DataFrame,
    fe_cfg: Dict,
    mode: str = "train",
    shared_state: Optional[Dict] = None,
    logger=DummyLogger()
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], Dict]:
    """
    特征构建主流程（单 DataFrame 输入 + 状态共享）。
    参数：
      - fe_cfg:
          {
            "registry": ["date_features", ...],
            "params": { "date_features": {...} },
            "external_modules": ["my_pkg.fe_ext"],
            "output_mode": "typed" | "raw",
            "save_feature_list": true/false,
            "ignore_cols": ["user_id", "is_positive"]
          }
      - mode: "train" 或 "test"
      - shared_state: 训练阶段产生的状态（编码器/分桶/统计量等），用于推理阶段复用
    返回：df, used_cols, num_cols, cat_cols, shared_state
    """
    fe_cfg = fe_cfg or {}
    registry = fe_cfg.get("registry", [])
    params_all = fe_cfg.get("params", {}) or {}
    external_modules = fe_cfg.get("external_modules", []) or []
    output_mode = fe_cfg.get("output_mode", "typed")  # "typed" 或 "raw"
    save_feature_list = fe_cfg.get("save_feature_list", True)
    ignore_cols = fe_cfg.get("ignore_cols", [])

    if shared_state is None:
        shared_state = {}

    # 1) 外部模块加载
    if external_modules:
        import_external_modules(external_modules, logger)

    if logger:
        logger.info(f"[Feature] Mode={mode}  OutputMode={output_mode}")
        logger.info(f"[Feature] Active modules: {registry}")
        if external_modules:
            logger.info(f"[Feature] External: {external_modules}")
        # 可选：打印当前可用注册表
        # logger.info(f"[Feature] Available registry: {list_features()}")

    # 2) 顺序执行特征模块（单 DataFrame 接口 + 状态共享）
    for name in registry:
        func = REGISTRY.get(name)
        if func is None:
            if logger:
                logger.warning(f"[Feature] Module '{name}' not found, skip.")
            continue

        mod_params = params_all.get(name, {})
        mod_state_in = shared_state.get(name)  # train 阶段为 None，test 阶段为先前状态
        try:
            if logger:
                logger.info(f"[Feature] Apply: {name}  params={mod_params}")

            # 统一接口：df, mode, params, shared_state(name)
            df, mod_state_out = func(df, mode, mod_params, mod_state_in, logger)
            # 存储/更新模块状态
            shared_state[name] = mod_state_out
        except Exception as e:
            if logger:
                logger.warning(f"[Feature] Module '{name}' failed with error: {e}. Skipped.")
            continue

    # 3) 输出模式与统一编码
    if output_mode == "typed":
        # 先检测类型，再对类别列进行统一编码
        num_cols, cat_cols = detect_feature_types(df, ignore_cols, logger)
        # 对原生类别列进行统一编码，生成 *_le
        df, shared_state = _encode_categoricals(df, mode, shared_state, include_cols=cat_cols, ignore_cols=ignore_cols, logger=logger)
        # 刷新类型信息：将编码列视为 categorical used
        num_cols, cat_cols = detect_feature_types(df, ignore_cols, logger)
        # cat 列只保留 *_le 作为模型输入
        cat_le_cols = [c for c in cat_cols if c.endswith("_le")]
        used_cols = [c for c in num_cols + cat_le_cols if c not in ignore_cols]
        cat_cols = cat_le_cols  # 对外暴露的类别列就是编码后的列
    else:
        # raw：不做类型区分，也不做统一编码
        num_cols, cat_cols = [], []
        used_cols = [c for c in df.columns if c not in ignore_cols]

    # 4) 保存特征清单与状态快照（可选）
    if save_feature_list:
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", "feature_list.txt"), "w", encoding="utf-8") as f:
            f.write(f"[Mode] {mode}\n")
            f.write(f"[OutputMode] {output_mode}\n\n")
            f.write("[Used Columns]\n")
            for c in used_cols:
                f.write(c + "\n")
            if output_mode == "typed":
                f.write("\n[Numeric Columns]\n")
                for c in num_cols:
                    f.write(c + "\n")
                f.write("\n[Categorical Columns]\n")
                for c in cat_cols:
                    f.write(c + "\n")

        # 共享状态快照（便于复现与推理），注意：若含有不可序列化对象需自行处理
        try:
            import pickle
            with open(os.path.join("output", "feature_state.pkl"), "wb") as pf:
                pickle.dump(shared_state, pf)
        except Exception:
            # 退化到 json（仅保存简单对象）
            try:
                import json
                simple_state = {k: str(v) for k, v in shared_state.items()}
                with open(os.path.join("output", "feature_state.json"), "w", encoding="utf-8") as jf:
                    json.dump(simple_state, jf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if logger:
            logger.info(f"[Feature] Saved feature list/state to output/")

    return df, used_cols, num_cols, cat_cols, shared_state
