"""
Metrics Layer
提供统一的二分类评测指标与动态综合评分计算。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict, Optional
import re

from utils.logger import DummyLogger


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算基础二分类评测指标。
    参数:
        y_true : 真实标签 (0/1)
        y_pred : 预测标签 (0/1)
    返回:
        dict : {"acc":..., "f1":..., "p":..., "r":...}
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return {"acc": acc, "f1": f1, "p": p, "r": r}


def compute_score(metrics: Dict[str, float],
                  formula: str = "0.7*acc + 0.3*f1") -> float:
    """
    根据配置中的公式计算综合得分。
    参数:
        metrics : evaluate_metrics 的输出结果
        formula : 字符串公式，例如 "0.7*acc + 0.3*f1"
    返回:
        float : 综合得分
    """
    # 允许的变量
    safe_env = {k: float(v) for k, v in metrics.items()}

    # 限制 eval 环境，禁用内置函数
    try:
        score = eval(formula, {"__builtins__": None}, safe_env)
    except Exception as e:
        raise ValueError(f"[Metrics] Invalid score formula: {formula}\nError: {e}")

    return float(score)


def format_metrics(metrics: Dict[str, float], score: Optional[float] = None) -> str:
    """
    格式化输出指标，便于日志记录。
    """
    line = (
        f"ACC={metrics['acc']:.5f}  "
        f"F1={metrics['f1']:.5f}  "
        f"P={metrics['p']:.5f}  "
        f"R={metrics['r']:.5f}"
    )
    if score is not None:
        line += f"  SCORE={score:.5f}"
    return line


def evaluate_with_formula(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          formula: str = "0.7*acc + 0.3*f1",
                          logger=DummyLogger()) -> Dict[str, float]:
    """
    综合评测函数：计算基础指标 + 综合得分。
    参数:
        y_true, y_pred : np.ndarray
        formula : 评分公式
        logger : 可选日志器
    返回:
        {"acc":..., "f1":..., "p":..., "r":..., "score":...}
    """
    metrics = evaluate_metrics(y_true, y_pred)
    score = compute_score(metrics, formula)
    result = {**metrics, "score": score}

    if logger:
        logger.info("[Eval] " + format_metrics(metrics, score))
        logger.info(f"[Eval] Formula: {formula}")

    return result