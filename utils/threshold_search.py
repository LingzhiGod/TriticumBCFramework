"""
Threshold Search Layer
自动搜索分类概率阈值以最大化综合得分。
"""

import numpy as np
from typing import Tuple, Dict, Optional

from .logger import DummyLogger
from .metrics import evaluate_metrics, compute_score, format_metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    formula: str = "0.7*acc + 0.3*f1",
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
    auto_adjust: bool = True,
    logger=DummyLogger()
) -> Tuple[float, float, Dict[str, float]]:
    """
    搜索最优分类阈值以最大化综合评分。
    参数:
        y_true : np.ndarray, 实际标签 (0/1)
        y_prob : np.ndarray, 模型输出概率
        formula : str, 综合得分公式
        start, end, step : float, 搜索区间与步长
        auto_adjust : bool, 是否自动局部细化搜索
        logger : 可选日志对象
    返回:
        best_thr : float, 最优阈值
        best_score : float, 对应综合得分
        best_metrics : dict, 对应的基础指标
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    thresholds = np.arange(start, end + 1e-9, step)
    best_thr, best_score, best_metrics = 0.5, -1, {}

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        metrics = evaluate_metrics(y_true, y_pred)
        score = compute_score(metrics, formula)

        if score > best_score:
            best_thr, best_score, best_metrics = thr, score, metrics

    if logger:
        logger.info(f"[THR Search] Coarse best = {best_thr:.3f}  "
                    f"Score={best_score:.5f}  {format_metrics(best_metrics)}")

    # 可选细化搜索
    if auto_adjust:
        fine_start = max(start, best_thr - step)
        fine_end = min(end, best_thr + step)
        fine_step = step / 10
        thresholds_fine = np.arange(fine_start, fine_end + 1e-9, fine_step)
        for thr in thresholds_fine:
            y_pred = (y_prob >= thr).astype(int)
            metrics = evaluate_metrics(y_true, y_pred)
            score = compute_score(metrics, formula)
            if score > best_score:
                best_thr, best_score, best_metrics = thr, score, metrics

        if logger:
            logger.info(f"[THR Search] Refined best = {best_thr:.4f}  "
                        f"Score={best_score:.5f}  {format_metrics(best_metrics)}")

    return best_thr, best_score, best_metrics


def threshold_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    formula: str = "0.7*acc + 0.3*f1",
    logger=DummyLogger()
) -> Dict[str, float]:
    """
    综合阈值优化与评测接口（用于 run.py）
    返回标准格式：
      {"thr":..., "score":..., "metrics": {...}}
    """
    best_thr, best_score, best_metrics = find_best_threshold(
        y_true=y_true,
        y_prob=y_prob,
        formula=formula,
        logger=logger
    )

    report = {
        "thr": best_thr,
        "score": best_score,
        "metrics": best_metrics
    }

    if logger:
        logger.info(f"[Eval] Best Thr={best_thr:.4f}  "
                    f"Score={best_score:.5f}  {format_metrics(best_metrics)}")

    return report
