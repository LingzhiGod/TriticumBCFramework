"""
TriticumBCFramework - Integration Runner
----------------------------------------
框架主运行入口（包内路径调用版）：
- 验证 Config、Logger、Feature Builder、Core 模块是否正常工作
- 输出特征构建结果与日志文件

运行方式（推荐）：
    python -m TriticumBCFramework.run
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# ✅ 包路径修正（允许从项目根或包内直接运行）
# ---------------------------------------------------------
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.dirname(_pkg_dir)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# ---------------------------------------------------------
# ✅ 包内导入
# ---------------------------------------------------------
from TriticumBCFramework.utils.config_class import Config
from TriticumBCFramework.utils.logger import setup_logger
from TriticumBCFramework.feature_engineering.builder import build_features


def main():
    # ----------------------------------
    # 1️⃣ 初始化配置
    # ----------------------------------
    cfg_dict = {
        "global": {
            "log_level": "DEBUG",
            "timestamp_format": "%Y%m%d_%H%M%S"
        },
        "feature_engineering": {
            "registry": ["date_features", "my_feature"],
            "params": {
                "date_features": {"date_col": "registration_date", "reference_date": "2020-12-31"}
            },
            "output_mode": "typed",
            "save_feature_list": True,
            "ignore_cols": ["registration_date","user_id", "is_positive"]
        }
    }

    cfg = Config(cfg_dict)
    logger = setup_logger(cfg)
    logger.info("=== TriticumBCFramework Integration Test Start ===")

    # ----------------------------------
    # 2️⃣ 构造虚拟训练 / 测试数据
    # ----------------------------------
    n = 8
    train_df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "registration_date": pd.date_range("2021-01-01", periods=n, freq="15D"),
        "gender": np.random.choice(["1", "2"], size=n),
        "is_positive": np.random.choice([0, 1], size=n)
    })
    test_df = pd.DataFrame({
        "user_id": [f"t{i}" for i in range(4)],
        "registration_date": pd.date_range("2022-01-01", periods=4, freq="30D"),
        "gender": np.random.choice(["1", "2"], size=4)
    })

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ----------------------------------
    # 3️⃣ 执行特征工程（训练阶段）
    # ----------------------------------
    logger.info("[Stage 1] Feature Engineering - Train Mode")
    train_fe, used_cols, num_cols, cat_cols, fe_state = build_features(
        train_df,
        cfg.feature_engineering.as_dict(),
        mode="train",
        logger=logger
    )

    logger.info(f"[Train] Used ({len(used_cols)}): {used_cols}")
    logger.info(f"[Train] Numeric ({len(num_cols)}): {num_cols}")
    logger.info(f"[Train] Categorical ({len(cat_cols)}): {cat_cols}")
    logger.info(f"[Train] Shared State: {list(fe_state.keys())}")

    # ----------------------------------
    # 4️⃣ 执行特征工程（测试阶段）
    # ----------------------------------
    logger.info("[Stage 2] Feature Engineering - Test Mode")
    test_fe, used_cols_t, num_cols_t, cat_cols_t, fe_state_t = build_features(
        test_df,
        cfg.feature_engineering.as_dict(),
        mode="test",
        shared_state=fe_state,
        logger=logger
    )

    logger.info(f"[Test] Used ({len(used_cols_t)}): {used_cols_t}")
    logger.info(f"[Test] Head:\n{test_fe.head()}")

    # ----------------------------------
    # 5️⃣ 检查输出
    # ----------------------------------
    if os.path.exists("output/feature_list.txt"):
        logger.info("[OK] feature_list.txt generated ✅")
    if os.path.exists("output/feature_state.pkl"):
        logger.info("[OK] feature_state.pkl generated ✅")

    logger.info("=== TriticumBCFramework Integration Test Finished ===")


if __name__ == "__main__":
    main()
