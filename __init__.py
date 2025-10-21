"""
TriticumBCFramework
-------------------
A modular binary-classification training and feature-engineering framework.
Version: 0.1.0
Author: trdyun
"""

import sys
import os

# ------------------------------------
# 自动路径修正（允许包内直接运行）
# ------------------------------------
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.dirname(_pkg_dir)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# ------------------------------------
# 元信息
# ------------------------------------
__version__ = "0.1.0"
__author__ = "trdyun"
__description__ = "A modular binary-classification training and feature-engineering framework."

# ------------------------------------
# 核心模块导出
# ------------------------------------
try:
    from .utils.config_class import Config
    from .utils.logger import setup_logger, DummyLogger
except Exception as e:
    print(f"[TBCF] Warning: utils modules not fully loaded ({e})")

try:
    from .feature_engineering.builder import build_features
    from .feature_engineering.core import apply_date_features
except Exception as e:
    print(f"[TBCF] Warning: feature_engineering modules not fully loaded ({e})")

__all__ = [
    "Config",
    "setup_logger",
    "DummyLogger",
    "build_features",
    "apply_date_features",
]

# ------------------------------------
# 可选初始化提示
# ------------------------------------
if os.environ.get("TBCF_SILENT", "0") != "1":
    print(f"[TBCF] TriticumBCFramework v{__version__} initialized.")
