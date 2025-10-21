"""
TriticumBCFramework Feature Registry
------------------------------------
提供统一的特征模块注册与自动加载机制。

特性：
1. 统一的 @register_feature 装饰器；
2. 自动扫描包内特征模块 (TriticumBCFramework/feature_engineering/*.py)；
3. 自动扫描外部特征模块 (custom_features/*.py)；
4. 打印友好日志，支持自定义 logger。
"""

import os
import importlib
import importlib.util
import pkgutil
from typing import Callable, Dict, Optional

# ---------------------------------------------------------
# 全局注册表
# ---------------------------------------------------------
REGISTRY: Dict[str, Callable] = {}

# ---------------------------------------------------------
# 注册装饰器
# ---------------------------------------------------------
def register_feature(name: str):
    """用于注册特征模块的装饰器"""
    def decorator(func: Callable):
        if name in REGISTRY:
            print(f"[TBCF][WARN] Duplicate feature registration for '{name}', overwriting.")
        REGISTRY[name] = func
        print(f"[TBCF][REG] Registered feature: {name}")
        return func
    return decorator

# ---------------------------------------------------------
# 自动扫描内部模块 (TriticumBCFramework/feature_engineering/)
# ---------------------------------------------------------
def _auto_import_internal_features(base_pkg: str = __name__.rsplit(".", 1)[0], logger=None):
    """自动导入包内 feature_engineering 目录下的所有模块"""
    try:
        package = importlib.import_module(base_pkg)
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            full_name = f"{base_pkg}.{module_name}"
            if module_name not in ("feature_registry", "__init__"):
                importlib.import_module(full_name)
                if logger:
                    logger.debug(f"[FeatureRegistry] Loaded internal module: {full_name}")
    except Exception as e:
        if logger:
            logger.warning(f"[FeatureRegistry] Internal feature auto-import failed: {e}")
        else:
            print(f"[TBCF][WARN] Internal auto-import failed: {e}")

# ---------------------------------------------------------
# 自动扫描外部模块 (custom_features/)
# ---------------------------------------------------------
def _auto_import_external_features(base_dir: str = "custom_features", logger=None):
    """自动导入外部自定义特征模块目录"""
    if not os.path.exists(base_dir):
        return

    loaded = 0
    for fname in os.listdir(base_dir):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        fpath = os.path.join(base_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded += 1
            if logger:
                logger.debug(f"[FeatureRegistry] Loaded external module: {fname}")
        except Exception as e:
            msg = f"[FeatureRegistry] Failed to load external module {fname}: {e}"
            if logger:
                logger.warning(msg)
            else:
                print("[TBCF][WARN]", msg)
    if loaded and logger:
        logger.info(f"[FeatureRegistry] Loaded {loaded} external feature module(s).")

# ---------------------------------------------------------
# 统一加载接口
# ---------------------------------------------------------
def initialize_feature_registry(logger=None, auto_load_external: bool = True):
    """
    初始化注册表，自动扫描内部和外部模块。
    在 builder.build_features() 的最开头调用即可。
    """
    _auto_import_internal_features(logger=logger)
    if auto_load_external:
        _auto_import_external_features(logger=logger)

# ---------------------------------------------------------
# 查询接口
# ---------------------------------------------------------
def get_feature(name: str) -> Optional[Callable]:
    """获取已注册特征模块"""
    return REGISTRY.get(name)

def list_features() -> list[str]:
    """列出所有注册特征"""
    return sorted(list(REGISTRY.keys()))
