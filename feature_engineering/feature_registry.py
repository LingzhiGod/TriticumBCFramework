"""
Feature Registry
内置特征模块注册中心。
支持 register_feature() 装饰器与外部注册机制。
"""

from typing import Callable, Dict

REGISTRY: Dict[str, Callable] = {}

def register_feature(name: str):
    """装饰器：用于注册特征构建函数"""
    def decorator(func: Callable):
        REGISTRY[name] = func
        return func
    return decorator


def list_features() -> list:
    """返回当前已注册的特征模块名称"""
    return list(REGISTRY.keys())
