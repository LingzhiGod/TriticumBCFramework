"""
TriticumBCFramework - Config System
-----------------------------------
增强版配置类：
- 支持 from_json() / from_dict() / merge_cli_args()
- 同时支持属性访问 (cfg.ensemble.mode) 与字典访问 (cfg["ensemble"]["mode"])
- 提供 dict 兼容接口：.get(), .keys(), .items(), .update()
- 支持快照保存 save_snapshot()
"""

import json
import os
from typing import Any, Dict
from .validator import validate_config


class Config:
    """统一配置对象，兼具 dict 与对象属性访问能力。"""

    def __init__(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError(f"[Config] Expected dict, got {type(data)}")
        # 验证与深拷贝配置
        self._data = validate_config(data)

    # ---------- 构造方法 ----------
    @classmethod
    def from_json(cls, path: str):
        """从 JSON 文件加载配置"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Config] {path} not found.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典初始化配置"""
        return cls(data)

    def merge_cli_args(self, args: Dict[str, Any]):
        """允许命令行参数覆盖配置，例如 --global.log_level=DEBUG"""
        for key, val in args.items():
            if val is None:
                continue
            section, _, field = key.partition(".")
            if section in self._data and isinstance(self._data[section], dict):
                if field in self._data[section]:
                    self._data[section][field] = val
        return self

    # ---------- 快照 ----------
    def save_snapshot(self, path: str):
        """保存当前配置快照"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

    # ---------- 属性访问 ----------
    def __getattr__(self, name: str):
        if name in self._data:
            value = self._data[name]
            # 递归包装成 Config
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(f"No config field named '{name}'")

    # ---------- 下标访问 ----------
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    # ---------- dict 兼容接口 ----------
    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, other: Dict[str, Any]):
        self._data.update(other)

    # ---------- 实用工具 ----------
    def as_dict(self):
        """返回原始字典"""
        return self._data

    def to_json(self, path: str):
        """保存为 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

    # ---------- 打印与调试 ----------
    def __repr__(self):
        keys = ", ".join(self._data.keys())
        return f"<Config sections=[{keys}]>"
