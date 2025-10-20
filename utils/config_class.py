"""
- from_json() / from_dict() / merge_cli_args()
- 属性访问 (cfg.ensemble.mode)
- 快照保存 save_snapshot()
"""

import json
import os
from typing import Any, Dict
from .validator import validate_config

class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = validate_config(data)

    # ---------- 构造方法 ----------
    @classmethod
    def from_json(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Config] {path} not found.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_cli_args(self, args: Dict[str, Any]):
        """允许命令行参数覆盖配置"""
        for key, val in args.items():
            if val is None:
                continue
            section, _, field = key.partition(".")
            if section in self._data and field in self._data[section]:
                self._data[section][field] = val
        return self

    # ---------- 快照 ----------
    def save_snapshot(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

    # ---------- 属性访问 ----------
    def __getattr__(self, name: str):
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"No config field named '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def as_dict(self):
        return self._data

    def __repr__(self):
        return f"<Config sections={list(self._data.keys())}>"