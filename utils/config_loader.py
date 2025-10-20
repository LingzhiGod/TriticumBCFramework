"""
统一入口，可从文件或字典加载
"""

from .config_class import Config

def load_config(source=None):
    """
    统一加载入口：
      - 若传入 None，则默认读取 input/config.json
      - 若传入 str，则视为路径
      - 若传入 dict，则直接解析
    """
    if source is None:
        source = "input/config.json"
    if isinstance(source, str):
        cfg = Config.from_json(source)
    elif isinstance(source, dict):
        cfg = Config.from_dict(source)
    else:
        raise TypeError("Config source must be None, str, or dict")
    return cfg
