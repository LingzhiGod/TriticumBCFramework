import copy

_DEFAULT = {
    "global": {
        "seed": 42,
        "log_level": "INFO",
        "overwrite": False,
        "timestamp_format": "%Y%m%d_%H%M%S"
    },
    "ensemble": {
        "mode": "weighted",
        "default_weights": {"lgb": 0.5, "xgb": 0.3, "cat": 0.2},
        "auto_weight_tune": True
    },
    "evaluation": {
        "metric_formula": "0.7*acc + 0.3*f1",
        "primary_metric": "f1"
    },
    "threshold_search": {
        "start": 0.05,
        "end": 0.95,
        "step": 0.01
    },
    "feature_engineering": {
        "registry": ["telecom_features", "nonlinear_features"],
        "save_feature_list": True
    },
    "tuning": {
        "enable_lgb": True,
        "enable_xgb": True,
        "enable_cat": False,
        "n_trials": 50
    }
}

def merge_dict(default, override):
    for k, v in default.items():
        if k not in override:
            override[k] = copy.deepcopy(v)
        elif isinstance(v, dict):
            merge_dict(v, override[k])
    return override

def validate_config(cfg: dict) -> dict:
    """递归补全默认值"""
    cfg = merge_dict(copy.deepcopy(_DEFAULT), cfg)

    if cfg["ensemble"]["mode"] not in ["weighted", "stacking"]:
        raise ValueError("ensemble.mode must be 'weighted' or 'stacking'")

    return cfg
