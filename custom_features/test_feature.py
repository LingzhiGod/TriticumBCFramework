from TriticumBCFramework.feature_engineering.feature_registry import register_feature
from TriticumBCFramework.utils.logger import DummyLogger

@register_feature("my_feature")
def my_feature(df, mode, params=None, shared_state=None, logger=DummyLogger()):
    logger.info("Hello World from My Feature!")
    return df, shared_state