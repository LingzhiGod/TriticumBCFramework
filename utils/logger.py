import logging
import os
import sys
from datetime import datetime

class LoggerManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, cfg=None, log_dir="output/logs"):
        self.cfg = cfg
        self.log_dir = log_dir
        self.logger = None
        os.makedirs(self.log_dir, exist_ok=True)

    def setup(self):
        """初始化日志器"""
        if self.logger:
            return self.logger

        log_level = (self.cfg.get("global", {}).get("log_level", "INFO")
                     if self.cfg else "INFO")
        ts_format = (self.cfg.get("global", {}).get("timestamp_format", "%Y%m%d_%H%M%S")
                     if self.cfg else "%Y%m%d_%H%M%S")

        timestamp = datetime.now().strftime(ts_format)
        log_file = os.path.join(self.log_dir, f"tbcf_{timestamp}.log")

        logger = logging.getLogger("TBCF")
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.propagate = False

        # 控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(message)s"
        ))

        # 文件 handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.info("========================================")
        logger.info("   TriticumBCFramework Logging Started  ")
        logger.info("========================================")
        logger.info(f"Log Level: {log_level}")
        logger.info(f"Log File : {log_file}")

        self.logger = logger
        return logger


# Global Logger Function
def setup_logger(cfg=None, log_dir="output/logs"):
    """初始化全局 logger"""
    return LoggerManager(cfg, log_dir).setup()