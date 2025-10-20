from utils.logger import setup_logger
from utils.config_loader import load_config

def main():
    cfg = load_config("input/config.json")
    logger = setup_logger(cfg)
    logger.info("Configuration loaded successfully.")
    logger.info(f"Ensemble mode: {cfg['ensemble']['mode']}")