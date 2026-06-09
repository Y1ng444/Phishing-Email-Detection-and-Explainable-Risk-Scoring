"""Utility functions for phishing detection project."""

import logging
import sys
from pathlib import Path
from typing import Optional
import random
import numpy as np

# Directory structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
LOGISTIC_REGRESSION_MAX_ITER = 1000

# Risk level thresholds (in percentages)
RISK_LEVELS = {
    "Low": (0, 30),
    "Medium": (31, 60),
    "High": (61, 80),
    "Critical": (81, 100),
}

# Standardized log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level.

    Args:
        probability: Probability value (0-1)

    Returns:
        Risk level string
    """
    risk_score = probability * 100

    for level, (min_score, max_score) in RISK_LEVELS.items():
        if min_score <= risk_score <= max_score:
            return level

    return "Critical"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float as percentage string.

    Args:
        value: Float value to format
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


logger = setup_logger(__name__)
