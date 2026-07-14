"""Shared paths and small helpers for the phishing email project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

STANDARDIZED_DATA_PATH = PROCESSED_DATA_DIR / "phishing_email_standardized.csv"
LEGACY_TEXT_MODEL_PATH = MODELS_DIR / "phishing_logreg_tfidf.pkl"
FINAL_MODEL_PATH = MODELS_DIR / "phishing_logreg_text_metadata.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_project_dirs() -> None:
    """Create the expected local output directories."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def format_percentage(value: float, decimals: int = 3) -> str:
    """Format a 0-1 metric value as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
