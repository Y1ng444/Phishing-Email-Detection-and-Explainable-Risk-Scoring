"""Prediction and risk scoring for optimized sklearn text pipelines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .text_cleaning import count_urls
from .utils import MODELS_DIR


DEFAULT_PIPELINE_NAME = "logistic_regression_pipeline.pkl"


def load_pipeline(filename: str = DEFAULT_PIPELINE_NAME):
    """Load a trained sklearn Pipeline from the models directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {path}")
    return joblib.load(path)


def load_model(model_name: str = DEFAULT_PIPELINE_NAME):
    """Backward-compatible alias for loading the optimized pipeline."""
    return load_pipeline(model_name)


def _score_email(email_text: str, pipeline: Any) -> tuple[int, float, str]:
    """Return prediction, normalized phishing score, and score type."""
    prediction = int(pipeline.predict([email_text])[0])

    if hasattr(pipeline, "predict_proba"):
        score = float(pipeline.predict_proba([email_text])[:, 1][0])
        return prediction, score, "probability"

    if hasattr(pipeline, "decision_function"):
        raw_score = float(pipeline.decision_function([email_text])[0])
        score = float(1.0 / (1.0 + np.exp(-raw_score)))
        return prediction, score, "normalized_decision_score"

    return prediction, float(prediction), "prediction"


def get_risk_level_info(score: float) -> dict[str, str | int]:
    """Convert a 0-1 phishing score to a 0-100 risk score and level."""
    risk_score = int(round(max(0.0, min(1.0, score)) * 100))

    if risk_score < 34:
        return {
            "risk_score": risk_score,
            "risk_level": "Low",
            "description": "Likely legitimate.",
        }
    if risk_score < 67:
        return {
            "risk_score": risk_score,
            "risk_level": "Medium",
            "description": "Suspicious enough to review.",
        }
    return {
        "risk_score": risk_score,
        "risk_level": "High",
        "description": "Likely phishing.",
    }


def predict_email(email_text: str, pipeline: Any | None = None, vectorizer: Any | None = None) -> dict[str, Any]:
    """Predict whether one email is ham or phishing.

    The unused ``vectorizer`` argument is kept for compatibility with older
    calls that passed a separate model and vectorizer.
    """
    del vectorizer

    if not email_text or not str(email_text).strip():
        raise ValueError("Email text cannot be empty.")

    if pipeline is None:
        pipeline = load_pipeline()

    prediction, score, score_type = _score_email(email_text, pipeline)
    risk = get_risk_level_info(score)

    return {
        "prediction": "Phishing" if prediction == 1 else "Ham",
        "predicted_label": prediction,
        "phishing_score": score,
        "probability": score,
        "score_type": score_type,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "description": risk["description"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "email_length": len(email_text),
        "word_count": len(email_text.split()),
        "url_count": count_urls(email_text),
    }


def batch_predict(email_texts: list[str], pipeline: Any | None = None) -> list[dict[str, Any] | None]:
    """Predict a list of email texts."""
    if pipeline is None:
        pipeline = load_pipeline()

    results: list[dict[str, Any] | None] = []
    for email_text in email_texts:
        try:
            results.append(predict_email(email_text, pipeline=pipeline))
        except Exception:
            results.append(None)
    return results


def model_exists(filename: str = DEFAULT_PIPELINE_NAME) -> bool:
    """Return True when the expected model artifact exists."""
    return Path(MODELS_DIR / filename).exists()
