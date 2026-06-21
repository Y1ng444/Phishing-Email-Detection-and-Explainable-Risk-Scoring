"""Risk scoring for the Logistic Regression phishing model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .utils import FINAL_MODEL_PATH


def load_model(model_path: str | Path = FINAL_MODEL_PATH) -> Any:
    """Load the final Logistic Regression TF-IDF pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run python train_optimized.py first."
        )
    return joblib.load(path)


def risk_level_from_score(risk_score: float) -> str:
    """Map a 0-100 phishing score to a simple risk level."""
    if risk_score < 40:
        return "Low risk"
    if risk_score < 70:
        return "Medium risk"
    return "High risk"


def score_email(email_text: str, model: Any | None = None) -> dict[str, float | int | str]:
    """Return prediction, phishing probability, risk score, and risk level."""
    if not str(email_text).strip():
        raise ValueError("Email text cannot be empty.")

    if model is None:
        model = load_model()

    predicted_label = int(model.predict([email_text])[0])
    phishing_probability = float(model.predict_proba([email_text])[:, 1][0])
    risk_score = phishing_probability * 100

    return {
        "predicted_label": predicted_label,
        "prediction": "phishing" if predicted_label == 1 else "legitimate",
        "phishing_probability": phishing_probability,
        "risk_score": risk_score,
        "risk_level": risk_level_from_score(risk_score),
    }


def predict_email(email_text: str, model: Any | None = None) -> dict[str, float | int | str]:
    """Compatibility wrapper used by the Streamlit app."""
    return score_email(email_text, model)


def model_exists(model_path: str | Path = FINAL_MODEL_PATH) -> bool:
    """Return True when the final model artifact exists."""
    return Path(model_path).exists()
