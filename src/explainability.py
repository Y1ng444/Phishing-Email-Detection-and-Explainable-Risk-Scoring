"""Coefficient-based explanations for the Logistic Regression TF-IDF model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _pipeline_parts(model: Pipeline) -> tuple[Any, Any]:
    """Return the TF-IDF vectorizer and classifier from a fitted pipeline."""
    if not isinstance(model, Pipeline):
        raise TypeError("Expected a fitted sklearn Pipeline.")
    if "tfidf" not in model.named_steps or "clf" not in model.named_steps:
        raise ValueError("Pipeline must contain 'tfidf' and 'clf' steps.")
    return model.named_steps["tfidf"], model.named_steps["clf"]


def _coefficients(model: Pipeline) -> np.ndarray:
    """Return Logistic Regression coefficients for the phishing class."""
    _, classifier = _pipeline_parts(model)
    if not hasattr(classifier, "coef_"):
        raise ValueError("Explanation requires a linear model with coefficients.")
    coefficients = classifier.coef_
    if coefficients.ndim == 2:
        return coefficients[0]
    return coefficients


def get_global_indicators(model: Pipeline, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top global phishing and legitimate indicators."""
    vectorizer, _ = _pipeline_parts(model)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = _coefficients(model)

    phishing_indices = np.argsort(coefficients)[-top_n:][::-1]
    legitimate_indices = np.argsort(coefficients)[:top_n]

    phishing = pd.DataFrame(
        {
            "term": feature_names[phishing_indices],
            "coefficient": coefficients[phishing_indices],
        }
    )
    legitimate = pd.DataFrame(
        {
            "term": feature_names[legitimate_indices],
            "coefficient": coefficients[legitimate_indices],
        }
    )
    return phishing, legitimate


def save_global_indicators(model: Pipeline, results_dir: str | Path, top_n: int = 20) -> None:
    """Save global coefficient indicators to CSV files."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    phishing, legitimate = get_global_indicators(model, top_n=top_n)
    phishing.to_csv(results_path / "top_phishing_indicators.csv", index=False)
    legitimate.to_csv(results_path / "top_legitimate_indicators.csv", index=False)


def explain_email_prediction(
    email_text: str,
    model: Pipeline,
    top_n: int = 10,
) -> pd.DataFrame:
    """Show active terms in one email that contribute most toward phishing."""
    vectorizer, _ = _pipeline_parts(model)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = _coefficients(model)

    tfidf_row = vectorizer.transform([email_text])
    values = tfidf_row.toarray()[0]
    contributions = values * coefficients

    active_positive = np.where((values > 0) & (contributions > 0))[0]
    sorted_indices = active_positive[np.argsort(contributions[active_positive])[::-1]]

    rows = [
        {
            "term": feature_names[index],
            "tfidf_value": float(values[index]),
            "coefficient": float(coefficients[index]),
            "contribution": float(contributions[index]),
        }
        for index in sorted_indices[:top_n]
    ]
    return pd.DataFrame(rows)


def get_email_explanation(email_text: str, model: Pipeline, top_n: int = 10) -> dict[str, Any]:
    """Return a dictionary form of a single-email explanation."""
    terms = explain_email_prediction(email_text, model, top_n=top_n)
    return {"top_phishing_terms": terms.to_dict(orient="records")}
