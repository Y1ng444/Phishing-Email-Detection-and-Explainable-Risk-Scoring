"""Coefficient-based explanations for phishing email classifiers.

The project keeps a legacy text-only Logistic Regression model and adds a
newer text+metadata Logistic Regression model. This module handles both shapes
so training scripts and the Streamlit app can share explanation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering import METADATA_FEATURES, MODEL_INPUT_COLUMNS, build_model_input


def _require_pipeline(model: Pipeline) -> None:
    """Validate that explanations receive a fitted sklearn Pipeline."""
    if not isinstance(model, Pipeline):
        raise TypeError("Expected a fitted sklearn Pipeline.")
    if "clf" not in model.named_steps:
        raise ValueError("Pipeline must contain a 'clf' step.")


def _is_text_metadata_pipeline(model: Pipeline) -> bool:
    """Return True for the new ColumnTransformer-based model."""
    return isinstance(model, Pipeline) and "features" in model.named_steps


def _legacy_pipeline_parts(model: Pipeline) -> tuple[Any, Any]:
    """Return the TF-IDF vectorizer and classifier from a text-only pipeline."""
    _require_pipeline(model)
    if "tfidf" not in model.named_steps:
        raise ValueError("Legacy explanation requires a 'tfidf' step.")
    return model.named_steps["tfidf"], model.named_steps["clf"]


def _coefficients(model: Pipeline) -> np.ndarray:
    """Return Logistic Regression coefficients for the phishing class."""
    _require_pipeline(model)
    classifier = model.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("Explanation requires a linear model with coefficients.")
    coefficients = classifier.coef_
    if coefficients.ndim == 2:
        return coefficients[0]
    return coefficients


def _normalize_feature_name(name: str) -> str:
    """Remove sklearn transformer prefixes from feature names when present."""
    return str(name).split("__")[-1]


def _feature_names(model: Pipeline) -> np.ndarray:
    """Return feature names for either supported pipeline shape."""
    _require_pipeline(model)
    if _is_text_metadata_pipeline(model):
        return np.asarray(model.named_steps["features"].get_feature_names_out())
    vectorizer, _ = _legacy_pipeline_parts(model)
    return np.asarray(vectorizer.get_feature_names_out())


def _as_row_array(matrix: Any) -> np.ndarray:
    """Convert a one-row sparse or dense matrix to a flat numpy array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix).ravel()


def _extract_text_value(email_input: Any) -> str:
    """Return the first email text from a string, Series, or DataFrame."""
    if isinstance(email_input, pd.DataFrame) and "text_combined" in email_input.columns:
        return str(email_input["text_combined"].iloc[0])
    if isinstance(email_input, pd.Series):
        return str(email_input.iloc[0])
    return str(email_input)


def _prepare_text_metadata_input(email_input: Any) -> pd.DataFrame:
    """Return the model input frame expected by the text+metadata pipeline."""
    if isinstance(email_input, pd.DataFrame) and set(MODEL_INPUT_COLUMNS).issubset(email_input.columns):
        return email_input[MODEL_INPUT_COLUMNS].copy()
    return build_model_input(email_input)


def _transformed_values(model: Pipeline, email_input: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return feature names and transformed feature values for one email."""
    feature_names = _feature_names(model)
    if _is_text_metadata_pipeline(model):
        x_row = _prepare_text_metadata_input(email_input).head(1)
        transformed = model.named_steps["features"].transform(x_row)
    else:
        vectorizer, _ = _legacy_pipeline_parts(model)
        transformed = vectorizer.transform([_extract_text_value(email_input)])
    return feature_names, _as_row_array(transformed)


def _metadata_mask(feature_names: np.ndarray) -> np.ndarray:
    """Identify metadata columns in transformed feature names."""
    normalized = np.asarray([_normalize_feature_name(name) for name in feature_names])
    return np.isin(normalized, METADATA_FEATURES)


def get_global_indicators(model: Pipeline, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top global text indicators for phishing and legitimate classes."""
    feature_names = _feature_names(model)
    coefficients = _coefficients(model)
    text_mask = ~_metadata_mask(feature_names)
    text_features = feature_names[text_mask]
    text_coefficients = coefficients[text_mask]

    phishing_indices = np.argsort(text_coefficients)[-top_n:][::-1]
    legitimate_indices = np.argsort(text_coefficients)[:top_n]

    phishing = pd.DataFrame(
        {
            "term": text_features[phishing_indices],
            "coefficient": text_coefficients[phishing_indices],
        }
    )
    legitimate = pd.DataFrame(
        {
            "term": text_features[legitimate_indices],
            "coefficient": text_coefficients[legitimate_indices],
        }
    )
    return phishing, legitimate


def save_global_indicators(model: Pipeline, results_dir: str | Path, top_n: int = 20) -> None:
    """Save global coefficient indicators to legacy CSV filenames."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    phishing, legitimate = get_global_indicators(model, top_n=top_n)
    phishing.to_csv(results_path / "top_phishing_indicators.csv", index=False)
    legitimate.to_csv(results_path / "top_legitimate_indicators.csv", index=False)


def get_metadata_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return scaled metadata coefficients for the text+metadata model."""
    if not _is_text_metadata_pipeline(model):
        return pd.DataFrame(columns=["feature", "coefficient"])

    feature_names = _feature_names(model)
    coefficients = _coefficients(model)
    mask = _metadata_mask(feature_names)
    rows = pd.DataFrame(
        {
            "feature": [_normalize_feature_name(name) for name in feature_names[mask]],
            "coefficient": coefficients[mask],
        }
    )
    return rows.sort_values("coefficient", ascending=False).reset_index(drop=True)


def explain_email_prediction(
    email_input: Any,
    model: Pipeline,
    top_n: int = 10,
) -> pd.DataFrame:
    """Show active text terms that contribute most toward phishing."""
    feature_names, values = _transformed_values(model, email_input)
    coefficients = _coefficients(model)
    contributions = values * coefficients
    text_mask = ~_metadata_mask(feature_names)

    active_positive = np.where((values > 0) & (contributions > 0) & text_mask)[0]
    sorted_indices = active_positive[np.argsort(contributions[active_positive])[::-1]]

    rows = [
        {
            "term": _normalize_feature_name(feature_names[index]),
            "tfidf_value": float(values[index]),
            "coefficient": float(coefficients[index]),
            "contribution": float(contributions[index]),
        }
        for index in sorted_indices[:top_n]
    ]
    return pd.DataFrame(rows)


def explain_metadata_prediction(
    email_input: Any,
    model: Pipeline,
    top_n: int = 10,
) -> pd.DataFrame:
    """Show metadata feature contributions for one email."""
    if not _is_text_metadata_pipeline(model):
        return pd.DataFrame(
            columns=["feature", "raw_value", "scaled_value", "coefficient", "contribution", "direction"]
        )

    x_row = _prepare_text_metadata_input(email_input).head(1)
    feature_names, values = _transformed_values(model, x_row)
    coefficients = _coefficients(model)
    contributions = values * coefficients
    mask = _metadata_mask(feature_names)
    metadata_names = [_normalize_feature_name(name) for name in feature_names[mask]]

    rows = []
    for feature, scaled_value, coefficient, contribution in zip(
        metadata_names,
        values[mask],
        coefficients[mask],
        contributions[mask],
    ):
        rows.append(
            {
                "feature": feature,
                "raw_value": float(x_row[feature].iloc[0]) if feature in x_row.columns else 0.0,
                "scaled_value": float(scaled_value),
                "coefficient": float(coefficient),
                "contribution": float(contribution),
                "direction": "toward_phishing" if contribution > 0 else "toward_legitimate",
            }
        )

    explanation = pd.DataFrame(rows)
    if explanation.empty:
        return explanation
    explanation["abs_contribution"] = explanation["contribution"].abs()
    explanation = explanation.sort_values("abs_contribution", ascending=False)
    return explanation.drop(columns=["abs_contribution"]).head(top_n).reset_index(drop=True)


def _sample_soc_row(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, Any]:
    """Select a high-risk predicted phishing sample for the SOC explanation."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    phishing_indices = np.where(predictions == 1)[0]
    if len(phishing_indices) == 0:
        selected = int(np.argmax(probabilities))
    else:
        selected = int(phishing_indices[np.argmax(probabilities[phishing_indices])])
    true_label = y_test.iloc[selected] if y_test is not None else None
    return x_test.iloc[[selected]].copy(), true_label


def save_text_metadata_explanations(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: str | Path,
    top_n: int = 20,
) -> None:
    """Save global and sample-level artifacts for the text+metadata model."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    phishing, legitimate = get_global_indicators(model, top_n=top_n)
    phishing.to_csv(results_path / "top_phishing_indicators_text_metadata.csv", index=False)
    legitimate.to_csv(results_path / "top_legitimate_indicators_text_metadata.csv", index=False)
    get_metadata_coefficients(model).to_csv(results_path / "metadata_coefficients.csv", index=False)

    sample_x, true_label = _sample_soc_row(model, x_test, y_test)
    probability = float(model.predict_proba(sample_x)[:, 1][0])
    predicted_label = int(model.predict(sample_x)[0])
    risk_score = probability * 100

    from src.risk_scoring import risk_level_from_score

    top_text = explain_email_prediction(sample_x, model, top_n=10)
    top_metadata = explain_metadata_prediction(sample_x, model, top_n=10)
    top_text_terms = ", ".join(top_text["term"].head(3).astype(str).tolist()) or "no active text terms"
    top_metadata_names = (
        ", ".join(top_metadata["feature"].head(3).astype(str).tolist()) or "no metadata signals"
    )

    explanation = {
        "original_email_preview": _extract_text_value(sample_x)[:700],
        "true_label": None if pd.isna(true_label) else int(true_label),
        "predicted_label": predicted_label,
        "predicted_class": "phishing" if predicted_label == 1 else "legitimate",
        "phishing_probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level_from_score(risk_score),
        "top_text_contributions": top_text.to_dict(orient="records"),
        "top_metadata_contributions": top_metadata.to_dict(orient="records"),
        "soc_explanation": (
            f"The model assigns {probability:.3f} phishing probability. "
            f"Top text drivers include {top_text_terms}; top metadata drivers include "
            f"{top_metadata_names}. Review the message body, links, and sender context before action."
        ),
    }

    with (results_path / "sample_soc_explanation.json").open("w", encoding="utf-8") as file:
        json.dump(explanation, file, indent=2, ensure_ascii=False)

    pd.DataFrame(
        [
            {
                "original_email_preview": explanation["original_email_preview"],
                "true_label": explanation["true_label"],
                "predicted_label": explanation["predicted_label"],
                "predicted_class": explanation["predicted_class"],
                "phishing_probability": explanation["phishing_probability"],
                "risk_score": explanation["risk_score"],
                "risk_level": explanation["risk_level"],
                "soc_explanation": explanation["soc_explanation"],
            }
        ]
    ).to_csv(results_path / "sample_soc_explanation.csv", index=False)


def get_email_explanation(email_text: str, model: Pipeline, top_n: int = 10) -> dict[str, Any]:
    """Return a dictionary form of a single-email explanation."""
    terms = explain_email_prediction(email_text, model, top_n=top_n)
    metadata = explain_metadata_prediction(email_text, model, top_n=top_n)
    return {
        "top_phishing_terms": terms.to_dict(orient="records"),
        "metadata_contributions": metadata.to_dict(orient="records"),
    }
