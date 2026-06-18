"""Simple coefficient-based explanations for sklearn text pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import hstack, issparse
from sklearn.pipeline import Pipeline

from .text_cleaning import count_urls
from .text_cleaning import clean_email_text


def _pipeline_parts(model_or_pipeline: Any):
    """Return vectorizer and classifier from a sklearn Pipeline."""
    if not isinstance(model_or_pipeline, Pipeline):
        raise TypeError("Expected a sklearn Pipeline with 'tfidf' and 'clf' steps.")

    try:
        return model_or_pipeline.named_steps["tfidf"], model_or_pipeline.named_steps["clf"]
    except KeyError as exc:
        raise ValueError("Pipeline must contain 'tfidf' and 'clf' steps.") from exc


def _get_coefficients(classifier: Any) -> np.ndarray:
    """Return linear model coefficients for the positive class."""
    if not hasattr(classifier, "coef_"):
        raise ValueError("Explanation requires a linear model with coef_.")

    coefficients = classifier.coef_
    if coefficients.ndim == 2:
        return coefficients[0]
    return coefficients


def get_feature_names(model_or_pipeline: Any, feature_names: list[str] | None = None) -> list[str]:
    """Return TF-IDF feature names from a pipeline or supplied list."""
    if feature_names is not None:
        return list(feature_names)

    vectorizer, _ = _pipeline_parts(model_or_pipeline)
    return list(vectorizer.get_feature_names_out())


def get_top_features(
    model_or_pipeline: Any,
    feature_names: list[str] | None = None,
    n: int = 20,
) -> tuple[list[str], list[float], list[str], list[float]]:
    """Return top phishing and ham indicators from model coefficients."""
    if isinstance(model_or_pipeline, Pipeline):
        _, classifier = _pipeline_parts(model_or_pipeline)
    else:
        classifier = model_or_pipeline

    names = get_feature_names(model_or_pipeline, feature_names)
    coefficients = _get_coefficients(classifier)
    usable = min(len(names), len(coefficients))
    names = names[:usable]
    coefficients = coefficients[:usable]

    top_phishing_indices = np.argsort(coefficients)[-n:][::-1]
    top_ham_indices = np.argsort(coefficients)[:n]

    top_phishing = [names[i] for i in top_phishing_indices]
    top_phishing_coefs = [float(coefficients[i]) for i in top_phishing_indices]
    top_ham = [names[i] for i in top_ham_indices]
    top_ham_coefs = [float(coefficients[i]) for i in top_ham_indices]

    return top_phishing, top_phishing_coefs, top_ham, top_ham_coefs


def get_email_explanation(
    email_text: str,
    model_or_pipeline: Any,
    vectorizer: Any | None = None,
    feature_names: list[str] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Explain which active TF-IDF features drive one email prediction.

    Preferred usage is ``get_email_explanation(email_text, pipeline)`` where the
    pipeline contains ``tfidf`` and ``clf`` steps.
    """
    if isinstance(model_or_pipeline, Pipeline):
        pipeline = model_or_pipeline
        vectorizer, classifier = _pipeline_parts(pipeline)
    else:
        if vectorizer is None:
            raise ValueError("A vectorizer is required when explaining a raw estimator.")
        classifier = model_or_pipeline

    names = get_feature_names(model_or_pipeline if isinstance(model_or_pipeline, Pipeline) else vectorizer, feature_names)
    transform_text = email_text if isinstance(model_or_pipeline, Pipeline) else clean_email_text(email_text)
    matrix = vectorizer.transform([transform_text])
    coefficients = _get_coefficients(classifier)
    if not isinstance(model_or_pipeline, Pipeline) and matrix.shape[1] < len(coefficients):
        try:
            from .feature_engineering import create_statistical_features

            stat_features, _ = create_statistical_features([email_text])
            matrix = hstack([matrix, stat_features], format="csr")
        except Exception:
            pass
    usable = min(matrix.shape[1], len(coefficients), len(names))
    matrix = matrix[:, :usable]
    coefficients = coefficients[:usable]
    names = names[:usable]

    if hasattr(classifier, "predict_proba"):
        probability = float(model_or_pipeline.predict_proba([email_text])[:, 1][0]) if isinstance(model_or_pipeline, Pipeline) else float(classifier.predict_proba(matrix)[:, 1][0])
        score_type = "probability"
        score = probability
    elif isinstance(model_or_pipeline, Pipeline) and hasattr(model_or_pipeline, "decision_function"):
        raw_score = float(model_or_pipeline.decision_function([email_text])[0])
        probability = float(1.0 / (1.0 + np.exp(-raw_score)))
        score_type = "decision_score"
        score = raw_score
    else:
        probability = 0.0
        score_type = "score"
        score = 0.0

    prediction = int(model_or_pipeline.predict([email_text])[0]) if isinstance(model_or_pipeline, Pipeline) else int(classifier.predict(matrix)[0])

    row = matrix.toarray()[0] if issparse(matrix) else np.asarray(matrix)[0]
    contributions = row * coefficients
    active_indices = np.flatnonzero(row)
    active_sorted = active_indices[np.argsort(np.abs(contributions[active_indices]))[::-1]]

    top_active_features = [
        {
            "feature": names[index],
            "tfidf_value": float(row[index]),
            "coefficient": float(coefficients[index]),
            "contribution": float(contributions[index]),
            "direction": "phishing" if coefficients[index] > 0 else "ham",
        }
        for index in active_sorted[:top_n]
    ]

    top_phishing, phishing_coefs, top_ham, ham_coefs = get_top_features(
        model_or_pipeline,
        feature_names=names,
        n=top_n,
    )

    return {
        "prediction": "Phishing" if prediction == 1 else "Ham",
        "probability": probability,
        "score": score,
        "score_type": score_type,
        "risk_score": int(round(probability * 100)),
        "top_active_features": top_active_features,
        "top_phishing_indicators": list(zip(top_phishing, phishing_coefs)),
        "top_ham_indicators": list(zip(top_ham, ham_coefs)),
        "email_length": len(email_text),
        "word_count": len(email_text.split()),
        "url_count": count_urls(email_text),
        "total_features_active": int(len(active_indices)),
    }


def display_feature_importance(model_or_pipeline: Any, feature_names: list[str] | None = None, n: int = 20) -> None:
    """Print top coefficient-based features."""
    top_phishing, phishing_coefs, top_ham, ham_coefs = get_top_features(
        model_or_pipeline,
        feature_names=feature_names,
        n=n,
    )
    print("Top phishing indicators:")
    for feature, coefficient in zip(top_phishing, phishing_coefs):
        print(f"  {feature}: {coefficient:+.4f}")

    print("Top ham indicators:")
    for feature, coefficient in zip(top_ham, ham_coefs):
        print(f"  {feature}: {coefficient:+.4f}")


def display_email_explanation(explanation: dict[str, Any]) -> None:
    """Print a compact per-email explanation."""
    print(f"Prediction: {explanation['prediction']}")
    print(f"Risk score: {explanation['risk_score']}/100")
    print("Top active features:")
    for item in explanation["top_active_features"]:
        print(
            f"  {item['feature']}: {item['direction']} "
            f"contribution={item['contribution']:+.4f}"
        )
