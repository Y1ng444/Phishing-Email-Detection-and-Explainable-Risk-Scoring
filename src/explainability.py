"""Explainability module for model interpretation."""

import logging
import numpy as np
import re
from typing import Dict, List, Any, Tuple
import joblib

from .utils import setup_logger, MODELS_DIR
from .preprocessing import clean_email, _count_urls
from .feature_engineering import load_vectorizer

logger = setup_logger(__name__)


def get_top_features(
    model, feature_names: List[str], n: int = 20
) -> Tuple[List[str], List[float]]:
    """Extract top feature coefficients from logistic regression model.

    Args:
        model: Trained LogisticRegression model
        feature_names: List of feature names
        n: Number of top features to return

    Returns:
        Tuple of (feature names, coefficients)
    """
    # Get coefficients for phishing class
    coefficients = model.coef_[0]

    # Get indices of top positive (phishing) indicators
    top_phishing_indices = np.argsort(coefficients)[-n:][::-1]

    # Get indices of top negative (ham) indicators
    top_ham_indices = np.argsort(coefficients)[:n]

    top_phishing_features = [feature_names[i] for i in top_phishing_indices]
    top_phishing_coefs = [coefficients[i] for i in top_phishing_indices]

    top_ham_features = [feature_names[i] for i in top_ham_indices]
    top_ham_coefs = [coefficients[i] for i in top_ham_indices]

    return (
        top_phishing_features,
        top_phishing_coefs,
        top_ham_features,
        top_ham_coefs,
    )


def display_feature_importance(
    model, feature_names: List[str], n: int = 20
) -> None:
    """Display global feature importance from model.

    Args:
        model: Trained LogisticRegression model
        feature_names: List of feature names
        n: Number of top features to display
    """
    logger.info("\n" + "=" * 70)
    logger.info("GLOBAL FEATURE IMPORTANCE (Logistic Regression)")
    logger.info("=" * 70)

    top_phishing, phishing_coefs, top_ham, ham_coefs = get_top_features(
        model, feature_names, n
    )

    logger.info("\nTop 20 PHISHING INDICATORS (Strong Predictors):")
    logger.info("-" * 70)
    for i, (feature, coef) in enumerate(zip(top_phishing, phishing_coefs), 1):
        logger.info(f"{i:2d}. {feature:<30} Coefficient: {coef:+.6f}")

    logger.info("\nTop 20 HAM INDICATORS (Legitimate Email Markers):")
    logger.info("-" * 70)
    for i, (feature, coef) in enumerate(zip(top_ham, ham_coefs), 1):
        logger.info(f"{i:2d}. {feature:<30} Coefficient: {coef:+.6f}")

    logger.info("=" * 70)


def extract_suspicious_keywords(
    email_text: str, suspicious_words: List[str]
) -> List[str]:
    """Extract suspicious keywords from email.

    Args:
        email_text: Raw email text
        suspicious_words: List of suspicious keywords to search for

    Returns:
        List of found suspicious keywords
    """
    email_lower = email_text.lower()
    found_keywords = []

    for keyword in suspicious_words:
        # Use word boundary matching
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, email_lower):
            found_keywords.append(keyword)

    return found_keywords


def get_email_explanation(
    email_text: str,
    model,
    vectorizer,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    """Generate detailed explanation for a single email.

    Args:
        email_text: Raw email text
        model: Trained model
        vectorizer: TF-IDF vectorizer
        feature_names: List of all feature names
        top_n: Number of top contributing features to show

    Returns:
        Dictionary with explanation details
    """
    # Clean email
    cleaned_text = clean_email(email_text)

    # Transform to features
    tfidf_matrix = vectorizer.transform([cleaned_text])

    from .feature_engineering import create_statistical_features
    from scipy.sparse import hstack

    stat_features, stat_feature_names = create_statistical_features([email_text])
    X = hstack([tfidf_matrix, stat_features])

    # Get prediction and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    # Get model coefficients
    coefficients = model.coef_[0]

    # Calculate feature contributions
    # Convert to CSR format if needed for slicing
    if X.format != 'csr':
        X = X.tocsr()

    # For sparse TF-IDF features
    tfidf_dense = X[:, : len(feature_names) - len(stat_feature_names)].toarray()
    tfidf_contributions = np.abs(
        tfidf_dense[0] * coefficients[: len(feature_names) - len(stat_feature_names)]
    )

    # For statistical features
    stat_contributions = np.abs(
        stat_features[0] * coefficients[len(feature_names) - len(stat_feature_names) :]
    )

    # Get top contributing TF-IDF features
    top_tfidf_indices = np.argsort(tfidf_contributions)[-top_n:][::-1]
    top_tfidf_features = [
        (
            feature_names[i],
            tfidf_contributions[i],
            coefficients[i],
        )
        for i in top_tfidf_indices
        if tfidf_contributions[i] > 0
    ]

    # Get top contributing statistical features
    top_stat_indices = np.argsort(stat_contributions)[-5:][::-1]
    top_stat_features = [
        (
            stat_feature_names[i],
            stat_contributions[i],
            coefficients[len(feature_names) - len(stat_feature_names) + i],
        )
        for i in top_stat_indices
        if stat_contributions[i] > 0
    ]

    # Extract suspicious keywords (sample list)
    suspicious_keywords = [
        "verify",
        "confirm",
        "click",
        "urgent",
        "action",
        "update",
        "suspended",
        "limited",
        "unusual",
        "activity",
    ]
    found_keywords = extract_suspicious_keywords(email_text, suspicious_keywords)

    # Get positive evidence (phishing indicators present)
    positive_evidence = []
    top_phishing, _, _, _ = get_top_features(model, feature_names, 20)
    for feature in top_phishing:
        if feature in cleaned_text or feature in email_text:
            positive_evidence.append(feature)

    # Get negative evidence (ham indicators present)
    negative_evidence = []
    _, _, top_ham, _ = get_top_features(model, feature_names, 20)
    for feature in top_ham:
        if feature in cleaned_text or feature in email_text:
            negative_evidence.append(feature)

    explanation = {
        "prediction": "Phishing" if prediction == 1 else "Ham",
        "probability": float(probability),
        "risk_score": int(probability * 100),
        "top_tfidf_features": top_tfidf_features,
        "top_stat_features": top_stat_features,
        "suspicious_keywords": found_keywords,
        "positive_evidence": positive_evidence[:5],
        "negative_evidence": negative_evidence[:5],
        "total_features_active": int(np.sum(X[0] != 0)),
        "email_length": len(email_text),
        "word_count": len(email_text.split()),
        "url_count": _count_urls(email_text),
    }

    return explanation


def display_email_explanation(explanation: Dict[str, Any]) -> None:
    """Display explanation for email in readable format.

    Args:
        explanation: Explanation dictionary from get_email_explanation()
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"PREDICTION: {explanation['prediction']}")
    logger.info(f"Probability: {explanation['probability']:.4f}")
    logger.info(f"Risk Score: {explanation['risk_score']}/100")
    logger.info("=" * 70)

    logger.info("\nTop Contributing Features (Text-based):")
    for i, (feature, contribution, coef) in enumerate(
        explanation["top_tfidf_features"][:5], 1
    ):
        direction = "Phishing" if coef > 0 else "Ham"
        logger.info(
            f"  {i}. {feature:<30} ({direction:>8}) - "
            f"Contribution: {contribution:.4f}"
        )

    logger.info("\nTop Contributing Features (Statistical):")
    for i, (feature, contribution, coef) in enumerate(
        explanation["top_stat_features"], 1
    ):
        direction = "Phishing" if coef > 0 else "Ham"
        logger.info(
            f"  {i}. {feature:<30} ({direction:>8}) - "
            f"Contribution: {contribution:.4f}"
        )

    if explanation["suspicious_keywords"]:
        logger.info(f"\nSuspicious Keywords Found: {', '.join(explanation['suspicious_keywords'])}")

    if explanation["positive_evidence"]:
        logger.info(
            f"Phishing Indicators Present: {', '.join(explanation['positive_evidence'][:3])}"
        )

    if explanation["negative_evidence"]:
        logger.info(
            f"Legitimacy Indicators: {', '.join(explanation['negative_evidence'][:3])}"
        )

    logger.info(f"\nEmail Stats:")
    logger.info(f"  Length: {explanation['email_length']} characters")
    logger.info(f"  Words: {explanation['word_count']}")
    logger.info(f"  URLs: {explanation['url_count']}")
    logger.info(f"  Active Features: {explanation['total_features_active']}")
    logger.info("=" * 70)
