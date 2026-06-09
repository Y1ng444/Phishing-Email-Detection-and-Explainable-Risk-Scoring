"""Risk scoring and prediction module."""

import logging
from datetime import datetime
from typing import Dict, Any
import joblib

from .utils import setup_logger, MODELS_DIR
from .preprocessing import clean_email
from .feature_engineering import create_all_features, load_vectorizer

logger = setup_logger(__name__)


def load_model(model_name: str = "logistic_regression_model.pkl"):
    """Load trained model from disk.

    Args:
        model_name: Model filename

    Returns:
        Trained model

    Raises:
        FileNotFoundError: If model not found
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def get_risk_level_info(probability: float) -> Dict[str, Any]:
    """Convert probability to risk level with details.

    Args:
        probability: Phishing probability (0-1)

    Returns:
        Dictionary with risk_score, risk_level, and color
    """
    risk_score = int(probability * 100)

    if risk_score <= 30:
        return {
            "risk_score": risk_score,
            "risk_level": "Low",
            "color": "#2ecc71",
            "description": "Email appears legitimate",
        }
    elif risk_score <= 60:
        return {
            "risk_score": risk_score,
            "risk_level": "Medium",
            "color": "#f39c12",
            "description": "Email has some phishing characteristics",
        }
    elif risk_score <= 80:
        return {
            "risk_score": risk_score,
            "risk_level": "High",
            "color": "#e74c3c",
            "description": "Email is likely phishing",
        }
    else:
        return {
            "risk_score": risk_score,
            "risk_level": "Critical",
            "color": "#c0392b",
            "description": "Email is highly likely phishing",
        }


def predict_email(
    email_text: str, model=None, vectorizer=None
) -> Dict[str, Any]:
    """Analyze email and return prediction with risk score.

    Args:
        email_text: Raw email text to analyze
        model: Trained model (loads from disk if None)
        vectorizer: TF-IDF vectorizer (loads from disk if None)

    Returns:
        Dictionary with prediction results:
            {
                "prediction": "Ham" | "Phishing",
                "risk_score": int (0-100),
                "risk_level": str,
                "probability": float (0-1),
                "color": str (hex color code),
                "description": str,
                "timestamp": str,
                "email_length": int,
                "word_count": int
            }

    Raises:
        ValueError: If email text is empty
        FileNotFoundError: If model or vectorizer not found
    """
    if not email_text or not email_text.strip():
        raise ValueError("Email text cannot be empty")

    # Load model and vectorizer if not provided
    if model is None:
        model = load_model("logistic_regression_model.pkl")

    if vectorizer is None:
        vectorizer = load_vectorizer("tfidf_vectorizer.pkl")

    # Clean email
    cleaned_text = clean_email(email_text)

    if not cleaned_text or len(cleaned_text.split()) == 0:
        raise ValueError("Email text is empty after cleaning")

    # Create features
    from .feature_engineering import create_statistical_features

    # TF-IDF features
    tfidf_matrix = vectorizer.transform([cleaned_text])

    # Statistical features
    stat_features, _ = create_statistical_features([email_text])

    # Combine features
    from scipy.sparse import hstack

    X = hstack([tfidf_matrix, stat_features])

    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Phishing probability

    # Get risk level info
    risk_info = get_risk_level_info(probability)

    # Prepare response
    result = {
        "prediction": "Phishing" if prediction == 1 else "Ham",
        "risk_score": risk_info["risk_score"],
        "risk_level": risk_info["risk_level"],
        "probability": float(probability),
        "color": risk_info["color"],
        "description": risk_info["description"],
        "timestamp": datetime.now().isoformat(),
        "email_length": len(email_text),
        "word_count": len(email_text.split()),
    }

    logger.info(
        f"Email analyzed: {result['prediction']} "
        f"(probability: {probability:.4f}, risk: {result['risk_level']})"
    )

    return result


def batch_predict(
    email_texts: list, model=None, vectorizer=None
) -> list:
    """Analyze multiple emails.

    Args:
        email_texts: List of email texts
        model: Trained model (loads from disk if None)
        vectorizer: TF-IDF vectorizer (loads from disk if None)

    Returns:
        List of prediction dictionaries
    """
    logger.info(f"Analyzing {len(email_texts)} emails...")

    if model is None:
        model = load_model("logistic_regression_model.pkl")

    if vectorizer is None:
        vectorizer = load_vectorizer("tfidf_vectorizer.pkl")

    results = []
    for i, email_text in enumerate(email_texts, 1):
        try:
            result = predict_email(email_text, model, vectorizer)
            results.append(result)

            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(email_texts)} emails")

        except (ValueError, Exception) as e:
            logger.warning(f"Error processing email {i}: {str(e)}")
            results.append(None)

    logger.info(f"Batch prediction completed: {len(results)} emails")

    return results
