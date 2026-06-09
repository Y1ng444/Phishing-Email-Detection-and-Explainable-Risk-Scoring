"""Feature engineering module for text and statistical features."""

import logging
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from .utils import (
    setup_logger,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    MODELS_DIR,
)
from .preprocessing import get_statistics

logger = setup_logger(__name__)


def create_tfidf_features(
    texts: pd.Series, vectorizer: TfidfVectorizer | None = None
) -> tuple:
    """Create TF-IDF features from text.

    Args:
        texts: Series of text documents
        vectorizer: Fitted TfidfVectorizer (if None, fits new one)

    Returns:
        Tuple of (sparse matrix, vectorizer, feature names)
    """
    if vectorizer is None:
        logger.info(
            f"Creating TF-IDF vectorizer with max_features={TFIDF_MAX_FEATURES}, "
            f"ngram_range={TFIDF_NGRAM_RANGE}"
        )
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    else:
        logger.info("Transforming text with existing TF-IDF vectorizer")
        tfidf_matrix = vectorizer.transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, vectorizer, feature_names


def create_statistical_features(texts: pd.Series) -> np.ndarray:
    """Create statistical features from text.

    Args:
        texts: Series of text documents

    Returns:
        Dense numpy array of statistical features
    """
    logger.info("Creating statistical features...")

    features_list = []
    feature_names = [
        "email_length",
        "word_count",
        "uppercase_ratio",
        "digit_ratio",
        "url_count",
    ]

    for text in texts:
        stats = get_statistics(text)
        features_list.append(
            [
                stats["email_length"],
                stats["word_count"],
                stats["uppercase_ratio"],
                stats["digit_ratio"],
                stats["url_count"],
            ]
        )

    features_array = np.array(features_list, dtype=np.float32)

    logger.info(f"Statistical features shape: {features_array.shape}")

    return features_array, feature_names


def combine_features(
    tfidf_matrix: csr_matrix, statistical_features: np.ndarray
) -> tuple:
    """Combine TF-IDF and statistical features.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix
        statistical_features: Dense statistical features array

    Returns:
        Tuple of (combined sparse matrix, all feature names)
    """
    logger.info("Combining TF-IDF and statistical features...")

    # Convert statistical features to sparse matrix
    stats_sparse = csr_matrix(statistical_features, dtype=np.float32)

    # Combine horizontally
    combined_matrix = hstack([tfidf_matrix, stats_sparse])

    logger.info(f"Combined feature matrix shape: {combined_matrix.shape}")

    return combined_matrix, stats_sparse


def create_all_features(
    texts: pd.Series,
    vectorizer: TfidfVectorizer | None = None,
    fit_vectorizer: bool = True,
) -> dict:
    """Create all features for modeling.

    Args:
        texts: Series of text documents
        vectorizer: Fitted TfidfVectorizer (if None and fit_vectorizer=True, creates new)
        fit_vectorizer: Whether to fit new vectorizer

    Returns:
        Dictionary containing feature matrix and metadata
    """
    logger.info("Creating all features for modeling...")

    # TF-IDF features
    tfidf_matrix, vectorizer, tfidf_feature_names = create_tfidf_features(
        texts, vectorizer=vectorizer if not fit_vectorizer else None
    )

    # Statistical features
    stat_features, stat_feature_names = create_statistical_features(texts)

    # Combine features
    combined_matrix, stats_sparse = combine_features(tfidf_matrix, stat_features)

    # Create full feature names list
    all_feature_names = list(tfidf_feature_names) + stat_feature_names

    result = {
        "X": combined_matrix,
        "X_tfidf": tfidf_matrix,
        "X_stats": stat_features,
        "vectorizer": vectorizer,
        "feature_names": all_feature_names,
        "tfidf_feature_names": list(tfidf_feature_names),
        "stat_feature_names": stat_feature_names,
    }

    logger.info(f"Total feature count: {len(all_feature_names)}")

    return result


def save_vectorizer(vectorizer: TfidfVectorizer, filename: str = "tfidf_vectorizer.pkl") -> None:
    """Save TF-IDF vectorizer to disk.

    Args:
        vectorizer: TfidfVectorizer to save
        filename: Output filename
    """
    path = MODELS_DIR / filename
    joblib.dump(vectorizer, path)
    logger.info(f"Vectorizer saved to {path}")


def load_vectorizer(filename: str = "tfidf_vectorizer.pkl") -> TfidfVectorizer:
    """Load TF-IDF vectorizer from disk.

    Args:
        filename: Input filename

    Returns:
        Loaded TfidfVectorizer

    Raises:
        FileNotFoundError: If vectorizer file not found
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {path}")

    vectorizer = joblib.load(path)
    logger.info(f"Vectorizer loaded from {path}")
    return vectorizer
