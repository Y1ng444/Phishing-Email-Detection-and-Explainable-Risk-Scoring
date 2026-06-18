"""Improved model training using Logistic Regression."""

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib

from .utils import setup_logger, RANDOM_SEED, TEST_SIZE, MODELS_DIR, LOGISTIC_REGRESSION_MAX_ITER
from .preprocessing import load_dataset, preprocess_dataset, explore_dataset
from .feature_engineering import create_all_features, load_vectorizer

logger = setup_logger(__name__)


def train_logistic_model(X_train, y_train) -> LogisticRegression:
    """Train Logistic Regression model with balanced classes.

    Args:
        X_train: Training features (sparse matrix)
        y_train: Training labels

    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression model...")

    model = LogisticRegression(
    solver="saga",
    penalty="l2",
    class_weight="balanced",
    max_iter=5000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
    )
    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model


def evaluate_and_compare(
    baseline_metrics: dict, logistic_metrics: dict
) -> None:
    """Compare baseline and improved model metrics.

    Args:
        baseline_metrics: Metrics from baseline model
        logistic_metrics: Metrics from logistic regression model
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    logger.info(f"{'Metric':<15} {'Naive Bayes':<20} {'Logistic Reg':<20} {'Improvement':<15}")
    logger.info("-" * 70)

    for name, key in zip(metrics_names, keys):
        baseline_val = baseline_metrics[key]
        logistic_val = logistic_metrics[key]
        improvement = logistic_val - baseline_val

        logger.info(
            f"{name:<15} {baseline_val:<20.4f} {logistic_val:<20.4f} {improvement:+.4f}"
        )

    logger.info("=" * 70)


def run_logistic_training() -> None:
    """Run complete logistic regression model training pipeline."""
    logger.info("=" * 60)
    logger.info("IMPROVED MODEL TRAINING - LOGISTIC REGRESSION")
    logger.info("=" * 60)

    # Load and preprocess
    logger.info("\n1. Loading and preprocessing data...")
    df, _ = load_dataset()
    df = preprocess_dataset(df)

    # Load vectorizer and create features
    logger.info("\n2. Loading vectorizer and creating features...")
    try:
        vectorizer = load_vectorizer("tfidf_vectorizer.pkl")
        features_dict = create_all_features(
            df["text_cleaned"], vectorizer=vectorizer, fit_vectorizer=False
        )
    except FileNotFoundError:
        logger.warning(
            "Vectorizer not found. Creating features with new vectorizer."
        )
        features_dict = create_all_features(df["text_cleaned"], fit_vectorizer=True)
        from .feature_engineering import save_vectorizer

        save_vectorizer(features_dict["vectorizer"], "tfidf_vectorizer.pkl")

    X = features_dict["X"]
    y = df["label"].values

    # Train-test split
    logger.info("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train model
    logger.info("\n4. Training logistic regression model...")
    model = train_logistic_model(X_train, y_train)

    # Evaluate
    logger.info("\n5. Evaluating model...")
    from .train_baseline import evaluate_model as eval_model

    logistic_metrics = eval_model(model, X_test, y_test, "Logistic Regression")

    # Load baseline metrics for comparison (if available)
    logger.info("\n6. Comparing with baseline model...")
    try:
        baseline_model = joblib.load(MODELS_DIR / "naive_bayes_model.pkl")
        baseline_metrics = eval_model(baseline_model, X_test, y_test, "Naive Bayes (from checkpoint)")
        evaluate_and_compare(baseline_metrics, logistic_metrics)
    except FileNotFoundError:
        logger.info("Baseline model not found for comparison")

    # Save model
    logger.info("\n7. Saving model...")
    model_path = MODELS_DIR / "logistic_regression_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature names for explainability
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    joblib.dump(features_dict["feature_names"], feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")

    logger.info("\n" + "=" * 60)
    logger.info("LOGISTIC REGRESSION TRAINING COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_logistic_training()
