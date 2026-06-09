"""Baseline model training using Multinomial Naive Bayes."""

import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib

from .utils import setup_logger, RANDOM_SEED, TEST_SIZE, MODELS_DIR
from .preprocessing import load_dataset, preprocess_dataset, explore_dataset, visualize_eda
from .feature_engineering import create_all_features

logger = setup_logger(__name__)


def train_baseline_model(X_train, y_train) -> MultinomialNB:
    """Train Multinomial Naive Bayes model.

    Args:
        X_train: Training features (sparse matrix)
        y_train: Training labels

    Returns:
        Trained MultinomialNB model
    """
    logger.info("Training Multinomial Naive Bayes baseline model...")

    model = MultinomialNB()
    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def run_baseline_training() -> None:
    """Run complete baseline model training pipeline."""
    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING - MULTINOMIAL NAIVE BAYES")
    logger.info("=" * 60)

    # Load and explore dataset
    logger.info("\n1. Loading and exploring dataset...")
    df, _ = load_dataset()
    explore_dataset(df)
    visualize_eda(df)

    # Preprocess
    logger.info("\n2. Preprocessing data...")
    df = preprocess_dataset(df)

    # Create features
    logger.info("\n3. Creating features...")
    features_dict = create_all_features(df["text_cleaned"], fit_vectorizer=True)
    X = features_dict["X"]
    y = df["label"].values

    # Train-test split
    logger.info("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train model
    logger.info("\n5. Training baseline model...")
    model = train_baseline_model(X_train, y_train)

    # Evaluate
    logger.info("\n6. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, "Multinomial Naive Bayes")

    # Save model and vectorizer
    logger.info("\n7. Saving model and vectorizer...")
    model_path = MODELS_DIR / "naive_bayes_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    from .feature_engineering import save_vectorizer

    save_vectorizer(features_dict["vectorizer"], "tfidf_vectorizer.pkl")

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE TRAINING COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_baseline_training()
