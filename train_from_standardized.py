"""Train phishing detection models from the standardized dataset."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.feature_engineering import create_statistical_features
from src.utils import RANDOM_SEED, TEST_SIZE


PROJECT_ROOT = Path(__file__).parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "phishing_email_standardized.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

EMPTY_VALUES = {"", "nan", "none", "null", "nat", "na", "n/a", "<na>"}


def setup_logger() -> logging.Logger:
    """Create a console logger for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("train_from_standardized")


logger = setup_logger()


def fallback_clean_email(text: str) -> str:
    """Small text cleaner used only if src.preprocessing.clean_email is unavailable."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_email_cleaner() -> Callable[[str], str]:
    """Return the project cleaner when available, otherwise a local fallback."""
    try:
        from src.preprocessing import clean_email

        return clean_email
    except Exception as exc:
        logger.warning(
            "Could not import src.preprocessing.clean_email (%s). "
            "Using the local fallback cleaner.",
            exc,
        )
        return fallback_clean_email


def clean_text_value(value: object) -> str:
    """Convert null-like values to empty strings."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text.lower() in EMPTY_VALUES:
        return ""

    return text


def read_csv_with_fallbacks(csv_path: Path) -> pd.DataFrame:
    """Read a CSV with UTF-8 first and common fallback encodings."""
    last_error: Exception | None = None

    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{csv_path} is empty") from exc
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not read {csv_path}: {last_error}")


def load_standardized_dataset(input_path: Path) -> pd.DataFrame:
    """Load and validate the standardized dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Standardized dataset not found at {input_path}. "
            "Run standardize_datasets.py first."
        )

    df = read_csv_with_fallbacks(input_path)
    required_columns = {"text_combined", "label"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required columns: {sorted(missing_columns)}"
        )

    rows_loaded = len(df)
    df = df.copy()
    df["text_combined"] = df["text_combined"].map(clean_text_value)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df = df[df["text_combined"] != ""]
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    logger.info("Rows loaded: %s", f"{rows_loaded:,}")
    logger.info("Rows usable for training: %s", f"{len(df):,}")

    if df.empty:
        raise ValueError("No usable rows remain after loading the standardized dataset.")

    class_counts = df["label"].value_counts().sort_index()
    logger.info("Label distribution:\n%s", class_counts.to_string())

    if len(class_counts) < 2:
        raise ValueError("Training requires both label classes: 0 and 1.")

    if class_counts.min() < 2:
        raise ValueError(
            "Stratified splitting requires at least two rows in each class."
        )

    return df


def add_cleaned_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add text_cleaned using the existing preprocessing function when available."""
    cleaner = get_email_cleaner()
    df = df.copy()
    logger.info("Cleaning email text...")
    df["text_cleaned"] = df["text_combined"].map(cleaner)
    df["text_cleaned"] = df["text_cleaned"].map(clean_text_value)
    before_drop = len(df)
    df = df[df["text_cleaned"] != ""].reset_index(drop=True)
    dropped = before_drop - len(df)

    if dropped:
        logger.info("Rows removed after text cleaning: %s", f"{dropped:,}")

    if df.empty:
        raise ValueError("No usable rows remain after text cleaning.")

    return df


def create_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
    ngram_range: tuple[int, int],
) -> tuple[csr_matrix, csr_matrix, TfidfVectorizer, list[str]]:
    """Fit TF-IDF on train only, transform train/test, and append stat features."""
    logger.info(
        "Fitting TF-IDF on training set only: max_features=%s, ngram_range=%s",
        max_features,
        ngram_range,
    )
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_train_tfidf = vectorizer.fit_transform(train_df["text_cleaned"])
    x_test_tfidf = vectorizer.transform(test_df["text_cleaned"])

    train_stats, stat_feature_names = create_statistical_features(
        train_df["text_combined"]
    )
    test_stats, _ = create_statistical_features(test_df["text_combined"])

    x_train = hstack([x_train_tfidf, csr_matrix(train_stats)], format="csr")
    x_test = hstack([x_test_tfidf, csr_matrix(test_stats)], format="csr")

    feature_names = list(vectorizer.get_feature_names_out()) + stat_feature_names

    logger.info("Training feature matrix shape: %s", x_train.shape)
    logger.info("Test feature matrix shape: %s", x_test.shape)

    return x_train, x_test, vectorizer, feature_names


def evaluate_model(model, x_test, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate one trained model."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    logger.info(
        "%s metrics: accuracy=%.4f, precision=%.4f, recall=%.4f, "
        "f1=%.4f, roc_auc=%.4f, pr_auc=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )

    return metrics


def build_error_analysis(
    model,
    x_test,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build false-positive/false-negative rows for Logistic Regression."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    error_mask = y_pred != y_test

    error_df = pd.DataFrame(
        {
            "text_combined": test_df.loc[error_mask, "text_combined"].to_numpy(),
            "true_label": y_test[error_mask],
            "predicted_label": y_pred[error_mask],
            "phishing_probability": y_proba[error_mask],
        }
    )

    if error_df.empty:
        error_df["error_type"] = pd.Series(dtype=str)
        return error_df[
            [
                "text_combined",
                "true_label",
                "predicted_label",
                "phishing_probability",
                "error_type",
            ]
        ]

    error_df["error_type"] = np.where(
        (error_df["true_label"] == 0) & (error_df["predicted_label"] == 1),
        "false_positive",
        "false_negative",
    )

    return error_df[
        [
            "text_combined",
            "true_label",
            "predicted_label",
            "phishing_probability",
            "error_type",
        ]
    ]


def save_artifacts(
    logistic_model,
    naive_bayes_model,
    vectorizer: TfidfVectorizer,
    feature_names: list[str],
    models_dir: Path,
) -> None:
    """Save model artifacts using filenames consumed by app.py."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(logistic_model, models_dir / "logistic_regression_model.pkl")
    joblib.dump(naive_bayes_model, models_dir / "naive_bayes_model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")
    joblib.dump(feature_names, models_dir / "feature_names.pkl")

    logger.info("Saved Logistic Regression model to %s", models_dir / "logistic_regression_model.pkl")
    logger.info("Saved Naive Bayes model to %s", models_dir / "naive_bayes_model.pkl")
    logger.info("Saved TF-IDF vectorizer to %s", models_dir / "tfidf_vectorizer.pkl")
    logger.info("Saved feature names to %s", models_dir / "feature_names.pkl")


def train_from_standardized(args: argparse.Namespace) -> None:
    """Run the complete training workflow from standardized data."""
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    df = load_standardized_dataset(args.input)
    df = add_cleaned_text(df)

    logger.info("Creating stratified train/test split before fitting TF-IDF...")
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info("Training rows: %s", f"{len(train_df):,}")
    logger.info("Test rows: %s", f"{len(test_df):,}")

    x_train, x_test, vectorizer, feature_names = create_features(
        train_df,
        test_df,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    logger.info("Training Naive Bayes baseline...")
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train, y_train)

    logger.info("Training Logistic Regression model...")
    logistic_model = LogisticRegression(
        class_weight="balanced",
        max_iter=args.logistic_max_iter,
        random_state=args.random_state,
        solver="liblinear",
    )
    logistic_model.fit(x_train, y_train)

    metrics = [
        evaluate_model(naive_bayes_model, x_test, y_test, "naive_bayes"),
        evaluate_model(logistic_model, x_test, y_test, "logistic_regression"),
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_path = args.results_dir / "metrics_from_standardized.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)

    error_df = build_error_analysis(logistic_model, x_test, y_test, test_df)
    error_path = (
        args.results_dir
        / "logistic_regression_error_analysis_from_standardized.csv"
    )
    error_df.to_csv(error_path, index=False)
    logger.info("Saved Logistic Regression error analysis to %s", error_path)

    save_artifacts(
        logistic_model,
        naive_bayes_model,
        vectorizer,
        feature_names,
        args.models_dir,
    )

    logger.info("Training workflow completed successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train phishing detection models from a standardized CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to data/processed/phishing_email_standardized.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for metrics and error analysis outputs.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for saved model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    train_from_standardized(args)


if __name__ == "__main__":
    main()
