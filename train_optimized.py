"""Optimized, course-aligned training pipeline for phishing email detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.text_cleaning import clean_email_text


DEFAULT_INPUT = Path("data/processed/phishing_email_standardized.csv")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MODELS_DIR = Path("models")
RANDOM_STATE = 42
TEST_SIZE = 0.2


MODEL_FILENAMES = {
    "naive_bayes": "naive_bayes_pipeline.pkl",
    "logistic_regression": "logistic_regression_pipeline.pkl",
    "linear_svm": "linear_svm_pipeline.pkl",
}


def read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    """Read a CSV with common encoding fallbacks."""
    last_error: Exception | None = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{path} is empty") from exc
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not read {path}: {last_error}")


def load_training_data(input_path: Path) -> pd.DataFrame:
    """Load standardized data and keep only usable rows."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Standardized dataset not found at {input_path}. "
            "Run standardize_datasets.py first."
        )

    df = read_csv_with_fallbacks(input_path)
    required = {"text_combined", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows_loaded = len(df)
    df = df.copy()
    df["text_combined"] = df["text_combined"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df = df[df["text_combined"] != ""]
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows found in the standardized dataset.")

    class_counts = df["label"].value_counts().sort_index()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Training requires at least two rows from each class.")

    print(f"Rows loaded: {rows_loaded:,}")
    print(f"Rows usable: {len(df):,}")
    print("Class distribution:")
    print(class_counts.to_string())
    return df


def make_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used by all model pipelines."""
    return TfidfVectorizer(
        preprocessor=clean_email_text,
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Create course-aligned sklearn pipelines."""
    return {
        "naive_bayes": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", MultinomialNB(alpha=1.0)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def get_model_scores(model: Pipeline, x_test: pd.Series) -> np.ndarray | None:
    """Return probability or decision scores for ROC/PR metrics."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None


def normalized_score_to_probability(scores: np.ndarray) -> np.ndarray:
    """Normalize arbitrary decision scores into a 0-1 range for reporting."""
    scores = np.asarray(scores, dtype=float)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score == min_score:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def evaluate_model(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict[str, object]:
    """Compute metrics for a fitted model."""
    y_pred = model.predict(x_test)
    scores = get_model_scores(model, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, object] = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
        metrics["pr_auc"] = average_precision_score(y_test, scores)

    print(
        f"{name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )
    return metrics


def save_confusion_matrix(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    """Save a confusion matrix plot for one model."""
    y_pred = model.predict(x_test)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["Ham", "Phishing"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)


def save_roc_curve(models: dict[str, Pipeline], x_test: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    """Save ROC curve comparison."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        scores = get_model_scores(model, x_test)
        if scores is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pr_curve(models: dict[str, Pipeline], x_test: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    """Save precision-recall curve comparison."""
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = float(pd.Series(y_test).mean())
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label=f"Baseline={baseline:.3f}")

    for name, model in models.items():
        scores = get_model_scores(model, x_test)
        if scores is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{name.replace('_', ' ')} AP={ap:.3f}")

    ax.set_title("Precision-Recall Curve Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_analysis(model: Pipeline, x_test: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    """Save Logistic Regression false positives and false negatives."""
    y_pred = model.predict(x_test)
    scores = get_model_scores(model, x_test)
    if scores is None:
        score_values = np.full(len(y_pred), np.nan)
    elif hasattr(model, "predict_proba"):
        score_values = scores
    else:
        score_values = normalized_score_to_probability(scores)

    error_mask = y_pred != y_test.to_numpy()
    errors = pd.DataFrame(
        {
            "text_combined": x_test.loc[error_mask].to_numpy(),
            "true_label": y_test.loc[error_mask].to_numpy(),
            "predicted_label": y_pred[error_mask],
            "phishing_probability_or_score": score_values[error_mask],
        }
    )
    errors["error_type"] = np.where(
        (errors["true_label"] == 0) & (errors["predicted_label"] == 1),
        "false_positive",
        "false_negative",
    )
    errors.to_csv(output_path, index=False)


def train_optimized(input_path: Path, results_dir: Path, models_dir: Path) -> None:
    """Run the optimized training workflow."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(input_path)
    x = df["text_combined"]
    y = df["label"]

    print("Splitting train/test before fitting TF-IDF...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")

    pipelines = build_pipelines()
    metrics = []

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(x_train, y_train)
        metrics.append(evaluate_model(name, pipeline, x_test, y_test))

        model_path = models_dir / MODEL_FILENAMES[name]
        joblib.dump(pipeline, model_path)
        print(f"Saved {name} pipeline to {model_path}")

    metrics_path = results_dir / "metrics_optimized.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    save_confusion_matrix(
        "naive_bayes",
        pipelines["naive_bayes"],
        x_test,
        y_test,
        results_dir / "naive_bayes_confusion_matrix.png",
    )
    save_confusion_matrix(
        "logistic_regression",
        pipelines["logistic_regression"],
        x_test,
        y_test,
        results_dir / "logistic_regression_confusion_matrix.png",
    )
    save_confusion_matrix(
        "linear_svm",
        pipelines["linear_svm"],
        x_test,
        y_test,
        results_dir / "linear_svm_confusion_matrix.png",
    )
    save_roc_curve(pipelines, x_test, y_test, results_dir / "roc_curve_comparison.png")
    save_pr_curve(pipelines, x_test, y_test, results_dir / "pr_curve_comparison.png")
    save_error_analysis(
        pipelines["logistic_regression"],
        x_test,
        y_test,
        results_dir / "logistic_regression_error_analysis.csv",
    )
    print(f"Saved error analysis to {results_dir / 'logistic_regression_error_analysis.csv'}")
    print("Optimized training complete.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train optimized phishing email classifiers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    train_optimized(args.input, args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
