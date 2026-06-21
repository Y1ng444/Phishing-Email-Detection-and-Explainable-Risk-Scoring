"""Train course-aligned phishing email classifiers and save final outputs."""

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

from src.explainability import save_global_indicators
from src.risk_scoring import risk_level_from_score
from src.text_cleaning import clean_text


DEFAULT_INPUT = Path("data/processed/phishing_email_standardized.csv")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MODELS_DIR = Path("models")
FINAL_MODEL_NAME = "phishing_logreg_tfidf.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_training_data(input_path: Path) -> pd.DataFrame:
    """Load standardized text and labels."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Standardized dataset not found at {input_path}. "
            "Run python standardize_datasets.py first."
        )

    df = pd.read_csv(input_path, low_memory=False)
    required = {"text_combined", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text_combined"] = df["text_combined"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    df = df[df["text_combined"] != ""]
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows found in standardized data.")

    class_counts = df["label"].value_counts().sort_index()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Training requires at least two examples from each class.")

    print(f"Rows loaded for training: {len(df):,}")
    print("Class distribution:")
    print(class_counts.to_string())
    return df


def make_vectorizer() -> TfidfVectorizer:
    """Create the shared TF-IDF unigram and bigram vectorizer."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_models() -> dict[str, Pipeline]:
    """Create the baseline, improved, and comparison model pipelines."""
    return {
        "naive_bayes": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", MultinomialNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
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
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def get_scores(model: Pipeline, texts: pd.Series) -> np.ndarray | None:
    """Return positive-class probabilities or decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    return None


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | int | str]:
    """Calculate requested metrics for one fitted model."""
    y_pred = model.predict(x_test)
    scores = get_scores(model, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float | int | str] = {
        "model": model_name,
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
        f"{model_name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )
    return metrics


def save_confusion_matrix(
    model_name: str,
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save a confusion matrix image for one model."""
    y_pred = model.predict(x_test)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"{model_name.replace('_', ' ').title()} Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)


def save_roc_curve(
    models: dict[str, Pipeline],
    x_test: pd.Series,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save a ROC curve comparison image."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, model in models.items():
        scores = get_scores(model, x_test)
        if scores is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{model_name.replace('_', ' ')} AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Random")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pr_curve(
    models: dict[str, Pipeline],
    x_test: pd.Series,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save a precision-recall curve comparison image."""
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = float(pd.Series(y_test).mean())
    ax.axhline(
        baseline,
        linestyle="--",
        color="black",
        linewidth=1,
        label=f"Class balance={baseline:.3f}",
    )

    for model_name, model in models.items():
        scores = get_scores(model, x_test)
        if scores is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{model_name.replace('_', ' ')} PR-AUC={pr_auc:.3f}")

    ax.set_title("Precision-Recall Curve Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_analysis(
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save up to 10 false positives and 10 false negatives."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)

    errors = pd.DataFrame(
        {
            "text_combined": x_test.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "predicted_label": predictions,
            "phishing_probability": probabilities,
        }
    )
    errors = errors[errors["true_label"] != errors["predicted_label"]].copy()
    errors["error_type"] = np.where(
        (errors["true_label"] == 0) & (errors["predicted_label"] == 1),
        "false_positive",
        "false_negative",
    )

    false_positives = errors[errors["error_type"] == "false_positive"].head(10)
    false_negatives = errors[errors["error_type"] == "false_negative"].head(10)
    sampled_errors = pd.concat([false_positives, false_negatives], ignore_index=True)
    sampled_errors.to_csv(output_path, index=False)


def save_sample_predictions(
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
    output_path: Path,
    sample_size: int = 20,
) -> None:
    """Save a small sample of Logistic Regression predictions."""
    sample_texts = x_test.head(sample_size).reset_index(drop=True)
    true_labels = y_test.head(sample_size).reset_index(drop=True)
    probabilities = model.predict_proba(sample_texts)[:, 1]
    predictions = model.predict(sample_texts)
    risk_scores = probabilities * 100

    sample = pd.DataFrame(
        {
            "text_combined": sample_texts,
            "true_label": true_labels,
            "predicted_label": predictions,
            "phishing_probability": probabilities,
            "risk_score": risk_scores,
            "risk_level": [risk_level_from_score(score) for score in risk_scores],
        }
    )
    sample.to_csv(output_path, index=False)


def train_optimized(input_path: Path, results_dir: Path, models_dir: Path) -> None:
    """Run the full training and evaluation workflow."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(input_path)
    x = df["text_combined"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Train rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")

    models = build_models()
    metrics = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        metrics.append(evaluate_model(model_name, model, x_test, y_test))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(results_dir / "metrics_optimized.csv", index=False)

    save_confusion_matrix(
        "naive_bayes",
        models["naive_bayes"],
        x_test,
        y_test,
        results_dir / "naive_bayes_confusion_matrix.png",
    )
    save_confusion_matrix(
        "logistic_regression",
        models["logistic_regression"],
        x_test,
        y_test,
        results_dir / "logistic_regression_confusion_matrix.png",
    )
    save_confusion_matrix(
        "linear_svm",
        models["linear_svm"],
        x_test,
        y_test,
        results_dir / "linear_svm_confusion_matrix.png",
    )
    save_roc_curve(models, x_test, y_test, results_dir / "roc_curve_comparison.png")
    save_pr_curve(models, x_test, y_test, results_dir / "pr_curve_comparison.png")

    logistic_model = models["logistic_regression"]
    save_error_analysis(logistic_model, x_test, y_test, results_dir / "error_samples.csv")
    save_sample_predictions(logistic_model, x_test, y_test, results_dir / "sample_predictions.csv")
    save_global_indicators(logistic_model, results_dir, top_n=20)

    final_model_path = models_dir / FINAL_MODEL_NAME
    joblib.dump(logistic_model, final_model_path)

    best_model = metrics_df.sort_values("f1", ascending=False).iloc[0]["model"]
    print(f"Best F1 model: {best_model}")
    print("Selected risk scoring model: logistic_regression")
    print(f"Saved final Logistic Regression model to {final_model_path}")
    print("Training complete.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train phishing email classifiers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the training script."""
    args = parse_args()
    train_optimized(args.input, args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
