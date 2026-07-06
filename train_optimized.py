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
    if "source_file" not in df.columns:
        df["source_file"] = "unknown"
    else:
        df["source_file"] = df["source_file"].fillna("unknown").astype(str)
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
    has_both_classes = pd.Series(y_test).nunique() == 2

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

    if scores is not None and has_both_classes:
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
    print(f"Saved confusion matrix: {output_path}")


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
    print(f"Saved ROC curve comparison: {output_path}")


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
    print(f"Saved PR curve comparison: {output_path}")


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
    print(f"Saved Logistic Regression error samples: {output_path}")


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
    print(f"Saved sample Logistic Regression predictions: {output_path}")


def save_cleaned_duplicate_check(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save duplicate statistics using both raw and cleaned email text."""
    working = df.copy()
    working["text_cleaned"] = working["text_combined"].map(clean_text)
    working = working[working["text_cleaned"] != ""].copy()

    label_counts_per_cleaned_text = working.groupby("text_cleaned")["label"].nunique()
    conflicting_cleaned_texts = label_counts_per_cleaned_text[
        label_counts_per_cleaned_text > 1
    ].index
    conflicting_rows = int(working["text_cleaned"].isin(conflicting_cleaned_texts).sum())

    non_conflicting = working[~working["text_cleaned"].isin(conflicting_cleaned_texts)].copy()
    deduplicated = non_conflicting.drop_duplicates(subset=["text_cleaned"], keep="first")

    stats = pd.DataFrame(
        [
            {"metric": "rows_before_cleaned_dedup", "value": len(working)},
            {
                "metric": "duplicate_rows_text_combined",
                "value": int(working.duplicated(subset=["text_combined"]).sum()),
            },
            {
                "metric": "duplicate_rows_text_cleaned",
                "value": int(working.duplicated(subset=["text_cleaned"]).sum()),
            },
            {
                "metric": "conflicting_cleaned_texts",
                "value": int(len(conflicting_cleaned_texts)),
            },
            {
                "metric": "rows_removed_conflicting_cleaned_labels",
                "value": conflicting_rows,
            },
            {
                "metric": "rows_after_drop_duplicate_text_cleaned",
                "value": int(len(deduplicated)),
            },
        ]
    )
    stats.to_csv(output_path, index=False)
    print(f"Saved cleaned duplicate check: {output_path}")
    return stats


def make_cleaned_deduplicated_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataset deduplicated by cleaned text and safe for retraining."""
    working = df.copy()
    working["text_cleaned"] = working["text_combined"].map(clean_text)
    working = working[working["text_cleaned"] != ""].copy()

    label_counts_per_cleaned_text = working.groupby("text_cleaned")["label"].nunique()
    conflicting_cleaned_texts = label_counts_per_cleaned_text[
        label_counts_per_cleaned_text > 1
    ].index
    working = working[~working["text_cleaned"].isin(conflicting_cleaned_texts)].copy()
    working = working.drop_duplicates(subset=["text_cleaned"], keep="first").reset_index(drop=True)
    return working


def train_cleaned_dedup_benchmark(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Train the same model set on cleaned-text deduplicated data."""
    deduplicated = make_cleaned_deduplicated_dataset(df)
    class_counts = deduplicated["label"].value_counts().sort_index()

    if len(class_counts) < 2 or class_counts.min() < 2:
        skipped = pd.DataFrame(
            [
                {
                    "model": "all",
                    "status": "skipped",
                    "reason": "cleaned deduplicated data does not contain two valid classes",
                }
            ]
        )
        skipped.to_csv(results_dir / "metrics_cleaned_dedup.csv", index=False)
        print("Skipped cleaned-dedup benchmark: not enough data in both classes.")
        return skipped

    x = deduplicated["text_cleaned"]
    y = deduplicated["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nCleaned-text deduplicated benchmark")
    print(f"Rows: {len(deduplicated):,}")
    print(f"Train rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")

    models = build_models()
    metrics = []
    for model_name, model in models.items():
        print(f"Training cleaned-dedup {model_name}...")
        model.fit(x_train, y_train)
        metrics.append(evaluate_model(model_name, model, x_test, y_test))
        save_confusion_matrix(
            f"cleaned_dedup_{model_name}",
            model,
            x_test,
            y_test,
            results_dir / f"cleaned_dedup_{model_name}_confusion_matrix.png",
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_path = results_dir / "metrics_cleaned_dedup.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved cleaned-dedup metrics: {metrics_path}")
    return metrics_df


def save_near_duplicate_check(
    vectorizer: TfidfVectorizer,
    x_train: pd.Series,
    x_test: pd.Series,
    output_path: Path,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Measure max cosine similarity from each test email to the training set."""
    train_matrix = vectorizer.transform(x_train)
    test_matrix = vectorizer.transform(x_test)
    max_similarities = np.zeros(test_matrix.shape[0], dtype=float)

    for start in range(0, test_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, test_matrix.shape[0])
        similarities = test_matrix[start:end] @ train_matrix.T
        row_max = similarities.max(axis=1)
        if hasattr(row_max, "toarray"):
            row_max = row_max.toarray()
        max_similarities[start:end] = np.asarray(row_max).ravel()

    summary = pd.DataFrame(
        [
            {
                "train_rows": int(train_matrix.shape[0]),
                "test_rows": int(test_matrix.shape[0]),
                "mean_max_similarity": float(np.mean(max_similarities)),
                "median_max_similarity": float(np.median(max_similarities)),
                "p90_max_similarity": float(np.quantile(max_similarities, 0.90)),
                "p95_max_similarity": float(np.quantile(max_similarities, 0.95)),
                "count_similarity_gt_0_90": int(np.sum(max_similarities > 0.90)),
                "ratio_similarity_gt_0_90": float(np.mean(max_similarities > 0.90)),
                "count_similarity_gt_0_95": int(np.sum(max_similarities > 0.95)),
                "ratio_similarity_gt_0_95": float(np.mean(max_similarities > 0.95)),
            }
        ]
    )
    summary.to_csv(output_path, index=False)
    print(f"Saved near-duplicate similarity check: {output_path}")
    return summary


def _source_mask(df: pd.DataFrame, source_keywords: list[str]) -> pd.Series:
    """Return a mask matching source_file values by keyword."""
    source_text = df["source_file"].fillna("unknown").astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for keyword in source_keywords:
        mask = mask | source_text.str.contains(keyword.lower(), regex=False)
    return mask


def _base_holdout_row(
    experiment: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    status: str,
    note: str,
) -> dict[str, float | int | str]:
    """Create a source-holdout metrics row with common metadata."""
    return {
        "experiment": experiment,
        "model": "logistic_regression",
        "status": status,
        "note": note,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_legitimate": int((train_df["label"] == 0).sum()) if not train_df.empty else 0,
        "train_phishing": int((train_df["label"] == 1).sum()) if not train_df.empty else 0,
        "test_legitimate": int((test_df["label"] == 0).sum()) if not test_df.empty else 0,
        "test_phishing": int((test_df["label"] == 1).sum()) if not test_df.empty else 0,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "phishing_recall": np.nan,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }


def evaluate_source_holdout(
    experiment: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Train Logistic Regression on one source split and evaluate on another."""
    row = _base_holdout_row(experiment, train_df, test_df, "ok", "")

    if train_df.empty or test_df.empty:
        row["status"] = "skipped"
        row["note"] = "empty train or test split"
        return row

    train_counts = train_df["label"].value_counts()
    if len(train_counts) < 2 or train_counts.min() < 2:
        row["status"] = "skipped"
        row["note"] = "training split does not contain at least two examples per class"
        return row

    model = build_models()["logistic_regression"]
    x_train = train_df["text_combined"]
    y_train = train_df["label"]
    x_test = test_df["text_combined"]
    y_test = test_df["label"]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    row.update(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )

    if pd.Series(y_test).nunique() == 2:
        row.update(
            {
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, scores),
                "pr_auc": average_precision_score(y_test, scores),
                "phishing_recall": recall_score(y_test, y_pred, zero_division=0),
                "note": "two-class external test set",
            }
        )
    elif int(y_test.iloc[0]) == 1:
        phishing_recall = recall_score(y_test, y_pred, labels=[1], zero_division=0)
        row.update(
            {
                "recall": phishing_recall,
                "phishing_recall": phishing_recall,
                "note": "single-class phishing test set; reporting phishing recall",
            }
        )
    else:
        row["note"] = "single-class legitimate test set; full binary metrics are not defined"

    return row


def run_source_holdout_experiments(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Run source-based external validation experiments when source_file exists."""
    output_path = results_dir / "source_holdout_metrics.csv"

    if "source_file" not in df.columns:
        skipped = pd.DataFrame(
            [
                {
                    "experiment": "all",
                    "model": "logistic_regression",
                    "status": "skipped",
                    "note": "source_file column is missing",
                }
            ]
        )
        skipped.to_csv(output_path, index=False)
        print("Skipped source holdout experiments: source_file column is missing.")
        return skipped

    experiments = [
        ("ceas_enron_to_nazario", ["ceas", "enron"], ["nazario"]),
        ("ceas_to_enron", ["ceas"], ["enron"]),
        ("enron_to_ceas", ["enron"], ["ceas"]),
    ]

    rows = []
    for experiment, train_sources, test_sources in experiments:
        train_df = df[_source_mask(df, train_sources)].reset_index(drop=True)
        test_df = df[_source_mask(df, test_sources)].reset_index(drop=True)
        print(f"Running source holdout: {experiment}")
        rows.append(evaluate_source_holdout(experiment, train_df, test_df))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_path, index=False)
    print(f"Saved source holdout metrics: {output_path}")
    return metrics


def train_optimized(input_path: Path, results_dir: Path, models_dir: Path) -> None:
    """Run the full training and evaluation workflow."""
    input_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(input_path)
    if "source_file" in df.columns:
        print("Source distribution:")
        print(df["source_file"].value_counts().to_string())

    save_cleaned_duplicate_check(df, results_dir / "cleaned_duplicate_check.csv")

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
    metrics_path = results_dir / "metrics_optimized.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved model comparison metrics: {metrics_path}")

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
    save_near_duplicate_check(
        logistic_model.named_steps["tfidf"],
        x_train,
        x_test,
        results_dir / "near_duplicate_check.csv",
    )
    save_error_analysis(logistic_model, x_test, y_test, results_dir / "error_samples.csv")
    save_sample_predictions(logistic_model, x_test, y_test, results_dir / "sample_predictions.csv")
    save_global_indicators(logistic_model, results_dir, top_n=20)
    print(f"Saved global phishing indicators: {results_dir / 'top_phishing_indicators.csv'}")
    print(f"Saved global legitimate indicators: {results_dir / 'top_legitimate_indicators.csv'}")

    final_model_path = models_dir / FINAL_MODEL_NAME
    joblib.dump(logistic_model, final_model_path)

    best_model = metrics_df.sort_values("f1", ascending=False).iloc[0]["model"]
    print(f"Best F1 model: {best_model}")
    print("Selected final explainable risk scoring model: logistic_regression")
    print(f"Saved final Logistic Regression model: {final_model_path}")

    train_cleaned_dedup_benchmark(df, results_dir)
    run_source_holdout_experiments(df, results_dir)

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
