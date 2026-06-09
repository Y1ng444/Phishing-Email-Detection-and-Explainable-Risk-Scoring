"""Evaluation, metrics, and error analysis module."""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
)

from .utils import setup_logger, RESULTS_DIR

logger = setup_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    # Calculate specificity (true negative rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """Print metrics in readable format.

    Args:
        metrics: Metrics dictionary from compute_metrics()
        model_name: Name of model
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{model_name} EVALUATION METRICS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {metrics['true_negatives']}")
    logger.info(f"  False Positives: {metrics['false_positives']}")
    logger.info(f"  False Negatives: {metrics['false_negatives']}")
    logger.info(f"  True Positives:  {metrics['true_positives']}")
    logger.info(f"{'=' * 60}")


def plot_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png") -> None:
    """Generate and save confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        filename: Output filename
    """
    logger.info("Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Ham", "Phishing"],
        yticklabels=["Ham", "Phishing"],
        ax=ax,
    )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_precision_recall_curve(
    y_true, y_proba, filename: str = "pr_curve.png"
) -> None:
    """Generate and save precision-recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        filename: Output filename
    """
    logger.info("Generating precision-recall curve...")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        recall, precision, color="#2c3e50", linewidth=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})"
    )
    ax.fill_between(recall, precision, alpha=0.2, color="#3498db")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {filename}")
    plt.close()


def analyze_errors(
    df: pd.DataFrame,
    y_pred,
    y_proba,
    filename: str = "error_analysis.csv",
) -> pd.DataFrame:
    """Identify and export false positives and false negatives.

    Args:
        df: Dataframe with text and labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        filename: Output CSV filename

    Returns:
        Error analysis dataframe
    """
    logger.info("Performing error analysis...")

    error_df = df.copy()
    error_df["predicted_label"] = y_pred
    error_df["probability"] = y_proba
    error_df["is_error"] = error_df["label"] != error_df["predicted_label"]

    # Get false positives and false negatives
    false_positives = error_df[(error_df["label"] == 0) & (error_df["predicted_label"] == 1)]
    false_negatives = error_df[(error_df["label"] == 1) & (error_df["predicted_label"] == 0)]

    logger.info(f"\nError Summary:")
    logger.info(f"  False Positives (Ham classified as Phishing): {len(false_positives)}")
    logger.info(f"  False Negatives (Phishing classified as Ham): {len(false_negatives)}")

    # Save errors
    errors_only = error_df[error_df["is_error"]].copy()
    errors_only = errors_only[["text_combined", "label", "predicted_label", "probability"]]
    errors_only.columns = ["text_combined", "true_label", "predicted_label", "probability"]

    output_path = RESULTS_DIR / filename
    errors_only.to_csv(output_path, index=False)
    logger.info(f"Error analysis saved to {filename} ({len(errors_only)} errors)")

    return errors_only


def print_classification_report(y_true, y_pred) -> None:
    """Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    logger.info("\n" + classification_report(y_true, y_pred, target_names=["Ham", "Phishing"]))


def run_evaluation(
    model,
    X_test,
    y_test,
    df_test: pd.DataFrame,
    model_name: str = "Model",
) -> dict:
    """Run complete evaluation pipeline.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        df_test: Test dataframe (for error analysis)
        model_name: Model name for logging

    Returns:
        Dictionary of all metrics and results
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"EVALUATION: {model_name}")
    logger.info("=" * 70)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, model_name)

    # Generate visualizations
    plot_confusion_matrix(y_test, y_pred, f"{model_name.lower()}_confusion_matrix.png")
    plot_precision_recall_curve(y_test, y_proba, f"{model_name.lower()}_pr_curve.png")

    # Classification report
    print_classification_report(y_test, y_pred)

    # Error analysis
    errors_df = analyze_errors(
        df_test, y_pred, y_proba, f"{model_name.lower()}_errors.csv"
    )

    results = {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_proba,
        "errors": errors_df,
    }

    return results
