"""Visualization module for comprehensive analysis and reporting."""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import setup_logger, RESULTS_DIR

logger = setup_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 25,
    filename: str = "feature_importance.png",
) -> None:
    """Plot top N most important features.

    Args:
        model: Logistic Regression model with coef_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        filename: Output filename
    """
    logger.info(f"Generating feature importance plot (top {top_n})...")

    coefficients = model.coef_[0]
    indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]

    top_features = [feature_names[i] for i in indices]
    top_coefs = [coefficients[i] for i in indices]
    colors = ['#e74c3c' if coef > 0 else '#2ecc71' for coef in top_coefs]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_coefs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features (Logistic Regression)', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Phishing Indicator'),
                      Patch(facecolor='#2ecc71', label='Ham Indicator')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_top_phishing_keywords(
    feature_names: List[str],
    coefficients: np.ndarray,
    top_n: int = 20,
    filename: str = "top_phishing_keywords.png",
) -> None:
    """Plot top phishing indicator keywords.

    Args:
        feature_names: List of feature names
        coefficients: Model coefficients
        top_n: Number of top keywords to display
        filename: Output filename
    """
    logger.info(f"Generating top phishing keywords plot (top {top_n})...")

    indices = np.argsort(coefficients)[-top_n:][::-1]
    top_keywords = [feature_names[i] for i in indices]
    top_coefs = [coefficients[i] for i in indices]

    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(top_keywords))
    ax.bar(x_pos, top_coefs, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_keywords, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Phishing Indicator Keywords', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(top_coefs):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_top_ham_keywords(
    feature_names: List[str],
    coefficients: np.ndarray,
    top_n: int = 20,
    filename: str = "top_ham_keywords.png",
) -> None:
    """Plot top legitimate email (ham) indicator keywords.

    Args:
        feature_names: List of feature names
        coefficients: Model coefficients
        top_n: Number of top keywords to display
        filename: Output filename
    """
    logger.info(f"Generating top ham keywords plot (top {top_n})...")

    indices = np.argsort(coefficients)[:top_n]
    top_keywords = [feature_names[i] for i in indices]
    top_coefs = [coefficients[i] for i in indices]

    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(top_keywords))
    ax.bar(x_pos, np.abs(top_coefs), color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_keywords, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient Value (Absolute)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Legitimate Email (Ham) Indicator Keywords', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(top_coefs):
        ax.text(i, abs(v) + 0.1, f'{abs(v):.2f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_model_comparison(
    baseline_metrics: Dict[str, float],
    logistic_metrics: Dict[str, float],
    filename: str = "model_comparison.png",
) -> None:
    """Create side-by-side model comparison bar chart.

    Args:
        baseline_metrics: Baseline model metrics
        logistic_metrics: Logistic Regression metrics
        filename: Output filename
    """
    logger.info("Generating model comparison plot...")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    baseline_values = [baseline_metrics[m] for m in metrics]
    logistic_values = [logistic_metrics[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(metrics))
    width = 0.35

    ax.bar(x_pos - width/2, baseline_values, width, label='Naive Bayes',
           color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, logistic_values, width, label='Logistic Regression',
           color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim([0.8, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bv, lv) in enumerate(zip(baseline_values, logistic_values)):
        ax.text(i - width/2, bv + 0.005, f'{bv:.4f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(i + width/2, lv + 0.005, f'{lv:.4f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_cross_validation_scores(
    cv_results: Dict[str, np.ndarray],
    model_name: str = "Model",
    filename: str = "cv_scores.png",
) -> None:
    """Plot cross-validation score distributions.

    Args:
        cv_results: Cross-validation results from cross_validate()
        model_name: Name of model
        filename: Output filename
    """
    logger.info("Generating cross-validation scores plot...")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        key = f'test_{metric}'
        scores = cv_results[key]

        axes[idx].boxplot(scores, vert=True)
        axes[idx].scatter(np.ones(len(scores)), scores, alpha=0.5, s=50, color='#e74c3c')
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(f'{metric.title()} (μ={np.mean(scores):.4f}, σ={np.std(scores):.4f})',
                           fontsize=10, fontweight='bold')
        axes[idx].set_xticklabels(['CV Folds'])
        axes[idx].grid(axis='y', alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    fig.suptitle(f'Cross-Validation Score Distributions - {model_name}',
                fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_roc_curves(
    baseline_model,
    logistic_model,
    X_test,
    y_test,
    filename: str = "roc_curves_comparison.png",
) -> None:
    """Plot ROC curves for both models.

    Args:
        baseline_model: Trained baseline model
        logistic_model: Trained logistic regression model
        X_test: Test features
        y_test: Test labels
        filename: Output filename
    """
    logger.info("Generating ROC curves comparison...")

    from sklearn.metrics import roc_curve, auc, roc_auc_score

    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    logistic_proba = logistic_model.predict_proba(X_test)[:, 1]

    baseline_fpr, baseline_tpr, _ = roc_curve(y_test, baseline_proba)
    baseline_auc = auc(baseline_fpr, baseline_tpr)

    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_proba)
    logistic_auc = auc(logistic_fpr, logistic_tpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(baseline_fpr, baseline_tpr, color='#3498db', lw=2.5,
           label=f'Naive Bayes (AUC = {baseline_auc:.4f})')
    ax.plot(logistic_fpr, logistic_tpr, color='#e74c3c', lw=2.5,
           label=f'Logistic Regression (AUC = {logistic_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def plot_learning_curves(
    model,
    X_train,
    y_train,
    model_name: str = "Model",
    filename: str = "learning_curves.png",
) -> None:
    """Plot learning curves to analyze model performance vs training size.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        model_name: Name of model
        filename: Output filename
    """
    logger.info("Generating learning curves...")

    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_mean, 'o-', color='#3498db', lw=2, label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#3498db')

    ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', lw=2, label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='#e74c3c')

    ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Learning Curves - {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {filename}")
    plt.close()


def create_metrics_table(
    baseline_metrics: Dict[str, float],
    logistic_metrics: Dict[str, float],
    cv_baseline: Dict[str, np.ndarray],
    cv_logistic: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Create comprehensive metrics comparison table.

    Args:
        baseline_metrics: Baseline model test metrics
        logistic_metrics: Logistic regression test metrics
        cv_baseline: Baseline cross-validation results
        cv_logistic: Logistic regression cross-validation results

    Returns:
        DataFrame with metrics comparison
    """
    logger.info("Creating metrics comparison table...")

    rows = []

    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for metric in metrics_list:
        row = {
            'Metric': metric.replace('_', ' ').title(),
            'Naive Bayes (Test)': f"{baseline_metrics[metric]:.4f}",
            'Naive Bayes (CV Mean)': f"{np.mean(cv_baseline[f'test_{metric}']):.4f} ± {np.std(cv_baseline[f'test_{metric}']):.4f}",
            'Logistic Reg (Test)': f"{logistic_metrics[metric]:.4f}",
            'Logistic Reg (CV Mean)': f"{np.mean(cv_logistic[f'test_{metric}']):.4f} ± {np.std(cv_logistic[f'test_{metric}']):.4f}",
            'Improvement': f"{(logistic_metrics[metric] - baseline_metrics[metric]):+.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
