"""Advanced ML techniques: SMOTE, cross-validation, and hyperparameter tuning."""

import logging
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

from .utils import (
    setup_logger,
    RANDOM_SEED,
    CV_FOLDS,
    CV_RANDOM_STATE,
    SMOTE_RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    SMOTE_SAMPLING_STRATEGY,
    LR_PARAM_GRID,
    NB_PARAM_GRID,
    MODELS_DIR,
)

logger = setup_logger(__name__)


def apply_smote(X_train, y_train) -> Tuple[Any, Any]:
    """Apply SMOTE to handle class imbalance in training data.

    Args:
        X_train: Training features (sparse matrix)
        y_train: Training labels

    Returns:
        Tuple of (X_train_smote, y_train_smote)
    """
    logger.info("Applying SMOTE for class imbalance handling...")

    # SMOTE requires dense matrix for some operations, but works with sparse
    smote = SMOTE(
        random_state=SMOTE_RANDOM_STATE,
        k_neighbors=SMOTE_K_NEIGHBORS,
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    )

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    logger.info(f"Original distribution: {np.bincount(y_train)}")
    logger.info(f"SMOTE distribution:  {np.bincount(y_train_smote)}")
    logger.info(f"Training set size: {X_train.shape[0]} → {X_train_smote.shape[0]}")

    return X_train_smote, y_train_smote


def get_stratified_kfold() -> StratifiedKFold:
    """Get StratifiedKFold cross-validator for stratified splits.

    Returns:
        StratifiedKFold instance
    """
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)


def cross_validate_model(
    model, X: Any, y: np.ndarray, cv: int = CV_FOLDS
) -> Dict[str, Any]:
    """Perform cross-validation on model.

    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Labels
        cv: Number of folds

    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation...")

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True
    )

    logger.info(f"Cross-validation completed")
    logger.info(
        f"  Mean Accuracy:  {cv_results['test_accuracy'].mean():.4f} "
        f"(±{cv_results['test_accuracy'].std():.4f})"
    )
    logger.info(
        f"  Mean Precision: {cv_results['test_precision'].mean():.4f} "
        f"(±{cv_results['test_precision'].std():.4f})"
    )
    logger.info(
        f"  Mean Recall:    {cv_results['test_recall'].mean():.4f} "
        f"(±{cv_results['test_recall'].std():.4f})"
    )
    logger.info(
        f"  Mean F1 Score:  {cv_results['test_f1'].mean():.4f} "
        f"(±{cv_results['test_f1'].std():.4f})"
    )
    logger.info(
        f"  Mean ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} "
        f"(±{cv_results['test_roc_auc'].std():.4f})"
    )

    return cv_results


def tune_logistic_regression(
    X_train, y_train, cv: int = CV_FOLDS
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Perform hyperparameter tuning for Logistic Regression using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best_model, grid_search_results)
    """
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER TUNING: Logistic Regression (GridSearchCV)")
    logger.info("=" * 70)

    logger.info(f"Parameter grid: {LR_PARAM_GRID}")

    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=RANDOM_SEED),
        param_grid=LR_PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"\nBest parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    results = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "best_model": grid_search.best_estimator_,
    }

    return grid_search.best_estimator_, results


def tune_naive_bayes(
    X_train, y_train, cv: int = CV_FOLDS
) -> Tuple[MultinomialNB, Dict[str, Any]]:
    """Perform hyperparameter tuning for Naive Bayes using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best_model, grid_search_results)
    """
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER TUNING: Multinomial Naive Bayes (GridSearchCV)")
    logger.info("=" * 70)

    logger.info(f"Parameter grid: {NB_PARAM_GRID}")

    grid_search = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=NB_PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"\nBest parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    results = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "best_model": grid_search.best_estimator_,
    }

    return grid_search.best_estimator_, results


def save_tuning_results(
    tuning_results: Dict[str, Any], filename: str = "tuning_results.pkl"
) -> None:
    """Save hyperparameter tuning results to disk.

    Args:
        tuning_results: Dictionary from tune_*() functions
        filename: Output filename
    """
    path = MODELS_DIR / filename
    joblib.dump(tuning_results, path)
    logger.info(f"Tuning results saved to {path}")


def load_tuning_results(
    filename: str = "tuning_results.pkl"
) -> Dict[str, Any]:
    """Load hyperparameter tuning results from disk.

    Args:
        filename: Input filename

    Returns:
        Tuning results dictionary

    Raises:
        FileNotFoundError: If file not found
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Tuning results not found at {path}")

    results = joblib.load(path)
    logger.info(f"Tuning results loaded from {path}")
    return results
