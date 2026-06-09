"""End-to-end training orchestration script."""

import logging
from sklearn.model_selection import train_test_split

from src.utils import (
    setup_logger,
    set_seed,
    RANDOM_SEED,
    TEST_SIZE,
    MODELS_DIR,
)
from src.preprocessing import (
    load_dataset,
    explore_dataset,
    visualize_eda,
    preprocess_dataset,
)
from src.feature_engineering import create_all_features, save_vectorizer
from src.train_baseline import train_baseline_model, evaluate_model as eval_baseline
from src.train_logistic import train_logistic_model, evaluate_and_compare
from src.explainability import display_feature_importance, get_email_explanation, display_email_explanation
from src.evaluate import run_evaluation
import joblib

logger = setup_logger(__name__)


def main():
    """Execute complete training pipeline."""
    logger.info("=" * 80)
    logger.info("PHISHING EMAIL DETECTION - COMPLETE PIPELINE")
    logger.info("=" * 80)

    # Set seed
    set_seed(RANDOM_SEED)

    # Phase 1: Load and explore
    logger.info("\n[PHASE 1] Loading and exploring dataset...")
    df, total_rows = load_dataset()
    explore_dataset(df)
    visualize_eda(df)

    # Phase 2: Preprocess
    logger.info("\n[PHASE 2] Preprocessing data...")
    df = preprocess_dataset(df)

    # Phase 3: Feature engineering
    logger.info("\n[PHASE 3] Creating features...")
    features_dict = create_all_features(df["text_cleaned"], fit_vectorizer=True)
    X = features_dict["X"]
    y = df["label"].values

    # Save vectorizer
    save_vectorizer(features_dict["vectorizer"], "tfidf_vectorizer.pkl")

    # Save feature names
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    joblib.dump(features_dict["feature_names"], feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")

    # Phase 4: Train-test split
    logger.info("\n[PHASE 4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Also split original dataframe for error analysis
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Phase 5: Train baseline
    logger.info("\n[PHASE 5] Training baseline model (Multinomial Naive Bayes)...")
    baseline_model = train_baseline_model(X_train, y_train)
    baseline_path = MODELS_DIR / "naive_bayes_model.pkl"
    joblib.dump(baseline_model, baseline_path)
    logger.info(f"Baseline model saved to {baseline_path}")

    baseline_results = run_evaluation(
        baseline_model, X_test, y_test, df_test, "Multinomial Naive Bayes"
    )

    # Phase 6: Train improved model
    logger.info("\n[PHASE 6] Training improved model (Logistic Regression)...")
    logistic_model = train_logistic_model(X_train, y_train)
    logistic_path = MODELS_DIR / "logistic_regression_model.pkl"
    joblib.dump(logistic_model, logistic_path)
    logger.info(f"Logistic Regression model saved to {logistic_path}")

    logistic_results = run_evaluation(
        logistic_model, X_test, y_test, df_test, "Logistic Regression"
    )

    # Phase 7: Compare models
    logger.info("\n[PHASE 7] Comparing models...")
    evaluate_and_compare(baseline_results["metrics"], logistic_results["metrics"])

    # Phase 8: Explainability
    logger.info("\n[PHASE 8] Generating feature importance...")
    display_feature_importance(logistic_model, features_dict["feature_names"], n=20)

    # Example explanation for a sample phishing email
    logger.info("\n[PHASE 9] Generating sample email explanation...")
    sample_email = df_test.iloc[0]["text_combined"]
    explanation = get_email_explanation(
        sample_email,
        logistic_model,
        features_dict["vectorizer"],
        features_dict["feature_names"],
    )
    display_email_explanation(explanation)

    # Phase 10: Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {MODELS_DIR}")
    logger.info(f"Results saved to: {MODELS_DIR.parent / 'results'}")
    logger.info(f"\nBest Model: Logistic Regression")
    logger.info(f"  Accuracy:  {logistic_results['metrics']['accuracy']:.4f}")
    logger.info(f"  Precision: {logistic_results['metrics']['precision']:.4f}")
    logger.info(f"  Recall:    {logistic_results['metrics']['recall']:.4f}")
    logger.info(f"  F1 Score:  {logistic_results['metrics']['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {logistic_results['metrics']['roc_auc']:.4f}")
    logger.info("\nTo run the Streamlit app:")
    logger.info("  streamlit run app.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
