"""End-to-end training orchestration script with advanced ML techniques."""

import logging
from sklearn.model_selection import train_test_split

from src.utils import (
    setup_logger,
    set_seed,
    RANDOM_SEED,
    TEST_SIZE,
    MODELS_DIR,
    CV_FOLDS,
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
from src.advanced_ml import (
    apply_smote,
    cross_validate_model,
    tune_logistic_regression,
    tune_naive_bayes,
    save_tuning_results,
)
from src.visualization import (
    plot_feature_importance,
    plot_top_phishing_keywords,
    plot_top_ham_keywords,
    plot_model_comparison,
    plot_cross_validation_scores,
    plot_roc_curves,
    plot_learning_curves,
    create_metrics_table,
)
import joblib

logger = setup_logger(__name__)


def main():
    """Execute complete training pipeline with advanced techniques."""
    logger.info("=" * 80)
    logger.info("PHISHING EMAIL DETECTION - ADVANCED PIPELINE WITH SMOTE & TUNING")
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
    logger.info("\n[PHASE 4] Splitting data with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Also split original dataframe for error analysis
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Phase 5: Apply SMOTE
    logger.info("\n[PHASE 5] Applying SMOTE for class imbalance handling...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Phase 6: Cross-validation on original training data
    logger.info("\n[PHASE 6] Performing cross-validation on ORIGINAL training data...")
    baseline_model_cv = train_baseline_model(X_train, y_train)
    cv_baseline_results = cross_validate_model(baseline_model_cv, X_train, y_train, cv=CV_FOLDS)

    logistic_model_cv = train_logistic_model(X_train, y_train)
    cv_logistic_results = cross_validate_model(logistic_model_cv, X_train, y_train, cv=CV_FOLDS)

    # Phase 7: Hyperparameter tuning
    logger.info("\n[PHASE 7] Hyperparameter tuning with GridSearchCV...")
    tuned_nb, nb_tuning = tune_naive_bayes(X_train_smote, y_train_smote, cv=CV_FOLDS)
    tuned_lr, lr_tuning = tune_logistic_regression(X_train_smote, y_train_smote, cv=CV_FOLDS)

    save_tuning_results(nb_tuning, "nb_tuning_results.pkl")
    save_tuning_results(lr_tuning, "lr_tuning_results.pkl")

    # Phase 8: Train final models on SMOTE-balanced data
    logger.info("\n[PHASE 8] Training final models on SMOTE-balanced data...")
    baseline_model_final = train_baseline_model(X_train_smote, y_train_smote)
    logistic_model_final = train_logistic_model(X_train_smote, y_train_smote)

    # Save final models
    baseline_path = MODELS_DIR / "naive_bayes_model_final.pkl"
    joblib.dump(baseline_model_final, baseline_path)
    logger.info(f"Final baseline model saved to {baseline_path}")

    logistic_path = MODELS_DIR / "logistic_regression_model_final.pkl"
    joblib.dump(logistic_model_final, logistic_path)
    logger.info(f"Final logistic regression model saved to {logistic_path}")

    # Phase 9: Evaluate on test set
    logger.info("\n[PHASE 9] Evaluating final models on test set...")
    baseline_results = run_evaluation(
        baseline_model_final, X_test, y_test, df_test, "Naive Bayes (Final)"
    )
    logistic_results = run_evaluation(
        logistic_model_final, X_test, y_test, df_test, "Logistic Regression (Final)"
    )

    # Phase 10: Model comparison
    logger.info("\n[PHASE 10] Comparing models...")
    evaluate_and_compare(baseline_results["metrics"], logistic_results["metrics"])

    # Phase 11: Explainability
    logger.info("\n[PHASE 11] Generating feature importance and explanations...")
    display_feature_importance(logistic_model_final, features_dict["feature_names"], n=20)

    # Example explanation
    if len(df_test) > 0:
        logger.info("\nGenerating sample email explanation...")
        sample_email = df_test.iloc[0]["text_combined"]
        explanation = get_email_explanation(
            sample_email,
            logistic_model_final,
            features_dict["vectorizer"],
            features_dict["feature_names"],
        )
        display_email_explanation(explanation)

    # Phase 12: Generate comprehensive visualizations
    logger.info("\n[PHASE 12] Generating comprehensive visualizations...")

    # Feature importance
    plot_feature_importance(
        logistic_model_final,
        features_dict["feature_names"],
        top_n=25,
        filename="03_feature_importance_top25.png",
    )

    # Top keywords
    coefficients = logistic_model_final.coef_[0]
    plot_top_phishing_keywords(
        features_dict["feature_names"],
        coefficients,
        top_n=15,
        filename="04_top_phishing_keywords.png",
    )
    plot_top_ham_keywords(
        features_dict["feature_names"],
        coefficients,
        top_n=15,
        filename="05_top_ham_keywords.png",
    )

    # Model comparison
    plot_model_comparison(
        baseline_results["metrics"],
        logistic_results["metrics"],
        filename="06_model_comparison.png",
    )

    # Cross-validation scores
    plot_cross_validation_scores(
        cv_logistic_results,
        model_name="Logistic Regression",
        filename="07_cv_scores_logistic.png",
    )

    # ROC curves
    plot_roc_curves(
        baseline_model_final,
        logistic_model_final,
        X_test,
        y_test,
        filename="08_roc_curves_comparison.png",
    )

    # Learning curves
    plot_learning_curves(
        logistic_model_final,
        X_train_smote,
        y_train_smote,
        model_name="Logistic Regression (Final)",
        filename="09_learning_curves_logistic.png",
    )

    # Create metrics table
    metrics_table = create_metrics_table(
        baseline_results["metrics"],
        logistic_results["metrics"],
        cv_baseline_results,
        cv_logistic_results,
    )

    logger.info("\n" + "=" * 80)
    logger.info("METRICS COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info("\n" + metrics_table.to_string(index=False))

    # Save metrics table
    table_csv_path = MODELS_DIR / "metrics_comparison_table.csv"
    metrics_table.to_csv(table_csv_path, index=False)
    logger.info(f"\nMetrics table saved to {table_csv_path}")

    # Phase 13: Summary
    logger.info("\n" + "=" * 80)
    logger.info("ADVANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {MODELS_DIR}")
    logger.info(f"Results saved to: {MODELS_DIR.parent / 'results'}")

    logger.info(f"\nBaseline Model Performance (SMOTE + Final):")
    logger.info(f"  Accuracy:  {baseline_results['metrics']['accuracy']:.4f}")
    logger.info(f"  Precision: {baseline_results['metrics']['precision']:.4f}")
    logger.info(f"  Recall:    {baseline_results['metrics']['recall']:.4f}")
    logger.info(f"  F1 Score:  {baseline_results['metrics']['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {baseline_results['metrics']['roc_auc']:.4f}")

    logger.info(f"\nBest Model: Logistic Regression (SMOTE + Final)")
    logger.info(f"  Accuracy:  {logistic_results['metrics']['accuracy']:.4f}")
    logger.info(f"  Precision: {logistic_results['metrics']['precision']:.4f}")
    logger.info(f"  Recall:    {logistic_results['metrics']['recall']:.4f}")
    logger.info(f"  F1 Score:  {logistic_results['metrics']['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {logistic_results['metrics']['roc_auc']:.4f}")

    logger.info(f"\nCross-Validation Performance (Original Data):")
    logger.info(f"  Logistic Regression:")
    logger.info(f"    Mean F1: {cv_logistic_results['test_f1'].mean():.4f} ± {cv_logistic_results['test_f1'].std():.4f}")
    logger.info(f"    Mean ROC-AUC: {cv_logistic_results['test_roc_auc'].mean():.4f} ± {cv_logistic_results['test_roc_auc'].std():.4f}")

    logger.info(f"\n{CV_FOLDS} Visualizations generated and saved to results/")
    logger.info(f"Metrics table saved to models/metrics_comparison_table.csv")

    logger.info("\nTo run the Streamlit app:")
    logger.info("  streamlit run app.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
