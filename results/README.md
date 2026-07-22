Results Directory

Exploratory analysis, evaluation outputs, robustness checks, and explanation artifacts are generated locally in this directory.

Run:

python eda_standardized.py
python train_optimized.py

Generated files are excluded from Git.

Exploratory Data Analysis

Typical EDA outputs:

class_distribution.png
text_length_distribution.png
missing_values.csv
dataset_source_distribution.csv
duplicate_statistics.csv
top_terms_by_class.csv
Model Metrics

Typical metric outputs:

metrics_optimized.csv
metrics_text_metadata.csv
metrics_cleaned_dedup.csv
source_holdout_metrics.csv

metrics_optimized.csv contains text-only model results.

metrics_text_metadata.csv contains the complete model comparison.

metrics_cleaned_dedup.csv contains the cleaned-text deduplicated benchmark.

source_holdout_metrics.csv contains cross-source evaluation results.

Robustness Checks
cleaned_duplicate_check.csv
near_duplicate_check.csv

These files help identify:

Duplicate content after text cleaning.
Similarity between training and testing emails.
Potentially optimistic random-split evaluation.
Confusion Matrices
naive_bayes_confusion_matrix.png
logistic_regression_confusion_matrix.png
linear_svm_confusion_matrix.png
logistic_regression_text_metadata_confusion_matrix.png
random_forest_text_metadata_confusion_matrix.png

The training script also creates confusion matrices for the cleaned-deduplicated benchmark.

ROC and Precision–Recall Curves
roc_curve_comparison.png
pr_curve_comparison.png
roc_curve_text_metadata_comparison.png
pr_curve_text_metadata_comparison.png

The current training script creates the two ROC files from the same model list and does the same for the two Precision–Recall files.

Error Analysis and Predictions
error_samples.csv
sample_predictions.csv

error_samples.csv contains selected false-positive and false-negative examples.

sample_predictions.csv contains sample predictions, probabilities, risk scores, and risk levels.

Explainability Outputs
top_phishing_indicators.csv
top_legitimate_indicators.csv
top_phishing_indicators_text_metadata.csv
top_legitimate_indicators_text_metadata.csv
metadata_coefficients.csv
sample_soc_explanation.json
sample_soc_explanation.csv

These files contain:

Global text indicators.
Text-and-metadata indicators.
Metadata coefficients.
A sample analyst-facing explanation.
Important Notes
Result files may change whenever the data, preprocessing, model settings, or dependency versions change.
Do not copy old metric values into documentation without regenerating them.
Random-split results should be interpreted together with duplicate and source-holdout checks.
Application behavior should be evaluated using the same decision threshold used by src/risk_scoring.py.