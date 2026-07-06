# Phishing Email Detection and Explainable Risk Scoring

## Project Overview

This project is an end-to-end classical machine learning pipeline for phishing
email detection. It standardizes raw email CSV files, performs exploratory data
analysis, trains TF-IDF based classifiers, evaluates model quality, checks for
duplicate/template leakage risk, and serves a small Streamlit demo with
explainable risk scoring.

## Cybersecurity Problem

Phishing email detection is a binary text classification task:

- `0` = legitimate / benign email
- `1` = phishing email

False negatives are especially dangerous because a missed phishing email can
reach a user and lead to credential theft, fraud, malware delivery, or business
email compromise. False positives also matter because excessive alerts reduce
trust in the system.

## Dataset Placement

Raw datasets are not committed to Git because they can be large. Place CSV files
in:

```text
data/raw/
```

The expected course datasets are:

```text
data/raw/CEAS_08.csv
data/raw/Enron.csv
data/raw/Nazario.csv
```

Supported schemas include:

- `subject,body,label`
- `subject,body,label,urls`
- `sender,receiver,date,subject,body,label,urls`
- `sender,receiver,date,subject,body,urls,label`
- `text_combined,label`

`standardize_datasets.py` writes:

```text
data/processed/phishing_email_standardized.csv
```

The standardized file includes `source_file`, `text_combined`, and `label`.
Metadata such as sender, receiver, date, subject, body, and urls is preserved
when present.

## Pipeline

1. Standardize raw CSV files and normalize labels.
2. Remove invalid labels, empty emails, exact duplicates, and conflicting labels.
3. Run EDA and save plots/CSV summaries.
4. Train Naive Bayes, Logistic Regression, and Linear SVM with TF-IDF features.
5. Evaluate random split benchmark metrics.
6. Run cleaned-text duplicate and near-duplicate checks.
7. Run source-holdout generalization checks.
8. Save the final Logistic Regression model.
9. Serve predictions, risk scores, and explanations in Streamlit.

## Models

- Naive Bayes: baseline model.
- Logistic Regression + TF-IDF: final explainable risk scoring model.
- Linear SVM: comparison model.

Linear SVM can score higher on F1, but Logistic Regression is selected as the
final model because it supports `predict_proba()` and has interpretable
coefficients for the explanation layer.

## Metrics

The training workflow reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- PR-AUC
- ROC-AUC when scores/probabilities are available

Random train/test split is treated as an in-distribution benchmark. Cleaned
deduplication, near-duplicate similarity, and source holdout are treated as
robustness and generalization checks.

## Explainable Risk Scoring

The final model is generated locally at:

```text
models/phishing_logreg_tfidf.pkl
```

Risk score:

```text
risk_score = P(phishing) * 100
```

Risk levels:

- Low risk: score < 40
- Medium risk: 40 <= score < 70
- High risk: score >= 70

Single-email explanations use:

```text
contribution = TF-IDF value * Logistic Regression coefficient
```

The app displays active terms that push the prediction toward phishing.

## Generated Files

Generated data, models, and results are ignored by Git:

```text
data/processed/phishing_email_standardized.csv
models/phishing_logreg_tfidf.pkl
results/metrics_optimized.csv
results/metrics_cleaned_dedup.csv
results/near_duplicate_check.csv
results/source_holdout_metrics.csv
results/cleaned_duplicate_check.csv
results/missing_values.csv
results/dataset_source_distribution.csv
results/duplicate_statistics.csv
results/top_terms_by_class.csv
results/error_samples.csv
results/sample_predictions.csv
results/top_phishing_indicators.csv
results/top_legitimate_indicators.csv
results/*_confusion_matrix.png
results/roc_curve_comparison.png
results/pr_curve_comparison.png
```

## How to Run

```bash
python standardize_datasets.py
python eda_standardized.py
python train_optimized.py
streamlit run app.py
```

See `HOW_TO_RUN.md` for setup from a clean environment.

## Latest Local Run Summary

Using the local raw CSV files currently present in `data/raw/`, standardization
produced 69,941 usable rows:

- Legitimate: 32,763
- Phishing: 37,178
- Sources: CEAS_08.csv, Enron.csv, Nazario.csv

Random split Logistic Regression result:

- Accuracy: 0.9862
- Precision: 0.9857
- Recall: 0.9884
- F1-score: 0.9870
- PR-AUC: 0.9991
- ROC-AUC: 0.9990

The near-duplicate check found that 30.51% of random-split test emails had a
maximum train similarity above 0.90, so random split metrics should not be
treated as the only evidence of generalization.

## Limitations

- Dataset quality and source overlap strongly affect performance.
- Random split metrics can be optimistic when templates or near-duplicates are
  shared between train and test sets.
- Source-holdout results are stricter and may be lower.
- The model is text-pattern based and can miss highly customized phishing.
- Probability scores are not explicitly calibrated.
- This is a course prototype, not a production email security gateway.
