# Phishing Email Detection and Explainable Risk Scoring
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-Educational-green.svg)

## Project Overview

This project is an end-to-end AI for Cybersecurity pipeline for phishing email
detection. It standardizes raw email datasets, performs EDA, trains classical ML
models, evaluates phishing detection quality beyond accuracy, and serves a
Streamlit demo with explainable risk scoring for SOC-style review.

The current final model is Logistic Regression with TF-IDF text features plus
engineered metadata features. It is selected because it supports
`predict_proba()` for risk scoring and has coefficients for explainability.

## Security Task

Binary labels:

- `0` = legitimate / benign email
- `1` = phishing email

False negatives are high risk because missed phishing emails can reach users.
False positives also matter because too many alerts create analyst and user
fatigue. For that reason, the project focuses on Precision, Recall, F1,
PR-AUC, ROC-AUC, and Confusion Matrix. Accuracy is reported only as supporting
context.
## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Workflow Pipeline](#workflow-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Performance](#model-performance)

## Dataset Placement

Place the course CSV files in:

```text
data/raw/
```

Expected files:

```text
data/raw/CEAS_08.csv
data/raw/Enron.csv
data/raw/Nazario.csv
```

`standardize_datasets.py` writes:

```text
data/processed/phishing_email_standardized.csv
```

The standardized CSV preserves these canonical columns when available:

```text
source_file, sender, receiver, date, subject, body, urls, text_combined, label
```

## Eight-Stage ML for Security Pipeline

1. Data collection: raw CEAS, Enron, and Nazario CSV files.
2. Data standardization: schema normalization and label mapping.
3. EDA: class balance, missing values, source distribution, text lengths, terms.
4. Preprocessing: HTML stripping, URL/email/number normalization, whitespace cleanup.
5. Feature engineering: TF-IDF unigrams/bigrams plus metadata features.
6. Modeling: Naive Bayes, Logistic Regression, Linear SVM, and Random Forest.
7. Evaluation: Precision, Recall, F1, PR-AUC, ROC-AUC, and Confusion Matrix.
8. Risk scoring + Explainability: probability-based score and coefficient reasons.

## Features

Text branch:

- `TfidfVectorizer(preprocessor=clean_text)`
- Unigram + bigram features
- `max_features=20000`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`

Metadata branch:

- `n_links`
- `has_ip_url`
- `html_tag_count`
- `html_ratio`
- `body_len`
- `subject_len`
- `n_exclamation`
- `n_upper_words`
- `sender_has_email`
- `url_count_from_column`

The text+metadata models use sklearn `Pipeline` and `ColumnTransformer`, so the
TF-IDF vectorizer, SVD transformer, and scaler are fit only on the train split.
This avoids train/test leakage.

## Models

- `naive_bayes`: text-only baseline.
- `logistic_regression`: text-only explainable baseline.
- `linear_svm`: strong text-only comparison model.
- `logistic_regression_text_metadata`: final explainable risk scoring model.
- `random_forest_text_metadata`: improved tree model using SVD-compressed text
  features plus metadata.

Generated model files:

```text
models/phishing_logreg_text_metadata.pkl
models/phishing_random_forest_text_metadata.pkl
models/phishing_logreg_tfidf.pkl
```

## Risk Scoring

```text
risk_score = P(phishing) * 100
```

Risk levels:

- Low: score < 40
- Medium: 40 <= score < 70
- High: 70 <= score < 90
- Critical: score >= 90

Single-email explanations use:

```text
contribution = transformed_feature_value * Logistic Regression coefficient
```

Text indicators and metadata contributions are shown separately.

## Latest Local Run Summary

Using the local raw CSV files currently present in `data/raw/`, standardization
produced 69,941 usable rows:

- Legitimate: 32,763
- Phishing: 37,178
- Sources: CEAS_08.csv, Enron.csv, Nazario.csv

Random split results from `results/metrics_text_metadata.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9578 | 0.9901 | 0.9299 | 0.9591 | 0.9961 | 0.9960 |
| Logistic Regression | 0.9862 | 0.9857 | 0.9884 | 0.9870 | 0.9990 | 0.9991 |
| Linear SVM | 0.9912 | 0.9915 | 0.9919 | 0.9917 | 0.9994 | 0.9994 |
| Logistic Regression + metadata | 0.9874 | 0.9862 | 0.9902 | 0.9882 | 0.9990 | 0.9991 |
| Random Forest + metadata | 0.9846 | 0.9842 | 0.9870 | 0.9856 | 0.9988 | 0.9989 |

The final explainable model is `logistic_regression_text_metadata`. It improves
random-split Recall and F1 over the text-only Logistic Regression while keeping
probability output and coefficient-based explanations.

The near-duplicate check still found many high-similarity random-split test
emails, so random split metrics should be treated as optimistic. Source-holdout
results are stricter and show that cross-source generalization remains a real
limitation.

## Key Generated Results

```text
results/metrics_text_metadata.csv
results/metrics_optimized.csv
results/metrics_cleaned_dedup.csv
results/source_holdout_metrics.csv
results/near_duplicate_check.csv
results/*_confusion_matrix.png
results/roc_curve_text_metadata_comparison.png
results/pr_curve_text_metadata_comparison.png
results/top_phishing_indicators_text_metadata.csv
results/top_legitimate_indicators_text_metadata.csv
results/metadata_coefficients.csv
results/sample_soc_explanation.json
results/sample_soc_explanation.csv
```

## How to Run

```bash
python standardize_datasets.py
python eda_standardized.py
python train_optimized.py
streamlit run app.py
```

See `HOW_TO_RUN.md` for setup from a clean environment.

## Limitations

- Random split metrics can be optimistic because of templates and near-duplicates.
- Source-holdout performance is lower and should be treated as a stricter check.
- Metadata availability differs by source, especially for Enron.
- The model does not inspect attachments, headers, sender reputation, or live URL reputation.
- Probability scores are not separately calibrated.
- This is a course prototype, not a production email security gateway.
