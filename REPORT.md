# Phishing Email Detection and Explainable Risk Scoring

## Executive Summary

This project implements an AI for Cybersecurity pipeline for phishing email
detection. The system standardizes raw public email datasets, performs EDA,
extracts TF-IDF text features and security metadata, trains multiple classifiers,
evaluates them with security-relevant metrics, and serves a Streamlit demo that
returns a phishing probability, risk score, risk level, and SOC-style reasons.

The latest run used `CEAS_08.csv`, `Enron.csv`, and `Nazario.csv`. After
standardization and cleaning, the dataset contained 69,941 usable emails:
32,763 legitimate and 37,178 phishing. The selected final model is
`logistic_regression_text_metadata` because it supports `predict_proba()` and
coefficient-based explanations for both text and metadata features.

Random-split results are strong, but near-duplicate and source-holdout checks
show that cross-source generalization is weaker. Therefore, the project should
be presented as a course prototype, not a production email gateway.

## 1. Security Problem

Phishing email detection is a binary classification problem:

| Label | Meaning |
| ---: | --- |
| 0 | Legitimate / benign email |
| 1 | Phishing email |

False negatives are especially dangerous because missed phishing messages can
reach users and lead to credential theft, fraud, malware delivery, or business
email compromise. False positives also matter because noisy alerts create user
and analyst fatigue.

For this reason, Accuracy is not the main success criterion. The report focuses
on Precision, Recall, F1, PR-AUC, ROC-AUC, and Confusion Matrix. Recall is
important for catching phishing, Precision is important for reducing false
alerts, and PR-AUC is useful when evaluating phishing detection quality across
thresholds.

## 2. Eight-Stage ML for Security Pipeline

1. Data collection: raw CEAS, Enron, and Nazario CSV files are placed in `data/raw/`.
2. Data standardization: schemas are normalized into one canonical CSV.
3. EDA: class balance, missing values, source distribution, lengths, and terms.
4. Preprocessing: text cleaning removes HTML and normalizes URLs, emails, and numbers.
5. Feature engineering: TF-IDF unigrams/bigrams plus metadata security signals.
6. Modeling: baseline, explainable, margin-based, and tree-based classifiers.
7. Evaluation: classification metrics, confusion matrices, PR/ROC curves, leakage checks.
8. Risk scoring + Explainability: probability-based scoring and SOC explanations.

## 3. Dataset and Standardization

Raw files:

| Source | Rows after cleaning | Share |
| --- | ---: | ---: |
| CEAS_08.csv | 38,963 | 55.71% |
| Enron.csv | 29,423 | 42.07% |
| Nazario.csv | 1,555 | 2.22% |

Class distribution:

| Label | Meaning | Count |
| ---: | --- | ---: |
| 0 | Legitimate | 32,763 |
| 1 | Phishing | 37,178 |

The standardized dataset preserves:

```text
source_file, sender, receiver, date, subject, body, urls, text_combined, label
```

Preserving `source_file` enables source-holdout experiments, and preserving
sender/body/subject/URL fields enables metadata feature extraction.

## 4. Preprocessing and Feature Engineering

Text preprocessing is implemented in `src/text_cleaning.py`:

- lowercase conversion
- HTML tag removal
- URL replacement with `url`
- email replacement with `email`
- number replacement with `num`
- special character removal
- whitespace normalization

Text features use:

```text
TfidfVectorizer(
    preprocessor=clean_text,
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

Metadata features are implemented in `src/feature_engineering.py`:

| Feature | Meaning |
| --- | --- |
| `n_links` | Number of URL-like strings in the email text or URL column |
| `has_ip_url` | Whether a URL contains a raw IPv4 address |
| `html_tag_count` | Number of HTML tags |
| `html_ratio` | HTML tag character ratio over raw text length |
| `body_len` | Body length in characters |
| `subject_len` | Subject length in characters |
| `n_exclamation` | Number of exclamation marks |
| `n_upper_words` | Number of all-uppercase words |
| `sender_has_email` | Whether sender field contains an email address |
| `url_count_from_column` | URL count from the standardized `urls` column |

## 5. Leakage Control

The training script uses sklearn `Pipeline` and `ColumnTransformer`. The
train/test split happens before fitting the vectorizer, scaler, SVD transformer,
or classifier. This means:

- TF-IDF vocabulary is fit only on training text.
- StandardScaler is fit only on training metadata.
- TruncatedSVD for Random Forest is fit only on training TF-IDF features.
- The test set is transformed only after fitted train-only preprocessing exists.

No SMOTE or oversampling is applied to the test set. Class imbalance is handled
with `class_weight='balanced'` for Logistic Regression and Linear SVM and
`class_weight='balanced_subsample'` for Random Forest.

## 6. Models

| Model | Feature set | Role |
| --- | --- | --- |
| `naive_bayes` | TF-IDF text | Baseline |
| `logistic_regression` | TF-IDF text | Legacy explainable baseline |
| `linear_svm` | TF-IDF text | Strong text comparison |
| `logistic_regression_text_metadata` | TF-IDF + metadata | Final risk scoring model |
| `random_forest_text_metadata` | SVD-compressed TF-IDF + metadata | Improved tree model |

The final app loads:

```text
models/phishing_logreg_text_metadata.pkl
```

The old text-only Logistic Regression model is still saved as:

```text
models/phishing_logreg_tfidf.pkl
```

## 7. Random Split Evaluation

Stratified random split settings:

```text
test_size = 0.2
random_state = 42
```

Metrics from `results/metrics_text_metadata.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9578 | 0.9901 | 0.9299 | 0.9591 | 0.9961 | 0.9960 | 6484 | 69 | 521 | 6915 |
| Logistic Regression | 0.9862 | 0.9857 | 0.9884 | 0.9870 | 0.9990 | 0.9991 | 6446 | 107 | 86 | 7350 |
| Linear SVM | 0.9912 | 0.9915 | 0.9919 | 0.9917 | 0.9994 | 0.9994 | 6490 | 63 | 60 | 7376 |
| Logistic Regression + metadata | 0.9874 | 0.9862 | 0.9902 | 0.9882 | 0.9990 | 0.9991 | 6450 | 103 | 73 | 7363 |
| Random Forest + metadata | 0.9846 | 0.9842 | 0.9870 | 0.9856 | 0.9988 | 0.9989 | 6435 | 118 | 97 | 7339 |

Interpretation:

- Linear SVM has the best random-split F1, but it is not selected as the final
  model because direct probability-based risk scoring is less natural.
- Logistic Regression + metadata improves Recall from 0.9884 to 0.9902 and F1
  from 0.9870 to 0.9882 compared with text-only Logistic Regression.
- Random Forest + metadata is a valid improved model with probability output,
  but it does not outperform Logistic Regression + metadata on this run.

## 8. Duplicate and Robustness Checks

Cleaned-text duplicate check from `results/cleaned_duplicate_check.csv`:

| Metric | Value |
| --- | ---: |
| Rows before cleaned dedup | 69,940 |
| Duplicate rows by original `text_combined` | 0 |
| Duplicate rows by cleaned text | 7,355 |
| Conflicting cleaned texts | 0 |
| Rows after drop duplicate by cleaned text | 62,585 |

Cleaned-dedup benchmark from `results/metrics_cleaned_dedup.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9665 | 0.9757 | 0.9564 | 0.9660 | 0.9955 | 0.9945 |
| Logistic Regression | 0.9853 | 0.9816 | 0.9889 | 0.9853 | 0.9985 | 0.9983 |
| Linear SVM | 0.9898 | 0.9867 | 0.9928 | 0.9897 | 0.9992 | 0.9991 |

Near-duplicate similarity from `results/near_duplicate_check.csv`:

| Metric | Value |
| --- | ---: |
| Train rows | 55,952 |
| Test rows | 13,989 |
| Mean max similarity | 0.6836 |
| Median max similarity | 0.7411 |
| Test samples with similarity > 0.90 | 4,268 |
| Ratio with similarity > 0.90 | 0.3051 |
| Test samples with similarity > 0.95 | 3,247 |
| Ratio with similarity > 0.95 | 0.2321 |

These results show that random-split performance is optimistic because many
test emails are highly similar to training emails.

## 9. Source-Holdout Evaluation

Source holdout trains on one source group and tests on another. It is stricter
than random split because source-specific templates and vocabulary do not appear
in both train and test.

Metrics from `results/source_holdout_metrics.csv` using
`logistic_regression_text_metadata`:

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEAS + Enron -> Nazario | 0.4630 | n/a | 0.4630 | n/a | n/a | n/a | Nazario has only phishing labels, so phishing recall is reported. |
| CEAS -> Enron | 0.4771 | 0.4754 | 0.9966 | 0.6437 | 0.7874 | 0.8145 | Very high recall but many false positives on Enron. |
| Enron -> CEAS | 0.8050 | 0.7439 | 0.9906 | 0.8497 | 0.8960 | 0.9020 | Better, but still lower than random split. |

This confirms source shift: the model catches many phishing emails but can
over-alert badly when trained and tested on different sources.

## 10. Explainable Risk Scoring

Risk scoring:

```text
risk_score = P(phishing) * 100
```

Risk levels:

| Score range | Risk level |
| --- | --- |
| `< 40` | Low |
| `40-69` | Medium |
| `70-89` | High |
| `>= 90` | Critical |

For Logistic Regression explanations:

```text
contribution = transformed_feature_value * coefficient
```

Text outputs:

```text
results/top_phishing_indicators_text_metadata.csv
results/top_legitimate_indicators_text_metadata.csv
```

Metadata outputs:

```text
results/metadata_coefficients.csv
```

Latest metadata coefficients:

| Feature | Coefficient |
| --- | ---: |
| `url_count_from_column` | 1.0873 |
| `body_len` | 0.4395 |
| `n_exclamation` | 0.1244 |
| `subject_len` | 0.0897 |
| `html_ratio` | 0.0278 |
| `has_ip_url` | -0.0681 |
| `html_tag_count` | -0.2153 |
| `n_upper_words` | -0.4860 |
| `n_links` | -1.1809 |
| `sender_has_email` | -1.2826 |

The sample SOC explanation is saved to:

```text
results/sample_soc_explanation.json
results/sample_soc_explanation.csv
```

It includes original email preview, true label, predicted label, phishing
probability, risk score, risk level, top text contributions, top metadata
contributions, and a short analyst-facing explanation.

## 11. Streamlit App

`app.py` allows users to paste an email and view:

- prediction
- phishing probability
- risk score
- risk level
- top text indicators
- metadata indicators
- latest model comparison metrics

If the final model is missing, the app tells the user to run:

```bash
python standardize_datasets.py
python eda_standardized.py
python train_optimized.py
streamlit run app.py
```

## 12. Generated Artifacts

Key model artifacts:

```text
models/phishing_logreg_text_metadata.pkl
models/phishing_random_forest_text_metadata.pkl
models/phishing_logreg_tfidf.pkl
```

Key result artifacts:

```text
results/metrics_text_metadata.csv
results/logistic_regression_text_metadata_confusion_matrix.png
results/random_forest_text_metadata_confusion_matrix.png
results/roc_curve_text_metadata_comparison.png
results/pr_curve_text_metadata_comparison.png
results/top_phishing_indicators_text_metadata.csv
results/top_legitimate_indicators_text_metadata.csv
results/metadata_coefficients.csv
results/sample_soc_explanation.json
results/sample_soc_explanation.csv
```

## 13. Limitations and Future Work

Limitations:

- Random split metrics are optimistic because of template overlap.
- Source-holdout performance is much lower and should be considered stricter.
- Metadata availability differs by dataset source.
- The system does not inspect attachments, full headers, SPF/DKIM/DMARC, URL
  reputation, or sender reputation.
- Probability scores are not separately calibrated.
- The current model can over-alert under source shift.

Future work:

- Add probability calibration and threshold tuning.
- Group near-duplicates before train/test split.
- Add URL reputation, domain age, and email authentication features.
- Evaluate on more external phishing datasets.
- Add cost-sensitive threshold recommendations for SOC operations.
- Package the final report as PDF/DOCX and prepare presentation slides.

## 14. AI Tool Usage Declaration

AI assistance was used to inspect and modify the repository, add metadata
features, update training/evaluation/explainability code, refresh documentation,
and run validation commands. Metrics in this report are generated from local
script outputs and should be reviewed by the project owner before submission.
