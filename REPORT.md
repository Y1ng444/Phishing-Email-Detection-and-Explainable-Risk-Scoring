# Phishing Email Detection and Explainable Risk Scoring

## 1. Introduction

This project implements a course-aligned machine learning pipeline for phishing
email detection. The system uses classical text classification methods to
predict whether an email is legitimate or phishing and provides a simple risk
score with coefficient-based explanations.

## 2. Problem Statement

The cybersecurity task is binary text classification:

- `0`: legitimate email
- `1`: phishing email

The goal is to identify phishing emails while keeping the workflow clear,
reproducible, and understandable.

## 3. Dataset

Raw CSV files are placed in `data/raw/`. The standardization script supports
schemas containing subject, body, label, URLs, metadata fields, or an existing
`text_combined` column.

Latest standardized dataset:

- Total raw rows before cleaning: 164,972
- Final usable rows after cleaning: 163,949
- Legitimate emails: 78,481
- Phishing emails: 85,468
- Duplicate rows removed: 1,022
- Conflicting duplicated texts removed: 0

The final processed file is:

```text
data/processed/phishing_email_standardized.csv
```

## 4. Preprocessing

The text cleaning step is intentionally simple and regex-based:

- Lowercase text
- Remove HTML tags
- Replace URLs with `url`
- Replace email addresses with `email`
- Replace numbers with `num`
- Remove special characters
- Normalize whitespace

Rows with invalid labels or empty `text_combined` values are removed. Exact
duplicates and duplicate text entries are removed before the train/test split.

## 5. Feature Extraction

The models use TF-IDF features with:

- Unigrams and bigrams
- `max_features=20000`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`

TF-IDF is inside each sklearn `Pipeline`, so the vectorizer is fit only on the
training split.

## 6. Models

Three classical machine learning models are trained:

- Multinomial Naive Bayes as the baseline model
- Logistic Regression as the improved explainable model
- Linear SVM as a comparison model

Logistic Regression is selected for risk scoring because it provides phishing
probabilities and interpretable coefficients.

## 7. Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion matrix

The split uses `test_size=0.2`, `stratify=y`, and `random_state=42`.

## 8. Results

Latest results from `results/metrics_optimized.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9604 | 0.9874 | 0.9361 | 0.9611 | 0.9962 | 0.9961 |
| Logistic Regression | 0.9882 | 0.9869 | 0.9904 | 0.9887 | 0.9990 | 0.9990 |
| Linear SVM | 0.9950 | 0.9947 | 0.9958 | 0.9952 | 0.9997 | 0.9997 |

Linear SVM had the highest F1-score in this run. Logistic Regression remains
the final risk scoring model because it supports probability-based scoring and
coefficient explanations.

## 9. Explainable Risk Scoring

Risk score is defined as:

```text
risk_score = P(phishing) * 100
```

Risk levels:

- Low risk: score < 40
- Medium risk: 40 <= score < 70
- High risk: score >= 70

The explanation layer uses Logistic Regression coefficients:

- Largest positive coefficients are global phishing indicators.
- Most negative coefficients are global legitimate indicators.
- For a single email, active TF-IDF terms are multiplied by their coefficients
  to show which terms contributed most toward phishing.

Generated explanation files:

```text
results/top_phishing_indicators.csv
results/top_legitimate_indicators.csv
```

## 10. Error Analysis

For Logistic Regression, the workflow saves up to 10 false positives and up to
10 false negatives to:

```text
results/error_samples.csv
```

False Positive: legitimate email predicted as phishing.

False Negative: phishing email predicted as legitimate.

In cybersecurity, false negatives are more dangerous because phishing emails are
missed. False positives are also important because they may annoy users or block
legitimate communication.

## 11. Security Discussion

Recall is important because missed phishing emails can expose users to credential
theft, malware, or fraud. Precision is also important because excessive false
alerts reduce trust in the system. PR-AUC is useful for this problem because it
summarizes precision and recall over decision thresholds.

The explanation layer helps users inspect which words or n-grams influenced a
prediction, making the model easier to discuss in a security context.

## 12. Limitations

- Results depend on the quality and labeling of the datasets.
- Exact duplicate removal does not catch near-duplicate emails.
- The model may not generalize to newer phishing styles without updated data.
- Coefficient explanations show model behavior, not guaranteed real-world cause.
- The Streamlit app is a simple course demo.

## 13. Conclusion

The final project satisfies the required classical machine learning workflow:
email cleaning, tokenization through TF-IDF preprocessing, unigram and bigram
feature extraction, a baseline model, improved model, comparison model, error
analysis, and a simple explanation layer. Logistic Regression is used as the
final explainable risk scoring model, while all three models are evaluated with
precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrices.
