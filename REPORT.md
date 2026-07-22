# Phishing Email Detection and Explainable Risk Scoring

## Executive Summary

This project implements an end-to-end machine-learning pipeline for phishing and suspicious email detection.

The system:

* Standardizes email datasets with different schemas.
* Performs exploratory data analysis.
* Cleans and normalizes email text.
* Extracts TF-IDF and metadata features.
* Trains and compares several classical machine-learning models.
* Evaluates duplicate leakage and cross-source generalization.
* Produces probability-based risk scores.
* Generates coefficient-based explanations.
* Provides a Streamlit demonstration application.

The selected deployment model is Logistic Regression with TF-IDF and engineered metadata features.

It was selected because it provides phishing probabilities and interpretable coefficients while maintaining strong classification capability.

This system is a course prototype and should not be treated as a production secure email gateway.

---

## 1. Security Problem

The project formulates email detection as a binary classification task:

| Label | Meaning                      |
| ----: | ---------------------------- |
|   `0` | Legitimate or benign email   |
|   `1` | Phishing or suspicious email |

False negatives are security-sensitive because a missed malicious email may reach a user.

False positives also matter because excessive warnings can produce user and analyst fatigue.

The project therefore evaluates models using:

* Precision
* Recall
* F1 score
* ROC-AUC
* PR-AUC / Average Precision
* Confusion Matrix
* Accuracy as supporting information

The code calculates PR-AUC with `average_precision_score()`.

---

## 2. Scope

The project analyzes text and basic email metadata supplied through CSV files or manually entered into the Streamlit application.

The project does not:

* Connect directly to an email provider.
* Read a mailbox automatically.
* Download attachments.
* Analyze attachment contents.
* Perform malware sandboxing.
* Fully parse authentication headers.
* Verify SPF, DKIM, or DMARC.
* Query live threat-intelligence services.
* Quarantine or delete messages.

---

## 3. Dataset Standardization

Dataset standardization is implemented in:

```text
standardize_datasets.py
```

The expected course datasets are placed in:

```text
data/raw/
```

The script supports different source-column names and maps them to a canonical schema.

Canonical fields include:

```text
source_file
sender
receiver
date
subject
body
urls
text_combined
label
```

If `text_combined` is not available, it is generated from the subject, body, and URL fields.

The standardization process removes:

* Rows with unrecognized labels.
* Rows with empty combined text.
* Exact duplicate rows.
* Identical text associated with conflicting labels.
* Duplicate combined-text records after conflict removal.

The standardized dataset is saved to:

```text
data/processed/phishing_email_standardized.csv
```

### Label mapping limitation

The standardization logic maps values such as `phishing`, `spam`, `malicious`, and similar positive labels into class `1`.

As a result, the positive class may represent a mixture of phishing, spam, and other suspicious email content.

This limitation must be considered when interpreting model outputs.

---

## 4. Exploratory Data Analysis

Exploratory analysis is implemented in:

```text
eda_standardized.py
```

The script examines:

* Class distribution.
* Dataset-source distribution.
* Missing values.
* Email-text length.
* Exact duplicate content.
* Duplicate content after text cleaning.
* Frequent unigrams and bigrams.

Frequent-term analysis is descriptive. A frequently occurring word is not necessarily an important model feature.

EDA artifacts are written to:

```text
results/
```

---

## 5. Text Preprocessing

Text preprocessing is implemented in:

```text
src/text_cleaning.py
```

The `clean_text()` function performs:

1. HTML entity decoding.
2. Lowercase conversion.
3. HTML tag removal.
4. URL replacement with `url`.
5. Email-address replacement with `email`.
6. Number replacement with `num`.
7. Unsupported-character removal.
8. Whitespace normalization.

Example:

```text
URGENT! Verify user@example.com at https://example.com in 24 hours.
```

Becomes approximately:

```text
urgent verify email at url in num hours
```

This process reduces memorization of specific URLs, addresses, and numbers.

However, the cleaning rule removes all characters outside lowercase English letters and whitespace. This limits support for Vietnamese, accented text, and non-Latin languages.

---

## 6. TF-IDF Features

The project uses:

```python
TfidfVectorizer(
    preprocessor=clean_text,
    analyzer="word",
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
    smooth_idf=True,
)
```

### Unigrams

Unigrams represent individual words, such as:

```text
verify
account
password
urgent
```

### Bigrams

Bigrams represent two consecutive words, such as:

```text
verify account
reset password
urgent action
click url
```

Bigrams help preserve limited local context.

### Sublinear term frequency

With `sublinear_tf=True`, repeated words receive diminishing additional influence.

This reduces the ability of a word repeated many times to dominate the feature vector.

---

## 7. Metadata Features

Metadata extraction is implemented in:

```text
src/feature_engineering.py
```

The final model uses:

| Feature                 | Meaning                                            |
| ----------------------- | -------------------------------------------------- |
| `n_links`               | Number of URL-like strings                         |
| `has_ip_url`            | Whether a URL contains an IPv4 address             |
| `html_tag_count`        | Number of HTML tags                                |
| `html_ratio`            | HTML-tag character ratio                           |
| `body_len`              | Email body length                                  |
| `subject_len`           | Subject length                                     |
| `n_exclamation`         | Number of exclamation marks                        |
| `n_upper_words`         | Number of uppercase words                          |
| `sender_has_email`      | Whether the sender field contains an email address |
| `url_count_from_column` | URL count from the standardized URL field          |

Metadata features are transformed with `StandardScaler`.

The model therefore uses standardized values rather than raw metadata values directly.

---

## 8. Models

The project trains five model configurations.

### Multinomial Naive Bayes

```text
TF-IDF → Multinomial Naive Bayes
```

This provides a fast text-classification baseline.

### Text-only Logistic Regression

```text
TF-IDF → Logistic Regression
```

This model supports probability output and interpretable coefficients.

### Linear Support Vector Machine

```text
TF-IDF → LinearSVC
```

Linear SVM is suitable for high-dimensional sparse text features.

It provides a decision score but does not provide `predict_proba()` directly.

### Logistic Regression with text and metadata

```text
TF-IDF text features
        +
Standardized metadata
        ↓
Logistic Regression
```

This is the selected deployment model.

### Random Forest with text and metadata

The Random Forest pipeline first reduces the dimensionality of the TF-IDF representation with `TruncatedSVD`.

```text
TF-IDF
   ↓
TruncatedSVD
   ↓
Compressed text features
        +
Standardized metadata
        ↓
Random Forest
```

---

## 9. Leakage Control

The training workflow uses a stratified train/test split.

TF-IDF, StandardScaler, TruncatedSVD, and the classifiers are placed inside sklearn pipelines.

Therefore:

* The TF-IDF vocabulary is fitted only on training text.
* IDF values are learned only from training data.
* Scaling statistics are learned only from training metadata.
* SVD components are learned only from training features.
* Test data is transformed after training preprocessing has been fitted.

This prevents direct preprocessing leakage from the test set.

No oversampling is applied to the test set.

---

## 10. Evaluation Methodology

The training script generates:

* Classification metrics.
* Confusion matrices.
* ROC curves.
* Precision–Recall curves.
* Error samples.
* Sample predictions.
* Global text indicators.
* Metadata coefficients.
* Example analyst explanations.

Results are generated locally and written to:

```text
results/
```

Hardcoded metric values are intentionally not stored in this report because they may become outdated when the data, preprocessing, model configuration, or software environment changes.

The current results should be read directly from the generated CSV files.

---

## 11. Duplicate and Robustness Checks

### Cleaned-text duplicates

Different original emails may become identical after:

* URLs are replaced.
* Email addresses are replaced.
* Numbers are replaced.
* HTML is removed.
* Capitalization is removed.
* Punctuation is removed.

The project therefore checks duplicates after applying `clean_text()`.

It also trains a separate text-only benchmark after cleaned-text deduplication.

### Near-duplicate analysis

The project calculates the maximum TF-IDF cosine similarity between each test email and the training set.

High similarity indicates that the random test set contains email templates that are close to training examples.

This can make random-split results appear more optimistic than real-world performance.

### Source-holdout evaluation

Source-holdout experiments train on one dataset source or source group and test on another.

These experiments are stricter because source-specific vocabulary, templates, formatting, and metadata availability are not shared between the training and testing sets.

Source-holdout results should be considered a stronger indicator of generalization than the standard random split.

---

## 12. Model Selection

The model with the highest score on one evaluation metric is not automatically the best deployment model.

The selected model is:

```text
logistic_regression_text_metadata
```

It was selected because it:

* Produces positive-class probabilities.
* Supports risk scoring.
* Supports coefficient-based explanations.
* Combines text and metadata.
* Is computationally practical.
* Is easier to interpret than the Random Forest pipeline.
* Is more suitable for probability-based application output than `LinearSVC`.

---

## 13. Risk Scoring

Risk scoring is implemented in:

```text
src/risk_scoring.py
```

The model probability is converted into a score:

```text
risk_score = P(phishing) × 100
```

The score is assigned to one of four levels:

* Low
* Medium
* High
* Critical

The application classifies an email as phishing when:

```text
P(phishing) >= 0.70
```

The normal Logistic Regression `predict()` method generally uses a threshold near `0.50`.

Therefore, evaluation metrics generated with `model.predict()` do not exactly represent application behavior at the deployment threshold.

Application-level metrics should be evaluated separately using the same threshold as the Streamlit application.

### Probability limitation

The model probabilities are not separately calibrated.

The risk score should therefore be interpreted as a relative model score rather than a guaranteed real-world probability.

---

## 14. Explainability

Explainability is implemented in:

```text
src/explainability.py
```

For Logistic Regression:

```text
contribution = transformed feature value × coefficient
```

The final decision also includes the intercept:

```text
logit = intercept + sum(feature contributions)
```

Positive contributions push the model toward the positive class.

Negative contributions push the model toward the legitimate class.

Text explanations report:

* Active word or n-gram.
* TF-IDF value.
* Model coefficient.
* Feature contribution.

Metadata explanations report:

* Metadata feature name.
* Raw feature value.
* Model coefficient.
* Feature contribution.

Metadata contributions use standardized feature values internally. The displayed raw value cannot be multiplied directly by the coefficient to reproduce the reported contribution.

---

## 15. Streamlit Application

The demonstration application is implemented in:

```text
app.py
```

The user can paste an email and review:

* Predicted class.
* Phishing probability.
* Risk score.
* Risk level.
* Active text indicators.
* Metadata indicators.
* Feature contributions.

The application loads:

```text
models/phishing_logreg_text_metadata.pkl
```

The application does not automatically retrieve emails from a mailbox.

---

## 16. Generated Artifacts

Model files:

```text
models/phishing_logreg_text_metadata.pkl
models/phishing_random_forest_text_metadata.pkl
models/phishing_logreg_tfidf.pkl
```

Important result groups include:

```text
results/metrics_*.csv
results/*_confusion_matrix.png
results/*roc_curve*.png
results/*pr_curve*.png
results/*duplicate*.csv
results/source_holdout_metrics.csv
results/error_samples.csv
results/sample_predictions.csv
results/top_*_indicators*.csv
results/metadata_coefficients.csv
results/sample_soc_explanation.json
results/sample_soc_explanation.csv
```

These files are generated locally and excluded from Git.

---

## 17. Limitations

* Positive labels from different sources may not have identical meanings.
* The positive class may include spam or other malicious emails.
* Random splitting may place similar templates in train and test.
* Cross-source performance may be lower.
* Metadata availability differs between sources.
* Missing metadata patterns may indirectly reveal dataset source.
* Model probabilities are not calibrated.
* Deployment and evaluation thresholds are not identical.
* The current text cleaner mainly supports English.
* The model does not inspect attachments.
* Full email headers are not analyzed.
* Email authentication is not verified.
* Live URL and domain reputation are not queried.
* The model is not protected against all adversarial text modifications.
* Generated joblib models may depend on compatible scikit-learn versions.

---

## 18. Future Work

Recommended improvements:

* Add a dedicated validation set.
* Tune the deployment threshold on validation data.
* Evaluate all metrics at the application threshold.
* Add probability calibration.
* Group duplicate templates before splitting.
* Use group-based or time-based evaluation.
* Separate phishing, spam, and other malicious labels.
* Add recent external datasets.
* Add multilingual preprocessing.
* Parse full email headers.
* Add SPF, DKIM, and DMARC signals.
* Add sender-domain and Reply-To mismatch features.
* Add URL and domain reputation.
* Add domain age and DNS features.
* Add attachment analysis.
* Pin dependency versions.
* Add automated tests and continuous integration.

---

## 19. AI Assistance Declaration

AI assistance was used during repository inspection, documentation improvement, code review, and explanation development.

All generated documentation, results, and conclusions should be reviewed by the project owner before academic submission or public presentation.

---

## 20. Disclaimer

This project is intended for educational and academic research purposes.

It must not be used as a standalone production-grade email security gateway.

Production environments should use layered security controls, including secure email gateways, email authentication, threat intelligence, URL and attachment scanning, endpoint protection, monitoring, and human analyst review.
