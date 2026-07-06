# Phishing Email Detection and Explainable Risk Scoring

## Executive Summary

This project implements a course-aligned AI for Cybersecurity pipeline for
phishing email detection. It uses classical machine learning with TF-IDF text
features, a Naive Bayes baseline, a Logistic Regression final model, and a
Linear SVM comparison model. The final system produces a phishing probability,
a 0-100 risk score, and coefficient-based explanations for active terms in an
email.

The latest local run used the raw files currently available in `data/raw/`:
`CEAS_08.csv`, `Enron.csv`, and `Nazario.csv`. After standardization and
cleaning, the dataset contained 69,941 usable emails: 32,763 legitimate and
37,178 phishing. Random split metrics were strong, but robustness checks show
that the random split is optimistic because many test emails are highly similar
to training emails. Source-holdout evaluation is therefore included as a more
realistic generalization check.

## 1. Introduction and Problem Statement

Phishing email detection is a binary text classification problem:

- `0`: legitimate / benign email
- `1`: phishing email

The goal is to classify email content and provide an interpretable risk score.
The project is designed as a reproducible course prototype rather than a
production email security gateway.

## 2. Threat Context and Security Use Case

Phishing emails are used for credential theft, malware delivery, financial
fraud, and business email compromise. A useful detector should identify
suspicious emails before they reach users while keeping false alerts low enough
that users still trust the system.

In this setting, false negatives are especially dangerous because a phishing
email incorrectly marked as safe can reach the victim. False positives are also
important because they can disrupt legitimate communication and cause alert
fatigue.

## 3. Dataset and Data Understanding

Raw CSV files are placed in `data/raw/`. The current local run used:

| Source | Rows after cleaning | Share |
| --- | ---: | ---: |
| CEAS_08.csv | 38,963 | 55.71% |
| Enron.csv | 29,423 | 42.07% |
| Nazario.csv | 1,555 | 2.22% |

Class distribution:

| Label | Meaning | Count |
| --- | --- | ---: |
| 0 | Legitimate | 32,763 |
| 1 | Phishing | 37,178 |

The standardization step preserves metadata when available and adds a
`source_file` column. This column is required for source-holdout experiments,
which test whether a model trained on one source can generalize to another.

## 4. Preprocessing and Feature Engineering

The preprocessing is intentionally simple and reproducible:

- Convert text to lowercase.
- Remove HTML tags.
- Replace URLs with `url`.
- Replace email addresses with `email`.
- Replace numbers with `num`.
- Remove special characters.
- Normalize whitespace.

Feature extraction uses `TfidfVectorizer` with unigram and bigram features,
`max_features=20000`, `min_df=2`, `max_df=0.95`, and `sublinear_tf=True`.
TF-IDF is inside each sklearn `Pipeline`, so the vectorizer is fit only on the
training split and then applied to the test split.

## 5. Model Design

Three classical ML models are trained:

- Naive Bayes: baseline. It is fast, common for text classification, and gives a
  simple reference point.
- Logistic Regression: final model. It supports `predict_proba()` for risk
  scoring and has interpretable coefficients for explainability.
- Linear SVM: comparison model. It is often strong for sparse TF-IDF features,
  but it is not selected as the final risk scoring model because probability
  output and coefficient-based probability explanations are less direct.

## 6. Experiments and Evaluation

The random split benchmark uses stratified train/test split with
`test_size=0.2` and `random_state=42`. It should be interpreted as an
in-distribution benchmark, not as the only evidence of real-world performance.

Latest random split results from `results/metrics_optimized.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9578 | 0.9901 | 0.9299 | 0.9591 | 0.9961 | 0.9960 |
| Logistic Regression | 0.9862 | 0.9857 | 0.9884 | 0.9870 | 0.9990 | 0.9991 |
| Linear SVM | 0.9912 | 0.9915 | 0.9919 | 0.9917 | 0.9994 | 0.9994 |

Metric interpretation:

- Precision answers: when the model flags phishing, how often is it correct?
- Recall answers: out of all phishing emails, how many did the model catch?
- F1 balances precision and recall.
- PR-AUC is important when focusing on phishing detection quality across
  thresholds.
- ROC-AUC summarizes ranking quality across false-positive and true-positive
  rates.

## 7. Robustness and Generalization Check

### Cleaned-Text Duplicate Check

Exact duplicate removal found no duplicate `text_combined` rows after
standardization, but cleaning exposed many template-like duplicates:

| Metric | Value |
| --- | ---: |
| Rows before cleaned dedup | 69,940 |
| Duplicate rows by original `text_combined` | 0 |
| Duplicate rows by `text_cleaned` | 7,355 |
| Conflicting cleaned texts | 0 |
| Rows after drop duplicate by `text_cleaned` | 62,585 |

Cleaned-dedup benchmark from `results/metrics_cleaned_dedup.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9665 | 0.9757 | 0.9564 | 0.9660 | 0.9955 | 0.9945 |
| Logistic Regression | 0.9853 | 0.9816 | 0.9889 | 0.9853 | 0.9985 | 0.9983 |
| Linear SVM | 0.9898 | 0.9867 | 0.9928 | 0.9897 | 0.9992 | 0.9991 |

The cleaned-dedup results remain high, but slightly lower than the original
random split. This suggests duplicate cleaning helps, but it does not fully
remove template overlap.

### Near-Duplicate Similarity Check

After the random train/test split, TF-IDF cosine similarity was measured from
each test email to its most similar training email:

| Metric | Value |
| --- | ---: |
| Train rows | 55,952 |
| Test rows | 13,989 |
| Mean max similarity | 0.6836 |
| Median max similarity | 0.7411 |
| Test samples with similarity > 0.90 | 4,268 |
| Ratio with similarity > 0.90 | 30.51% |
| Test samples with similarity > 0.95 | 3,247 |
| Ratio with similarity > 0.95 | 23.21% |

This is a strong warning sign that random split results are optimistic. The
model may be learning recurring templates and source-specific wording, not only
general phishing behavior.

### Source Holdout Evaluation

Source holdout is stricter because the model is trained on one dataset source
and tested on another.

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CEAS + Enron -> Nazario | 0.5878 | n/a | 0.5878 | n/a | n/a | n/a | Nazario test set has only phishing labels, so phishing recall is reported. |
| CEAS -> Enron | 0.7731 | 0.8944 | 0.5912 | 0.7118 | 0.8428 | 0.8557 | Two-class external test set. |
| Enron -> CEAS | 0.8222 | 0.7617 | 0.9903 | 0.8611 | 0.9164 | 0.9226 | Two-class external test set. |

These results show that generalization across sources is meaningfully weaker
than random split performance. The model is useful as a prototype, but it should
not be presented as production-ready without broader external validation.

## 8. Explainable Risk Scoring

The final risk score is:

```text
risk_score = P(phishing) * 100
```

Risk levels:

- Low risk: score < 40
- Medium risk: 40 <= score < 70
- High risk: score >= 70

The explanation layer uses Logistic Regression coefficients:

```text
contribution = TF-IDF value * coefficient
```

For a single email, only active terms are shown, sorted by phishing contribution
in descending order. Global top phishing and legitimate indicators are saved to
CSV files under `results/`.

## 9. Error Analysis

`train_optimized.py` saves Logistic Regression errors to:

```text
results/error_samples.csv
```

False positives are legitimate emails predicted as phishing. False negatives
are phishing emails predicted as legitimate. In cybersecurity, false negatives
usually carry higher risk because they let attacks reach users. Error samples
should be reviewed to understand which phrases, templates, or source-specific
patterns confuse the classifier.

## 10. Security Discussion

The model is appropriate for demonstrating text classification, risk scoring,
and explainability. However, phishing is adaptive. Attackers can change wording,
avoid suspicious keywords, use images or attachments, or impersonate internal
communications. A production system would need richer features, continuous
monitoring, adversarial testing, user feedback loops, and integration with
email security controls.

## 11. Limitations, Ethics, and Operational Risks

- Random split metrics can be inflated by duplicate or near-duplicate templates.
- Source-holdout performance is lower and should be treated as more realistic.
- Labels may contain noise or dataset-specific conventions.
- The model only uses text patterns and does not inspect attachments, sender
  reputation, authentication headers, or URL reputation.
- The probability score is not calibrated with a separate calibration method.
- False positives can interrupt legitimate communication.
- False negatives can expose users to real attacks.
- The project should not be deployed as an autonomous production gateway.

## 12. Conclusion and Future Work

The project satisfies the course requirements for an end-to-end AI pipeline for
cybersecurity: data standardization, EDA, baseline model, main model, comparison
model, quantitative evaluation, error analysis, risk scoring, explainability,
demo, reproducibility, and robustness checks.

Future work:

- Add probability calibration and calibration plots.
- Add near-duplicate grouping before train/test split.
- Evaluate on more external phishing datasets.
- Add sender/domain/URL/header features.
- Add threshold tuning based on the cost of false negatives.
- Package the report as PDF/DOCX and prepare defense slides.

## 13. AI Tool Usage Declaration

AI assistance was used to review and improve the repository structure, code
quality, documentation, and robustness evaluation workflow. Metrics reported in
this document are generated from local script outputs and are not fabricated.
The project owner should still review the final code, report wording, and
course submission requirements before submission.

## 14. References

- scikit-learn documentation: TF-IDF, Naive Bayes, Logistic Regression, Linear
  SVM, model evaluation metrics, and pipelines.
- Streamlit documentation for the interactive demo.
- Course materials for AIC211 - AI for Cybersecurity.
- Public phishing email dataset documentation associated with CEAS, Enron, and
  Nazario sources, where applicable.
