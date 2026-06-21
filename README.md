# Phishing Email Detection and Explainable Risk Scoring

## Project Overview

This course project builds an end-to-end classical machine learning pipeline for
binary phishing email classification. It standardizes raw email CSV files,
cleans email text, extracts TF-IDF unigram and bigram features, trains three
models, evaluates security-focused metrics, and explains Logistic Regression
predictions using model coefficients.

## Cybersecurity Problem

The task is to classify an email as:

- `0`: legitimate
- `1`: phishing

The final app converts the Logistic Regression phishing probability into a
simple 0-100 risk score.

## Dataset Schema

Raw CSV files should be placed in `data/raw/`. Supported schemas include:

- `subject, body, label`
- `subject, body, label, urls`
- `sender, receiver, date, subject, body, label, urls`
- `sender, receiver, date, subject, body, urls, label`
- `text_combined, label`

The standardized dataset is saved to:

```text
data/processed/phishing_email_standardized.csv
```

It contains `text_combined` and `label`, plus available metadata columns:
`sender`, `receiver`, `date`, `subject`, `body`, and `urls`.

## Methodology

1. Standardize column names and binary labels.
2. Build `text_combined` from subject, body, and URLs when needed.
3. Remove missing labels, empty text, exact duplicates, conflicting duplicate
   texts, and repeated `text_combined` rows.
4. Clean text with regex-based preprocessing:
   lowercase, remove HTML tags, replace URLs, replace email addresses, replace
   numbers, remove special characters, and normalize whitespace.
5. Extract TF-IDF features with unigrams and bigrams.
6. Split data with stratified train/test split before fitting TF-IDF.
7. Train and compare classical machine learning models.

## Models

- Baseline: Multinomial Naive Bayes
- Improved model: Logistic Regression
- Comparison model: Linear SVM

Linear SVM achieved the highest F1-score in the latest run, but Logistic
Regression is selected as the main explainable risk scoring model because it
provides `predict_proba` probabilities and interpretable coefficients.

## Metrics

The training script reports:

- Accuracy
- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion matrix

Latest generated results from `results/metrics_optimized.csv`:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.9604 | 0.9874 | 0.9361 | 0.9611 | 0.9961 |
| Logistic Regression | 0.9882 | 0.9869 | 0.9904 | 0.9887 | 0.9990 |
| Linear SVM | 0.9950 | 0.9947 | 0.9958 | 0.9952 | 0.9997 |

## Explainable Risk Scoring

The final Logistic Regression model is saved to:

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

The explanation layer uses Logistic Regression coefficients and TF-IDF feature
values to show global phishing/legitimate indicators and the strongest phishing
terms active in a single email.

## How To Run

```bash
python standardize_datasets.py
python eda_standardized.py
python train_optimized.py
streamlit run app.py
```

See `HOW_TO_RUN.md` for setup steps.

## Final Project Structure

```text
phishing-email-detection/
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── text_cleaning.py
│   ├── explainability.py
│   ├── risk_scoring.py
│   └── utils.py
├── results/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── standardize_datasets.py
├── eda_standardized.py
├── train_optimized.py
├── app.py
├── README.md
├── HOW_TO_RUN.md
├── REPORT.md
├── requirements.txt
└── .gitignore
```

## Limitations

- The model depends on the quality and labels of the input datasets.
- Duplicate removal catches exact repeated text, not near-duplicates.
- Coefficient explanations describe model behavior, not guaranteed human intent.
- Risk scores are model probabilities and should be interpreted cautiously.
