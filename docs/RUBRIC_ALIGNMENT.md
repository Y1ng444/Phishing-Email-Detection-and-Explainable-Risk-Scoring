# Rubric Alignment

## Problem Definition and Security Context

The project detects phishing emails as a binary text classification task:

- `0`: ham / legitimate
- `1`: phishing / malicious

Security use: reduce missed phishing emails while understanding false positives
that may block or quarantine legitimate messages.

Relevant files:

- `README.md`
- `app.py`
- `src/risk_scoring.py`

## Data, EDA, and Data Quality

Raw CSV files with different schemas are standardized into one fixed schema.
The standardization report records rows loaded, rows kept, rows dropped, and
class distribution by source file.

Relevant files:

- `standardize_datasets.py`
- `eda_standardized.py`

Outputs:

- `data/processed/phishing_email_standardized.csv`
- `data/processed/standardization_report.csv`
- `results/eda_summary.csv`
- `results/class_distribution.png`
- `results/text_length_distribution.png`

## Preprocessing and Feature Engineering

The optimized workflow uses simple text cleaning and TF-IDF n-grams, matching
the NLP sample code style.

Relevant files:

- `src/text_cleaning.py`
- `train_optimized.py`

Methods:

- Lowercasing
- HTML cleanup
- URL and email token replacement
- Punctuation cleanup
- TF-IDF unigram and bigram features

## Model Design and Baseline

The project includes:

- Baseline: Multinomial Naive Bayes
- Improved model: Logistic Regression
- Additional comparison: Linear SVM

Relevant file:

- `train_optimized.py`

Saved models:

- `models/naive_bayes_pipeline.pkl`
- `models/logistic_regression_pipeline.pkl`
- `models/linear_svm_pipeline.pkl`

## Experiments and Evaluation

The optimized training script reports:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix counts

Plots:

- `results/naive_bayes_confusion_matrix.png`
- `results/logistic_regression_confusion_matrix.png`
- `results/linear_svm_confusion_matrix.png`
- `results/roc_curve_comparison.png`
- `results/pr_curve_comparison.png`

Metrics table:

- `results/metrics_optimized.csv`

## Error Analysis and Security Interpretation

False positives and false negatives from Logistic Regression are saved for
inspection.

Output:

- `results/logistic_regression_error_analysis.csv`

Columns:

- `text_combined`
- `true_label`
- `predicted_label`
- `phishing_probability_or_score`
- `error_type`

Operational interpretation:

- False positive: a legitimate email may be blocked or investigated.
- False negative: a phishing email may reach the user.

## Explainability and Risk Scoring

The explanation layer uses simple linear model coefficients and TF-IDF feature
names. It shows global phishing/ham indicators and active features in one email.

Relevant files:

- `src/explainability.py`
- `src/risk_scoring.py`
- `app.py`

Risk score:

- Converts probability or normalized decision score to 0-100.
- Levels: Low, Medium, High.

## Reproducibility

The workflow uses fixed random seeds and explicit CLI commands:

```bash
python standardize_datasets.py --input-dir data/raw --output data/processed/phishing_email_standardized.csv
python eda_standardized.py --input data/processed/phishing_email_standardized.csv
python train_optimized.py --input data/processed/phishing_email_standardized.csv
streamlit run app.py
```

Data leakage prevention:

- The stratified train/test split happens before fitting TF-IDF.
- TF-IDF is inside sklearn `Pipeline`, so it is fit only on training data.
- The test set is transformed using the already-fitted training vocabulary.

## Scope Control

The optimized workflow intentionally avoids:

- SHAP
- LIME
- Transformers
- Deep learning
- XGBoost
- External APIs
- LLM/RAG methods
