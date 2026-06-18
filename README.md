# Phishing Email Detection and Explainable Risk Scoring

This project is a reproducible AI for Cybersecurity pipeline for binary phishing
email classification.

The system standardizes raw CSV datasets, performs EDA, trains baseline and
improved text classifiers, evaluates security-focused metrics, saves error
analysis, and provides a Streamlit demo with simple coefficient-based
explanations.

## Objective

Detect whether an email is:

- `0`: ham, legitimate, benign, safe, or non-phishing
- `1`: phishing, malicious, spam, fraud, or scam

The prediction is converted into a 0-100 risk score with Low, Medium, and High
risk levels.

## Course-Aligned Methods

The optimized workflow uses only methods shown in the provided course slides and
sample code:

- Text cleaning and tokenization
- TF-IDF with unigram and bigram features
- sklearn `Pipeline`
- Multinomial Naive Bayes baseline
- Logistic Regression improved model
- Linear SVM comparison model
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion matrices, ROC curves, PR curves
- Coefficient-based feature explanations

The project does not use SHAP, LIME, transformers, deep learning, XGBoost,
external APIs, LLM/RAG, or other methods outside the requested scope.

## Dataset Format

Put raw CSV files in:

```text
data/raw/
```

Supported raw schemas:

```text
sender,receiver,date,subject,body,label,urls
sender,receiver,date,subject,body,urls,label
subject,body,label
text_combined,label
```

All files are standardized to:

```text
source_file,row_id,sender,receiver,date,subject,body,urls,text_combined,label
```

Missing columns are filled with empty strings. Invalid labels are dropped.
Duplicates are dropped using `text_combined` and `label`.

## Folder Structure

```text
.
├── app.py
├── standardize_datasets.py
├── eda_standardized.py
├── train_optimized.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── RUBRIC_ALIGNMENT.md
├── models/
├── results/
└── src/
    ├── explainability.py
    ├── risk_scoring.py
    └── text_cleaning.py
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Standardize raw CSV files:

```bash
python standardize_datasets.py --input-dir data/raw --output data/processed/phishing_email_standardized.csv
```

Run EDA:

```bash
python eda_standardized.py --input data/processed/phishing_email_standardized.csv
```

Train optimized models:

```bash
python train_optimized.py --input data/processed/phishing_email_standardized.csv
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Generated Outputs

Standardization:

- `data/processed/phishing_email_standardized.csv`
- `data/processed/standardization_report.csv`

EDA:

- `results/eda_summary.csv`
- `results/class_distribution.png`
- `results/text_length_distribution.png`

Training:

- `results/metrics_optimized.csv`
- `results/naive_bayes_confusion_matrix.png`
- `results/logistic_regression_confusion_matrix.png`
- `results/linear_svm_confusion_matrix.png`
- `results/roc_curve_comparison.png`
- `results/pr_curve_comparison.png`
- `results/logistic_regression_error_analysis.csv`

Models:

- `models/naive_bayes_pipeline.pkl`
- `models/logistic_regression_pipeline.pkl`
- `models/linear_svm_pipeline.pkl`

## Data Leakage Control

The optimized training script splits the dataset with stratification before any
TF-IDF fitting occurs. Each model is an sklearn `Pipeline`, so the TF-IDF
vectorizer is fit only on `X_train` during `pipeline.fit(X_train, y_train)`.
The test set is transformed only through the fitted training vocabulary during
evaluation.

## AI Tool Usage Declaration

AI assistance was used to help refactor code, improve documentation, and align
the implementation with the course rubric. The implemented workflow uses
standard sklearn methods from the course materials and should be reviewed and
understood by the project team before submission.

## Limitations

- Label quality depends on the original CSV datasets.
- Duplicate removal is text-based and may not catch near-duplicates.
- The Streamlit app is a demo, not a production email security gateway.
- Risk scores are model scores, not guaranteed real-world probabilities.
