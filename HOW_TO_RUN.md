# How to Run

This repository keeps raw data, processed data, trained models, and generated
results out of Git. Reproduce the project locally with the steps below.

## 1. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Add Raw Dataset Files

Place CSV files in:

```text
data/raw/
```

Expected course files:

```text
data/raw/CEAS_08.csv
data/raw/Enron.csv
data/raw/Nazario.csv
```

Supported schemas:

- `subject,body,label`
- `subject,body,label,urls`
- `sender,receiver,date,subject,body,label,urls`
- `sender,receiver,date,subject,body,urls,label`
- `text_combined,label`

## 4. Standardize Datasets

```bash
python standardize_datasets.py
```

This creates:

```text
data/processed/phishing_email_standardized.csv
```

The standardized CSV includes `source_file`, `text_combined`, and `label`.

## 5. Run EDA

```bash
python eda_standardized.py
```

This creates:

```text
results/class_distribution.png
results/text_length_distribution.png
results/missing_values.csv
results/dataset_source_distribution.csv
results/duplicate_statistics.csv
results/top_terms_by_class.csv
```

## 6. Train and Evaluate Models

```bash
python train_optimized.py
```

This trains Naive Bayes, Logistic Regression, and Linear SVM, then saves:

```text
models/phishing_logreg_tfidf.pkl
results/metrics_optimized.csv
results/metrics_cleaned_dedup.csv
results/near_duplicate_check.csv
results/source_holdout_metrics.csv
results/cleaned_duplicate_check.csv
results/error_samples.csv
results/sample_predictions.csv
results/top_phishing_indicators.csv
results/top_legitimate_indicators.csv
results/*_confusion_matrix.png
results/roc_curve_comparison.png
results/pr_curve_comparison.png
```

On the full course dataset this step can take several minutes because it also
runs robustness checks and source-holdout experiments.

## 7. Launch the Streamlit Demo

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Paste an email or use the example
button, then click Analyze Email.

## Troubleshooting

- If `app.py` cannot find `models/phishing_logreg_tfidf.pkl`, run
  `python train_optimized.py`.
- If `train_optimized.py` cannot find
  `data/processed/phishing_email_standardized.csv`, run
  `python standardize_datasets.py`.
- If `standardize_datasets.py` finds no CSV files, confirm the raw CSV files are
  inside `data/raw/`.
- If imports fail, activate the virtual environment and run
  `pip install -r requirements.txt`.
- If Streamlit does not start, confirm `streamlit` is installed in the active
  environment.
