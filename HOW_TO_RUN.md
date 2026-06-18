# How To Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add raw CSV files

Put raw CSV files into:

```text
data/raw/
```

Supported schemas:

```text
sender,receiver,date,subject,body,label,urls
sender,receiver,date,subject,body,urls,label
subject,body,label
text_combined,label
```

## 3. Standardize datasets

```bash
python standardize_datasets.py --input-dir data/raw --output data/processed/phishing_email_standardized.csv
```

Outputs:

```text
data/processed/phishing_email_standardized.csv
data/processed/standardization_report.csv
```

## 4. Run EDA

```bash
python eda_standardized.py --input data/processed/phishing_email_standardized.csv
```

Outputs:

```text
results/eda_summary.csv
results/class_distribution.png
results/text_length_distribution.png
```

## 5. Train models

```bash
python train_optimized.py --input data/processed/phishing_email_standardized.csv
```

Outputs:

```text
results/metrics_optimized.csv
results/logistic_regression_error_analysis.csv
models/naive_bayes_pipeline.pkl
models/logistic_regression_pipeline.pkl
models/linear_svm_pipeline.pkl
```

## 6. Run Streamlit app

```bash
streamlit run app.py
```

## Data leakage note

`train_optimized.py` performs the stratified train/test split before fitting
TF-IDF. Because TF-IDF is inside each sklearn `Pipeline`, it is fit on training
text only and then reused to transform the test set.
