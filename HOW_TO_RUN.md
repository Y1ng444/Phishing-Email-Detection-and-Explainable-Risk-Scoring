# How To Run

## 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 2. Install requirements

```bash
pip install -r requirements.txt
```

## 3. Place raw CSV files in `data/raw/`

Supported schemas:

- `subject, body, label`
- `subject, body, label, urls`
- `sender, receiver, date, subject, body, label, urls`
- `sender, receiver, date, subject, body, urls, label`
- `text_combined, label`

## 4. Run standardization

```bash
python standardize_datasets.py
```

Output:

```text
data/processed/phishing_email_standardized.csv
```

## 5. Run EDA

```bash
python eda_standardized.py
```

Outputs:

```text
results/class_distribution.png
results/text_length_distribution.png
```

## 6. Train models

```bash
python train_optimized.py
```

Outputs include:

```text
models/phishing_logreg_tfidf.pkl
results/metrics_optimized.csv
results/error_samples.csv
results/sample_predictions.csv
results/top_phishing_indicators.csv
results/top_legitimate_indicators.csv
```

## 7. Launch Streamlit app

```bash
streamlit run app.py
```
