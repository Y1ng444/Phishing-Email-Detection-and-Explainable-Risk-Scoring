# How to Standardize and Train

## Step 1

Put raw CSV files into:

```text
data/raw/
```

## Step 2

Standardize raw datasets:

```bash
python standardize_datasets.py --input-dir data/raw --output data/processed/phishing_email_standardized.csv
```

## Step 3

Run EDA:

```bash
python eda_standardized.py --input data/processed/phishing_email_standardized.csv
```

## Step 4

Train optimized sklearn pipelines:

```bash
python train_optimized.py --input data/processed/phishing_email_standardized.csv
```

## Step 5

Run the Streamlit app:

```bash
streamlit run app.py
```

## Optional NLTK Data

The optimized workflow does not require NLTK. The legacy preprocessing module
does not download NLTK resources during import; if you use the legacy scripts and
want full NLTK preprocessing, install resources manually:

```bash
python -m nltk.downloader stopwords punkt wordnet
```
