# Phishing Email Detection and Explainable Risk Scoring

![Python](https://img.shields.io/badge/Python-Machine%20Learning-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikit-learn\&logoColor=white)
![Purpose](https://img.shields.io/badge/Purpose-Educational-green.svg)

An end-to-end machine-learning project for detecting phishing or suspicious email content and generating explainable risk scores.

The project standardizes multiple email datasets, performs exploratory data analysis, extracts TF-IDF and security-related metadata features, compares several classical machine-learning models, and provides a Streamlit application for single-email analysis.

The selected deployment model is **Logistic Regression with TF-IDF and metadata features** because it supports probability-based risk scoring and coefficient-based explanations.

---

## Project Workflow

```text
Raw email CSV files
        ↓
Dataset standardization
        ↓
Exploratory data analysis
        ↓
Text cleaning
        ↓
TF-IDF and metadata extraction
        ↓
Model training and evaluation
        ↓
Risk scoring and explainability
        ↓
Streamlit application
```

---

## Security Task

The project uses binary classification:

| Label | Meaning                      |
| ----: | ---------------------------- |
|   `0` | Legitimate or benign email   |
|   `1` | Phishing or suspicious email |

During dataset standardization, positive labels such as `phishing`, `spam`, `malicious`, and similar values are mapped to class `1`.

Therefore, the positive class may contain spam or other malicious messages in addition to confirmed credential-phishing emails.

The project evaluates models using:

* Precision
* Recall
* F1 score
* ROC-AUC
* PR-AUC / Average Precision
* Confusion Matrix
* Accuracy as supporting information

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   └── README.md
├── src/
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── risk_scoring.py
│   ├── text_cleaning.py
│   └── utils.py
├── app.py
├── eda_standardized.py
├── standardize_datasets.py
├── train_optimized.py
├── requirements.txt
├── HOW_TO_RUN.md
└── REPORT.md
```

### Main files

| File                         | Purpose                                                 |
| ---------------------------- | ------------------------------------------------------- |
| `standardize_datasets.py`    | Standardizes dataset schemas, labels, and email content |
| `eda_standardized.py`        | Performs exploratory data analysis                      |
| `train_optimized.py`         | Trains, evaluates, explains, and saves the models       |
| `app.py`                     | Runs the Streamlit application                          |
| `src/text_cleaning.py`       | Cleans and normalizes email text                        |
| `src/feature_engineering.py` | Extracts metadata features                              |
| `src/explainability.py`      | Calculates feature contributions                        |
| `src/risk_scoring.py`        | Generates predictions, risk scores, and risk levels     |

---

## Dataset

Place the following course dataset files in:

```text
data/raw/
```

Expected file names:

```text
CEAS_08.csv
Enron.csv
Nazario.csv
```

Raw and processed datasets are not included in the repository.

Run:

```bash
python standardize_datasets.py
```

The standardized dataset is written to:

```text
data/processed/phishing_email_standardized.csv
```

The canonical output fields are:

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

---

## Text Preprocessing

Text preprocessing is implemented in:

```text
src/text_cleaning.py
```

The cleaning process:

* Decodes HTML entities.
* Converts text to lowercase.
* Removes HTML tags.
* Replaces URLs with `url`.
* Replaces email addresses with `email`.
* Replaces numbers with `num`.
* Removes unsupported characters.
* Normalizes whitespace.

The current cleaning rule mainly supports English text because only lowercase letters from `a` to `z` are retained after normalization.

---

## Feature Engineering

### TF-IDF text features

The project uses word unigrams and bigrams.

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

### Metadata features

The final model also uses:

* `n_links`
* `has_ip_url`
* `html_tag_count`
* `html_ratio`
* `body_len`
* `subject_len`
* `n_exclamation`
* `n_upper_words`
* `sender_has_email`
* `url_count_from_column`

Metadata features are standardized with `StandardScaler`.

---

## Models

The project compares:

1. Multinomial Naive Bayes
2. Text-only Logistic Regression
3. Linear Support Vector Machine
4. Logistic Regression with text and metadata
5. Random Forest with compressed text and metadata

The selected deployment model is:

```text
logistic_regression_text_metadata
```

It is saved as:

```text
models/phishing_logreg_text_metadata.pkl
```

Linear SVM is included as a strong comparison model, but it does not provide `predict_proba()` directly.

---

## Risk Scoring and Explainability

The application converts the model probability into a risk score:

```text
risk_score = P(phishing) × 100
```

Risk levels:

|          Score | Level    |
| -------------: | -------- |
|       Below 40 | Low      |
| 40 to below 70 | Medium   |
| 70 to below 90 | High     |
|   90 or higher | Critical |

The application classifies an email as phishing when:

```text
P(phishing) >= 0.70
```

Logistic Regression explanations use:

```text
contribution = transformed feature value × model coefficient
```

Positive contributions push the prediction toward the positive class. Negative contributions push it toward the legitimate class.

---

## Installation and Usage

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run the project in this order:

```bash
python standardize_datasets.py
python eda_standardized.py
python train_optimized.py
streamlit run app.py
```

Detailed setup and troubleshooting instructions are available in:

```text
HOW_TO_RUN.md
```

Technical explanations and evaluation methodology are available in:

```text
REPORT.md
```

---

## Generated Artifacts

The scripts generate files inside:

```text
data/processed/
models/
results/
```

Typical artifacts include:

* Standardized datasets
* Trained model pipelines
* Classification metrics
* Confusion matrices
* ROC and Precision–Recall curves
* Duplicate and near-duplicate checks
* Source-holdout evaluation
* Error samples
* Model coefficients
* Example explanations

Generated artifacts are excluded from Git and must be created locally.

---

## Limitations

* The positive class may contain spam or other malicious emails in addition to phishing.
* Similar email templates may occur in both training and testing data.
* Random train/test splits may produce optimistic results.
* Performance may decrease on previously unseen dataset sources.
* Metadata availability differs between datasets.
* Model probabilities are not separately calibrated.
* The application threshold differs from the default Logistic Regression evaluation threshold.
* Text preprocessing mainly supports English email content.
* The system does not inspect attachments.
* The system does not fully analyze email headers.
* The system does not verify SPF, DKIM, or DMARC.
* The system does not query live URL, domain, or sender reputation services.
* The application does not connect directly to Gmail or Outlook.
* The application does not quarantine or delete messages.

---

## Disclaimer

This project is intended for educational and academic research purposes.

The predictions, probabilities, risk levels, and explanations must not be used as a standalone production email-security system.

A legitimate prediction does not guarantee that an email is safe, and a phishing prediction does not prove that an email is malicious.

Production environments should combine machine-learning detection with secure email gateways, email authentication, threat intelligence, URL and attachment scanning, endpoint protection, and human analyst review.

---

## About

**Course:** AIC211
**Project:** Phishing Email Detection and Explainable Risk Scoring
