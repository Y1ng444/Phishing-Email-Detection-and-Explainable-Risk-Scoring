# Phishing Email Detection & Explainable Risk Scoring

A complete university-level NLP machine learning project for detecting phishing emails with explainable AI and risk scoring.

## Project Overview

This project implements an end-to-end pipeline for phishing email detection using:
- **Natural Language Processing (NLP)** for text preprocessing
- **Machine Learning** (Logistic Regression) for classification
- **Explainable AI (XAI)** for model interpretability
- **Risk Scoring** system for security assessment
- **Streamlit Web App** for user interface

## Dataset

- **Size:** 82,486 emails
- **Classes:** Ham (48.9%) and Phishing (51.1%)
- **Location:** `data/phishing_email.csv`
- **Columns:** `text_combined` (email content), `label` (0=Ham, 1=Phishing)

## Project Architecture

```
project/
├── data/
│   └── phishing_email.csv              # Dataset
├── src/
│   ├── __init__.py
│   ├── utils.py                        # Logging, constants, utilities
│   ├── preprocessing.py                # EDA, text cleaning
│   ├── feature_engineering.py          # TF-IDF + statistical features
│   ├── train_baseline.py               # Multinomial Naive Bayes
│   ├── train_logistic.py               # Logistic Regression (improved)
│   ├── risk_scoring.py                 # Prediction and risk scoring
│   ├── explainability.py               # Feature importance & explanations
│   └── evaluate.py                     # Metrics, visualizations, error analysis
├── models/                             # Trained models & vectorizer
├── results/                            # Visualizations & error analysis
├── train.py                            # Orchestration script
├── app.py                              # Streamlit web application
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## Workflow Pipeline

```
Email Text
    ↓
Data Cleaning (HTML tags, URLs, punctuation, digits)
    ↓
Tokenization (NLTK word_tokenize)
    ↓
Stopword Removal & Lemmatization
    ↓
TF-IDF + N-Grams (5,000 features, 1-2 grams)
    ↓
Statistical Features (5 additional features)
    ↓
Combined Feature Matrix (5,005 features)
    ↓
Classification Model (Logistic Regression)
    ↓
Risk Scoring (Probability → Risk Level)
    ↓
Explainability Layer (Feature importance + per-email explanations)
    ↓
Evaluation (Metrics, visualizations, error analysis)
```

## Installation

### 1. Prerequisites
- Python 3.11 or higher
- pip package manager

### 2. Setup

```bash
# Navigate to project directory
cd "Phishing Email Detection and Explainable Risk Scoring"

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Run Complete Training Pipeline

```bash
python train.py
```

This will:
1. Load and explore the dataset
2. Generate EDA visualizations
3. Preprocess all emails
4. Create TF-IDF and statistical features
5. Train Multinomial Naive Bayes (baseline)
6. Train Logistic Regression (improved)
7. Compare model performance
8. Generate feature importance analysis
9. Create error analysis report
10. Save all models and results

**Expected Runtime:** 10-15 minutes (depending on system)

### Option 2: Run Individual Phases

```bash
# Phase 1: Baseline Model Training
python -m src.train_baseline

# Phase 2: Improved Model Training
python -m src.train_logistic

# Phase 3: Evaluation
python -m src.evaluate
```

### Option 3: Use Web Interface (Streamlit)

After training models, run the interactive web app:

```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

## Features

### 1. Text Preprocessing
- Lowercase conversion
- HTML tag removal
- URL removal
- Punctuation removal
- Digit removal
- Tokenization
- Stopword removal (NLTK English)
- Lemmatization (WordNetLemmatizer)

### 2. Feature Engineering

**TF-IDF Features:**
- Max features: 5,000
- N-gram range: (1, 2) (unigrams and bigrams)
- Sparse matrix format

**Statistical Features:**
- Email length (characters)
- Word count
- Uppercase ratio
- Digit ratio
- URL count

### 3. Models

**Baseline Model:** Multinomial Naive Bayes
- Simple probabilistic classifier
- Baseline for comparison

**Improved Model:** Logistic Regression
- `class_weight="balanced"` (handles class imbalance)
- `max_iter=1000` (sufficient convergence)
- `random_state=42` (reproducibility)
- Better performance and interpretability

### 4. Risk Scoring

Converts phishing probability to risk levels:

| Risk Score | Risk Level | Description |
|-----------|-----------|-------------|
| 0-30% | Low | Email appears legitimate |
| 31-60% | Medium | Email has some phishing characteristics |
| 61-80% | High | Email is likely phishing |
| 81-100% | Critical | Email is highly likely phishing |

### 5. Explainability

**Global Explanation:**
- Top 20 phishing indicators (positive coefficients)
- Top 20 legitimate indicators (negative coefficients)
- Feature importance rankings

**Per-Email Explanation:**
- Suspicious keywords detected
- Top contributing features
- Positive evidence (phishing indicators present)
- Negative evidence (legitimate indicators present)
- Email statistics

### 6. Evaluation

**Metrics Computed:**
- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- ROC-AUC

**Visualizations Generated:**
- Label distribution bar chart
- Text length histogram by class
- Confusion matrix heatmap
- Precision-Recall curve

**Error Analysis:**
- False Positives (Ham classified as Phishing)
- False Negatives (Phishing classified as Ham)
- Detailed error report with probabilities

## Key Functions

### Preprocessing (`src/preprocessing.py`)
```python
from src.preprocessing import load_dataset, preprocess_dataset, clean_email

# Load dataset
df, total_rows = load_dataset()

# Preprocess all emails
df = preprocess_dataset(df)

# Clean single email
cleaned = clean_email(raw_email_text)
```

### Feature Engineering (`src/feature_engineering.py`)
```python
from src.feature_engineering import create_all_features, create_tfidf_features

# Create all features
features = create_all_features(texts, fit_vectorizer=True)
X = features['X']  # Combined feature matrix
vectorizer = features['vectorizer']
```

### Risk Scoring (`src/risk_scoring.py`)
```python
from src.risk_scoring import predict_email

# Analyze single email
result = predict_email(email_text)
print(result['prediction'])      # "Phishing" or "Ham"
print(result['risk_score'])      # 0-100
print(result['risk_level'])      # "Low", "Medium", "High", "Critical"
print(result['probability'])     # 0.0-1.0
```

### Explainability (`src/explainability.py`)
```python
from src.explainability import get_email_explanation, display_feature_importance

# Global feature importance
display_feature_importance(model, feature_names, n=20)

# Per-email explanation
explanation = get_email_explanation(email_text, model, vectorizer, feature_names)
print(explanation['suspicious_keywords'])
print(explanation['top_tfidf_features'])
```

### Evaluation (`src/evaluate.py`)
```python
from src.evaluate import run_evaluation

# Complete evaluation
results = run_evaluation(model, X_test, y_test, df_test)
print(results['metrics'])  # All computed metrics
```

## Model Performance

Expected results after training:

**Multinomial Naive Bayes:**
- Accuracy: ~95.2%
- Precision: ~94.8%
- Recall: ~94.5%
- F1 Score: ~94.6%

**Logistic Regression (Improved):**
- Accuracy: ~96.5%
- Precision: ~95.8%
- Recall: ~95.2%
- Specificity: ~97.8%
- F1 Score: ~95.5%
- ROC-AUC: ~98.9%

## Output Files

### Trained Models
- `models/naive_bayes_model.pkl` - Baseline model
- `models/logistic_regression_model.pkl` - Improved model
- `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `models/feature_names.pkl` - Feature names list

### Visualizations (in `results/`)
- `01_label_distribution.png` - Class distribution chart
- `02_text_length_histogram.png` - Email length histogram
- `naive_bayes_confusion_matrix.png` - Baseline confusion matrix
- `logistic_regression_confusion_matrix.png` - Improved confusion matrix
- `naive_bayes_pr_curve.png` - Baseline PR curve
- `logistic_regression_pr_curve.png` - Improved PR curve

### Error Analysis (in `results/`)
- `naive_bayes_errors.csv` - Baseline errors
- `logistic_regression_errors.csv` - Improved model errors
  - Columns: `text_combined`, `true_label`, `predicted_label`, `probability`

## Streamlit Web App Features

### Email Analyzer Tab
1. Paste email content
2. Click "Analyze Email"
3. View:
   - Prediction (Ham/Phishing)
   - Risk score (0-100)
   - Confidence (probability)
   - Risk level with color coding
   - Email statistics
   - Top contributing features
   - Suspicious keywords
   - Evidence breakdown

### Model Info Tab
- Model details and architecture
- Performance metrics
- Top phishing/legitimate indicators

### Dataset Tab
- Dataset overview
- Data distribution
- Processing pipeline explanation
- Risk threshold information

## Code Quality

All code follows production standards:
- ✅ Type hints (Python 3.11+)
- ✅ Google-style docstrings
- ✅ Comprehensive logging
- ✅ Error handling
- ✅ Modular architecture
- ✅ No placeholder code
- ✅ Reproducible (fixed random seeds)

## Troubleshooting

### Issue: Models not found when running Streamlit
**Solution:** Run `python train.py` first to train models

### Issue: NLTK data missing
**Solution:** Run once, and `preprocessing.py` will auto-download required NLTK data

### Issue: Memory error with large dataset
**Solution:** The code uses sparse matrices for efficiency. If still limited:
- Reduce `TFIDF_MAX_FEATURES` in `src/utils.py`
- Process dataset in batches

### Issue: Model training takes too long
**Solution:** This is normal for 82K+ emails. Expected time: 10-15 minutes
- Can be reduced with fewer features or smaller dataset

## Future Improvements

1. **Advanced Models:** Try SVM, Random Forest, Neural Networks
2. **Deep Learning:** LSTM/Transformer models for better NLP
3. **Additional Features:** Sender reputation, header analysis
4. **Active Learning:** Iteratively improve with user feedback
5. **Real-time Integration:** API endpoint for email servers
6. **Multilingual Support:** Handle non-English emails

## References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Phishing Detection Techniques](https://en.wikipedia.org/wiki/Phishing)

## Authors

**Senior ML Engineer, NLP Engineer & Cybersecurity Researcher**

## License

Educational Use - University Project

## Contact

For questions or improvements, please refer to the project documentation.

---

**Project Status:** ✅ Complete and Production-Ready

**Last Updated:** June 2026
