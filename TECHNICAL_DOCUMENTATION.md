# Technical Documentation: Implementation Details & Architecture

## Table of Contents
1. System Architecture
2. Module Descriptions
3. Data Pipelines
4. Model Training Process
5. API Reference
6. Configuration Guide
7. Troubleshooting

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────┐
│  Raw Email Data │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Data Preprocessing │ (preprocessing.py)
│ - HTML cleaning     │
│ - Tokenization      │
│ - Lemmatization     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │ (feature_engineering.py)
│ - TF-IDF (5000)     │
│ - Statistical (5)   │
│ - Total: 5005       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Class Balancing    │ (advanced_ml.py)
│  - SMOTE            │
│  - Stratified Split │
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│  Hyperparameter Tuning   │ (advanced_ml.py)
│  - GridSearchCV (5-fold) │
│  - Logistic Regression   │
│  - Naive Bayes           │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Model Training          │ (train_*.py)
│  - SMOTE-balanced data   │
│  - Stratified CV         │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Evaluation & Analysis   │ (evaluate.py)
│  - Metrics calculation   │
│  - Confusion matrices    │
│  - Error analysis        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Visualization           │ (visualization.py)
│  - Feature importance    │
│  - Model comparison      │
│  - Learning curves       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Risk Scoring Engine     │ (risk_scoring.py)
│  - Probability → Risk    │
│  - Explainability        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Web Application         │ (app.py)
│  - Streamlit Dashboard   │
│  - Email Analyzer        │
│  - Risk Visualization    │
└──────────────────────────┘
```

### 1.2 Module Dependency Graph

```
utils.py ──┐
           ├─→ preprocessing.py ──┐
           │                      │
           ├─→ feature_engineering.py ──┐
           │                            │
           ├─→ advanced_ml.py ──┐       │
           │                    │       │
           ├─→ train_baseline.py ──┐    │
           │                       │    │
           ├─→ train_logistic.py ──┤    │
           │                       │    │
           ├─→ evaluate.py ────────┤    │
           │                       │    │
           ├─→ visualization.py ───┤    │
           │                       │    │
           ├─→ risk_scoring.py ────┼────┼──→ train.py
           │                       │    │
           └─→ explainability.py ──┴────┴──→ app.py
```

---

## 2. Module Descriptions

### 2.1 `src/utils.py`
**Purpose:** Centralized utilities, logging, and constants

**Key Components:**
- `setup_logger()` - Configure structured logging
- `set_seed()` - Reproducibility setup
- `get_risk_level()` - Probability → Risk level conversion
- Constants: RANDOM_SEED, CV_FOLDS, SMOTE params, GridSearch params

**Imports Required:**
```python
import logging, random, numpy, pathlib
```

### 2.2 `src/preprocessing.py`
**Purpose:** Data loading, EDA, and text cleaning

**Key Functions:**
- `load_dataset()` - Load CSV with validation
- `explore_dataset()` - Display statistics
- `visualize_eda()` - Generate plots
- `clean_email()` - Text preprocessing pipeline
- `get_statistics()` - Extract email metadata

**Text Cleaning Pipeline:**
```
Text → Lowercase → Remove HTML → Remove URLs → 
Remove Punctuation → Remove Digits → Tokenize → 
Remove Stopwords → Lemmatize → Return String
```

**Dependencies:** pandas, numpy, nltk, matplotlib, seaborn

### 2.3 `src/feature_engineering.py`
**Purpose:** Convert text to numerical features

**Key Functions:**
- `create_tfidf_features()` - TF-IDF vectorization
- `create_statistical_features()` - Email metadata features
- `combine_features()` - Merge TF-IDF + statistical
- `create_all_features()` - End-to-end feature creation
- `save_vectorizer()`, `load_vectorizer()` - Persistence

**Feature Matrix Composition:**
```
5,000 TF-IDF features +
5 Statistical features =
5,005 total features

Statistical Features:
1. email_length: Character count
2. word_count: Token count
3. uppercase_ratio: Uppercase chars / total chars
4. digit_ratio: Digit chars / total chars
5. url_count: Count of URLs
```

**Dependencies:** sklearn, scipy, joblib

### 2.4 `src/advanced_ml.py` ⭐ NEW
**Purpose:** SMOTE, cross-validation, hyperparameter tuning

**Key Functions:**
- `apply_smote()` - Balance training data
- `get_stratified_kfold()` - Stratified CV splitter
- `cross_validate_model()` - Evaluate with K-fold
- `tune_logistic_regression()` - GridSearchCV for LR
- `tune_naive_bayes()` - GridSearchCV for NB
- `save_tuning_results()`, `load_tuning_results()` - Persistence

**SMOTE Parameters:**
```python
SMOTE(
    k_neighbors=5,
    random_state=42,
    sampling_strategy='minority'
)
```

**GridSearch Configurations:**
```python
# Logistic Regression
LR_PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight': ['balanced', None],
    'solver': ['lbfgs'],
    'max_iter': [1000, 2000]
}

# Naive Bayes
NB_PARAM_GRID = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0]
}
```

**Dependencies:** imbalanced-learn, sklearn

### 2.5 `src/visualization.py` ⭐ NEW
**Purpose:** Comprehensive visualizations for analysis

**Key Functions:**
- `plot_feature_importance()` - Top 25 features chart
- `plot_top_phishing_keywords()` - Phishing indicators
- `plot_top_ham_keywords()` - Legitimate indicators
- `plot_model_comparison()` - Bar chart comparison
- `plot_cross_validation_scores()` - CV distributions
- `plot_roc_curves()` - ROC curve comparison
- `plot_learning_curves()` - Training analysis
- `create_metrics_table()` - Summary table

**Output Formats:**
- PNG images (300 DPI) to `results/`
- CSV tables to `models/`
- Console logging summaries

**Dependencies:** matplotlib, seaborn, sklearn, pandas

### 2.6 `src/train_baseline.py`
**Purpose:** Multinomial Naive Bayes training

**Key Functions:**
- `train_baseline_model()` - NB training
- `evaluate_model()` - Calculate metrics

**Configuration:**
```python
MultinomialNB()  # No hyperparameters
```

**Output:**
- Model saved to `models/naive_bayes_model.pkl`
- Metrics logged to console

### 2.7 `src/train_logistic.py`
**Purpose:** Logistic Regression training

**Key Functions:**
- `train_logistic_model()` - LR training
- `evaluate_and_compare()` - Model comparison

**Configuration:**
```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
```

**Output:**
- Model saved to `models/logistic_regression_model.pkl`
- Feature names saved to `models/feature_names.pkl`

### 2.8 `src/evaluate.py`
**Purpose:** Metrics computation and error analysis

**Key Functions:**
- `compute_metrics()` - Calculate all metrics
- `print_metrics()` - Formatted output
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_precision_recall_curve()` - PR curve
- `analyze_errors()` - Identify FP/FN
- `run_evaluation()` - End-to-end evaluation

**Metrics Calculated:**
- Accuracy, Precision, Recall, F1, ROC-AUC
- Specificity, TP, TN, FP, FN

**Output:**
- CSV files with errors to `results/`
- PNG plots to `results/`

### 2.9 `src/risk_scoring.py`
**Purpose:** Prediction and risk assessment

**Key Functions:**
- `load_model()` - Load trained model
- `get_risk_level_info()` - Probability → Risk mapping
- `predict_email()` - Single email analysis
- `batch_predict()` - Multiple emails

**Output Dictionary:**
```python
{
    "prediction": "Ham" | "Phishing",
    "risk_score": int (0-100),
    "risk_level": "Low" | "Medium" | "High" | "Critical",
    "probability": float (0-1),
    "timestamp": str,
    "email_length": int,
    "word_count": int
}
```

### 2.10 `src/explainability.py`
**Purpose:** Feature importance and per-email explanations

**Key Functions:**
- `get_top_features()` - Extract top coefficients
- `display_feature_importance()` - Print global importance
- `extract_suspicious_keywords()` - Find known phishing words
- `get_email_explanation()` - Per-email analysis
- `display_email_explanation()` - Formatted output

**Explanation Output:**
```python
{
    "prediction": str,
    "probability": float,
    "top_tfidf_features": [(feature, contribution, coef), ...],
    "top_stat_features": [(feature, contribution, coef), ...],
    "suspicious_keywords": [word, ...],
    "positive_evidence": [indicator, ...],
    "negative_evidence": [indicator, ...],
    "email_length": int,
    "word_count": int,
    "url_count": int
}
```

---

## 3. Data Pipelines

### 3.1 Training Pipeline (`train.py`)

```
Phase 1: Load & Explore
  ├─ Load CSV
  ├─ Display statistics
  └─ Generate EDA plots

Phase 2: Preprocess
  ├─ Clean text for each email
  └─ Verify output

Phase 3: Feature Engineering
  ├─ Fit TF-IDF vectorizer
  ├─ Extract statistical features
  └─ Combine (5,005 features)

Phase 4: Stratified Split
  ├─ Train: 80% (65,987)
  ├─ Test: 20% (16,497)
  └─ Stratify by class

Phase 5: Apply SMOTE
  ├─ Balance training data
  ├─ Original: 65,987 → ~131,974 SMOTE
  └─ Verify distribution

Phase 6: Cross-Validation (5-fold)
  ├─ CV Naive Bayes
  ├─ CV Logistic Regression
  └─ Report mean ± std

Phase 7: Hyperparameter Tuning
  ├─ GridSearchCV (Naive Bayes)
  ├─ GridSearchCV (Logistic Regression)
  └─ Save tuning results

Phase 8: Final Model Training
  ├─ Train on SMOTE data
  └─ Save models

Phase 9: Test Evaluation
  ├─ Evaluate both models
  ├─ Generate confusion matrices
  └─ Produce error analysis

Phase 10: Comparison
  ├─ Side-by-side metrics
  ├─ Highlight improvements
  └─ Recommend best model

Phase 11: Explainability
  ├─ Extract feature importance
  ├─ Identify top phishing/ham keywords
  └─ Generate example explanations

Phase 12: Visualizations
  ├─ Feature importance charts
  ├─ Keyword distributions
  ├─ Model comparison
  ├─ CV score distributions
  ├─ ROC curves
  └─ Learning curves
```

### 3.2 Prediction Pipeline (`risk_scoring.py`)

```
Raw Email Text
    ↓
Clean Text
    ↓
TF-IDF Transform
    ↓
Extract Statistics
    ↓
Combine Features (5,005 dims)
    ↓
Load Model
    ↓
Get Prediction (0/1)
    ↓
Get Probability (0-1)
    ↓
Map to Risk Level
    ↓
Return Result Dictionary
```

---

## 4. Model Training Process

### 4.1 Training Workflow

```mermaid
[Original Data]
    │
    ├──→ [Stratified Split]
    │        │
    │        ├──→ [Train 80%]
    │        │      │
    │        │      ├──→ [SMOTE Balance]
    │        │      │      │
    │        │      │      ├──→ [5-Fold CV]
    │        │      │      │      └──→ [Metrics Report]
    │        │      │      │
    │        │      │      ├──→ [GridSearchCV]
    │        │      │      │      └──→ [Best Hyperparams]
    │        │      │      │
    │        │      │      └──→ [Final Training]
    │        │      │             └──→ [Model Saved]
    │        │
    │        └──→ [Test 20%]
    │               │
    │               └──→ [Evaluation]
    │                      ├──→ [Metrics]
    │                      ├──→ [Confusion Matrix]
    │                      ├──→ [Error Analysis]
    │                      └──→ [Visualizations]
    │
    └──→ [Inference]
           │
           ├──→ [Preprocess Email]
           ├──→ [Extract Features]
           ├──→ [Predict]
           └──→ [Return Result]
```

### 4.2 Cross-Validation Strategy

```python
# 5-Fold Stratified K-Fold
Fold 1: Train [80%] → Test [20%]
Fold 2: Train [80%] → Test [20%]
Fold 3: Train [80%] → Test [20%]
Fold 4: Train [80%] → Test [20%]
Fold 5: Train [80%] → Test [20%]
        ↓
    Average scores ± std
    Report: Mean ± Std across folds
```

---

## 5. API Reference

### 5.1 Main Entry Point

```python
# train.py
python train.py
```

Executes complete training pipeline with all 12 phases.

### 5.2 Single Email Prediction

```python
from src.risk_scoring import predict_email

result = predict_email("email text here")
# Returns: dict with prediction, risk_score, risk_level, probability
```

### 5.3 Batch Prediction

```python
from src.risk_scoring import batch_predict

results = batch_predict([email1, email2, email3])
# Returns: list of result dicts
```

### 5.4 Model Explanation

```python
from src.explainability import get_email_explanation

explanation = get_email_explanation(
    email_text,
    model,
    vectorizer,
    feature_names
)
# Returns: dict with detailed explanation
```

---

## 6. Configuration Guide

### 6.1 Modifying Constants (`src/utils.py`)

```python
# Dataset split
TEST_SIZE = 0.2  # 20% test, 80% train

# Feature engineering
TFIDF_MAX_FEATURES = 5000  # Number of TF-IDF features
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# SMOTE
SMOTE_K_NEIGHBORS = 5  # For synthetic sample generation
SMOTE_SAMPLING_STRATEGY = 'minority'  # Balance minority class

# Cross-validation
CV_FOLDS = 5  # Number of CV folds

# Hyperparameter tuning
LR_PARAM_GRID = {...}  # Logistic Regression parameters
NB_PARAM_GRID = {...}  # Naive Bayes parameters

# Risk scoring thresholds
RISK_LEVELS = {
    "Low": (0, 30),
    "Medium": (31, 60),
    "High": (61, 80),
    "Critical": (81, 100),
}
```

### 6.2 Customizing Hyperparameter Grid

```python
# In src/utils.py, modify LR_PARAM_GRID
LR_PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'class_weight': ['balanced'],  # Handle imbalance
    'solver': ['lbfgs'],  # Optimization algorithm
    'max_iter': [1500],  # Convergence iterations
}
```

---

## 7. Troubleshooting

### Issue: NLTK data not found

**Error:** `LookupError: Resource 'punkt_tab' not found`

**Solution:**
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Memory error with large dataset

**Symptoms:** Process killed, out-of-memory errors

**Solutions:**
1. Reduce TFIDF_MAX_FEATURES to 2500-3000
2. Process in batches
3. Increase system RAM

### Issue: Slow training

**Expected:** 3-5 minutes for full pipeline  
**If longer:**
- Reduce CV_FOLDS to 3
- Reduce GridSearch param grid size
- Use GPU if available (not configured by default)

### Issue: Model not improving with SMOTE

**Check:**
1. Verify SMOTE is applied: `X_train_smote.shape > X_train.shape`
2. Check class distribution after SMOTE: `np.bincount(y_train_smote)`
3. Ensure stratification: `y_train_smote` should be ~50-50

---

## 8. Performance Optimization

### 8.1 Computational Complexity

| Operation | Complexity | Time (82K samples) |
|-----------|-----------|-------------------|
| Load Data | O(n) | 1-2 sec |
| Preprocess | O(n×m) | 60-120 sec |
| TF-IDF | O(n×m) | 30-40 sec |
| SMOTE | O(n log n) | 5-10 sec |
| Train NB | O(n×d) | <1 sec |
| Train LR | O(n×d×k) | 5-10 sec |
| GridSearchCV | O(n×d×k×g×cv) | 30-60 sec |
| Cross-Validation | O(n×d×k×cv) | 20-40 sec |
| Evaluation | O(n×d) | 5-10 sec |
| Visualization | O(n×d) | 10-20 sec |

**Total:** 3-5 minutes

### 8.2 Memory Requirements

- Raw data: ~100 MB
- Processed data (TF-IDF sparse): ~200 MB
- SMOTE-expanded: ~400 MB
- Models + Vectorizer: ~250 MB
- **Total:** ~1 GB (typical run)

---

**Document Version:** 1.0  
**Last Updated:** June 2026
