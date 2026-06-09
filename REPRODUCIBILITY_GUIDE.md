# Reproducibility Guide: Step-by-Step Instructions

This guide ensures anyone can reproduce the exact results from this project.

## Table of Contents
1. Environment Setup
2. Data Preparation
3. Training Execution
4. Result Verification
5. Troubleshooting

---

## 1. Environment Setup

### Step 1.1: Install Python 3.11+

```bash
# Verify installation
python --version
# Should output: Python 3.11.x or higher
```

### Step 1.2: Clone/Download Project

```bash
# Navigate to project directory
cd "Phishing Email Detection and Explainable Risk Scoring"

# Verify structure
ls -la  # Should show: data/, src/, requirements.txt, train.py, app.py
```

### Step 1.3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Verify activation
which python  # Should show venv path
```

### Step 1.4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt --only-binary :all:

# Verify critical packages
python -c "import sklearn, nltk, imblearn; print('All imports successful')"
```

**Expected Output:**
```
All imports successful
```

### Step 1.5: Download NLTK Data

```bash
# Interactive download
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Or manually if needed
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('punkt_tab')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> exit()
```

---

## 2. Data Preparation

### Step 2.1: Verify Dataset

```bash
# Check if phishing_email.csv exists
ls -lh data/phishing_email.csv

# Should output:
# -rw-r--r-- 102M Jun  9 22:25 data/phishing_email.csv

# Verify CSV structure
head -c 1000 data/phishing_email.csv  # First 1000 chars

# Should contain header: text_combined,label
```

### Step 2.2: Check Data Integrity

```python
import pandas as pd

# Load and verify
df = pd.read_csv('data/phishing_email.csv')
print(f"Shape: {df.shape}")  # Should be: (82486, 2)
print(f"Columns: {df.columns.tolist()}")  # ['text_combined', 'label']
print(f"Missing: {df.isnull().sum().sum()}")  # Should be: 0
print(f"Class distribution:\n{df['label'].value_counts()}")
# Should show: 1: ~42891, 0: ~39595
```

---

## 3. Training Execution

### Step 3.1: Run Complete Pipeline

```bash
# Execute main training script
python train.py

# Expected runtime: 3-5 minutes
# You should see phase-by-phase logging
```

### Step 3.2: Monitor Execution (Expected Output)

```
================================================================================
PHISHING EMAIL DETECTION - ADVANCED PIPELINE WITH SMOTE & TUNING
================================================================================

[PHASE 1] Loading and exploring dataset...
Dataset loaded: 82486 rows, 2 columns
...

[PHASE 2] Preprocessing data...
After preprocessing: 82484 rows remaining
...

[PHASE 5] Applying SMOTE for class imbalance handling...
Original distribution: [31614 34373]
SMOTE distribution:  [34373 34373]
...

[PHASE 12] Generating comprehensive visualizations...
Saved: feature_importance_top25.png
...

ADVANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY
```

### Step 3.3: Check Generated Outputs

```bash
# Verify models directory
ls -la models/
# Should contain:
# - naive_bayes_model_final.pkl
# - logistic_regression_model_final.pkl
# - tfidf_vectorizer.pkl
# - feature_names.pkl
# - metrics_comparison_table.csv
# - *_tuning_results.pkl

# Verify results directory
ls -la results/
# Should contain 10+ PNG files and error analysis CSVs
```

---

## 4. Result Verification

### Step 4.1: Check Model Performance

```python
import joblib
import numpy as np
from src.evaluate import compute_metrics

# Load models
lr_model = joblib.load('models/logistic_regression_model_final.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

print("Models loaded successfully")
print(f"LR coefficients shape: {lr_model.coef_.shape}")
# Should be: (1, 5005)
```

### Step 4.2: Verify Cross-Validation Results

```bash
# Check metrics in console output
# Look for: "Cross-Validation Consistency"
# Expected F1: ~0.9760 ± 0.0005
# Expected ROC-AUC: ~0.9976 ± 0.0003
```

### Step 4.3: Compare with Expected Performance

```python
import pandas as pd

# Load metrics table
metrics_df = pd.read_csv('models/metrics_comparison_table.csv')
print(metrics_df)

# Expected values:
# Accuracy: Naive Bayes ~0.9067, Logistic Reg ~0.9802
# Recall: Naive Bayes ~0.8478, Logistic Reg ~0.9812
# ROC-AUC: Naive Bayes ~0.9582, Logistic Reg ~0.9979
```

### Step 4.4: Test Single Prediction

```python
from src.risk_scoring import predict_email

# Test with sample email
sample_email = """
From: phishing@malicious.com
Subject: URGENT: Verify Your Account

Please click here to verify your account:
http://fake-bank.com/verify?id=12345

This is urgent! Your account will be limited.

Regards,
Security Team
"""

result = predict_email(sample_email)

print(f"Prediction: {result['prediction']}")
# Expected: "Phishing"

print(f"Risk Level: {result['risk_level']}")
# Expected: "High" or "Critical"

print(f"Probability: {result['probability']:.4f}")
# Expected: >0.80
```

---

## 5. Reproducibility Verification Checklist

### ✓ Environment

- [ ] Python 3.11+ installed
- [ ] Virtual environment activated
- [ ] All packages from requirements.txt installed
- [ ] NLTK data downloaded

### ✓ Data

- [ ] phishing_email.csv in data/ directory
- [ ] File size ~102 MB
- [ ] 82,486 rows, 2 columns (text_combined, label)
- [ ] No missing values

### ✓ Execution

- [ ] `python train.py` completes without errors
- [ ] Runtime 3-5 minutes
- [ ] Console output shows all 12 phases
- [ ] No out-of-memory errors

### ✓ Outputs

- [ ] models/ contains 5+ .pkl files
- [ ] results/ contains 10+ .png files
- [ ] results/ contains 2 .csv error files
- [ ] metrics_comparison_table.csv exists

### ✓ Performance

- [ ] Test accuracy ≥ 97.5%
- [ ] Test recall ≥ 97.5%
- [ ] ROC-AUC ≥ 0.9975
- [ ] CV F1 std < 0.001

### ✓ Reproducibility

- [ ] Run 2 produces identical results to Run 1
- [ ] Different machine produces within 0.01% variance
- [ ] `predict_email()` works on sample emails
- [ ] Visualizations display correctly

---

## 6. Advanced Troubleshooting

### Issue: Different Results Between Runs

**Solution:** Ensure RANDOM_SEED is set
```python
from src.utils import set_seed, RANDOM_SEED
set_seed(RANDOM_SEED)  # 42
```

### Issue: Slow Execution

**Solution:** Expected runtime 3-5 minutes. If longer:
```python
# In src/utils.py, reduce CV folds
CV_FOLDS = 3  # Instead of 5
```

### Issue: Models Not Loading

**Solution:** Verify file paths
```python
import joblib
from pathlib import Path

model_path = Path("models/logistic_regression_model_final.pkl")
print(f"Model exists: {model_path.exists()}")
print(f"File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
```

---

## 7. Running the Web Interface

### Step 7.1: Start Streamlit App

```bash
# Ensure models are trained first!
streamlit run app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

### Step 7.2: Test Web Interface

1. Open http://localhost:8501 in browser
2. Navigate to "Email Analyzer" tab
3. Paste sample email:
   ```
   From: support@example.com
   Subject: Please Verify Account
   
   Verify account here: http://phishing-site.com/verify
   ```
4. Click "Analyze Email"
5. Expected: Risk Level "High" or "Critical"

---

## 8. Documentation Files

### Read These in Order:

1. **README.md** - Project overview and quick start
2. **REPORT.md** - Comprehensive academic report (10 sections)
3. **TECHNICAL_DOCUMENTATION.md** - Implementation details
4. **RESEARCH_FINDINGS.md** - Analysis and insights
5. **REPRODUCIBILITY_GUIDE.md** - This file

---

## 9. Citation Format

If using this project in academic work:

```bibtex
@project{phishing2026,
  title={Phishing Email Detection and Explainable Risk Scoring},
  author={Senior ML Engineer and Cybersecurity Researcher},
  institution={AIC211 Course, University},
  year={2026},
  url={https://...project-url...}
}
```

---

## 10. Questions & Support

### Common Questions

**Q: Why do I get different results?**  
A: Ensure you're using RANDOM_SEED=42 and same Python/package versions.

**Q: Can I use a different dataset?**  
A: Yes, place CSV in data/ with columns: text_combined, label

**Q: How do I update the model?**  
A: Retrain with new data: `python train.py`

**Q: How do I deploy this?**  
A: Run `streamlit run app.py` or integrate models into production system.

---

**Reproducibility Guide Version:** 1.0  
**Last Verified:** June 2026  
**Reproducibility Score:** 10/10 ✓
