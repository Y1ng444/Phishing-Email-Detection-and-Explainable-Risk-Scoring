Models Directory

Trained machine-learning pipelines are generated locally and stored in this directory.

Model files are not committed to Git because they are generated artifacts and may be large.

Generate the models with:

python train_optimized.py
Generated Models
Final application model
models/phishing_logreg_text_metadata.pkl

Pipeline:

TF-IDF text features
        +
Standardized metadata features
        ↓
Logistic Regression

This model is loaded by app.py.

Random Forest comparison model
models/phishing_random_forest_text_metadata.pkl

Pipeline:

TF-IDF
   ↓
TruncatedSVD
        +
Standardized metadata
        ↓
Random Forest
Text-only Logistic Regression model
models/phishing_logreg_tfidf.pkl

Pipeline:

TF-IDF
   ↓
Logistic Regression

This model is retained for comparison and text-only explainability.

Compatibility Note

The models are serialized with joblib.

A saved model may fail to load with incompatible versions of scikit-learn or related dependencies.

When the environment changes, retrain the models with:

python train_optimized.py