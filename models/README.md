# Models

Trained model files are generated locally.

Run:

```bash
python train_optimized.py
```

The final Logistic Regression text+metadata model will be saved as:

```text
models/phishing_logreg_text_metadata.pkl
```

Additional model artifacts:

```text
models/phishing_random_forest_text_metadata.pkl
models/phishing_logreg_tfidf.pkl
```

Model files are not committed to GitHub because they are generated artifacts.
