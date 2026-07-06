# Results

Evaluation outputs are generated locally.

Run:

```bash
python eda_standardized.py
python train_optimized.py
```

Typical generated files include:

- `metrics_optimized.csv`
- `metrics_cleaned_dedup.csv`
- `near_duplicate_check.csv`
- `source_holdout_metrics.csv`
- `cleaned_duplicate_check.csv`
- `missing_values.csv`
- `dataset_source_distribution.csv`
- `duplicate_statistics.csv`
- `top_terms_by_class.csv`
- `class_distribution.png`
- `text_length_distribution.png`
- `naive_bayes_confusion_matrix.png`
- `logistic_regression_confusion_matrix.png`
- `linear_svm_confusion_matrix.png`
- `roc_curve_comparison.png`
- `pr_curve_comparison.png`
- `error_samples.csv`
- `sample_predictions.csv`
- `top_phishing_indicators.csv`
- `top_legitimate_indicators.csv`

Generated result files are not committed to GitHub because they are generated
artifacts.
