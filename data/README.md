# Data

The dataset is not included in this GitHub repository because raw and processed
email datasets can be large. Keep the repository lightweight by storing data
locally.

Place raw CSV files in:

```text
data/raw/
```

Supported schemas:

- `subject,body,label`
- `subject,body,label,urls`
- `sender,receiver,date,subject,body,label,urls`
- `sender,receiver,date,subject,body,urls,label`
- `text_combined,label`

Run standardization with:

```bash
python standardize_datasets.py
```

The standardized dataset will be generated locally at:

```text
data/processed/phishing_email_standardized.csv
```

The final required columns are:

```text
source_file,text_combined,label
```

`source_file` is used for source-holdout evaluation, for example training on
CEAS + Enron and testing on Nazario.
