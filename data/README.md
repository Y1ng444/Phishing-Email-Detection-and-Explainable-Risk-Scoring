# Data Folder

Place raw phishing email CSV files in `data/raw/`.

The standardization script reads CSV files from `data/raw/` and writes:

```text
data/processed/phishing_email_standardized.csv
```

Supported input schemas include:

- `subject, body, label`
- `subject, body, label, urls`
- `sender, receiver, date, subject, body, label, urls`
- `sender, receiver, date, subject, body, urls, label`
- `text_combined, label`

Labels are converted to binary values:

- `1`: phishing, spam, malicious, bad, true, yes
- `0`: legitimate, legit, ham, benign, safe, false, no
