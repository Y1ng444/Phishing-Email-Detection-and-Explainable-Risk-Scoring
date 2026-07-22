Data Directory

Raw and processed email datasets are stored locally in this directory.

Datasets are not committed to Git because they may be large and may have separate usage conditions.

Directory Structure
data/
├── raw/
├── processed/
└── README.md
Raw Data

Place the course CSV files in:

data/raw/

Expected file names:

CEAS_08.csv
Enron.csv
Nazario.csv

The standardization script supports different column names for fields such as:

sender
receiver
date
subject
body
urls
text_combined
label

It also supports several common CSV encodings.

Processed Data

Generate the standardized dataset with:

python standardize_datasets.py

Output:

data/processed/phishing_email_standardized.csv

Canonical columns:

source_file
sender
receiver
date
subject
body
urls
text_combined
label

The training pipeline requires:

text_combined
label

The source_file field is used for source-holdout evaluation.

Other fields are used for metadata feature extraction when available.

Important Notes
Do not commit private or sensitive email data.
Review the usage conditions of each dataset.
Do not place API keys, passwords, or .env files in this directory.
Re-run standardization whenever the raw datasets or label mapping change.