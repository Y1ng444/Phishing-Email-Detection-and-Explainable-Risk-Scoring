How to Run

This repository excludes raw datasets, processed datasets, trained models, and generated evaluation outputs from Git.

Follow the steps below to reproduce the project locally.

1. Requirements

Recommended environment:

Python 3.11 or newer

Verify Python:

python --version

On Windows, the command may also be:

py --version
2. Clone the Repository
git clone https://github.com/Y1ng444/Phishing-Email-Detection-and-Explainable-Risk-Scoring.git
cd Phishing-Email-Detection-and-Explainable-Risk-Scoring
3. Create a Virtual Environment
Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1

If PowerShell blocks script execution, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
Windows Command Prompt
py -m venv .venv
.venv\Scripts\activate.bat
macOS or Linux
python3 -m venv .venv
source .venv/bin/activate
4. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
5. Add Raw Dataset Files

Place the course CSV files in:

data/raw/

Expected file names:

data/raw/CEAS_08.csv
data/raw/Enron.csv
data/raw/Nazario.csv

Raw data is not included in the repository.

The standardization script supports datasets containing combinations of fields such as:

sender
receiver
date
subject
body
urls
text_combined
label

Different column aliases are mapped to the canonical project schema automatically.

6. Standardize the Datasets

Run:

python standardize_datasets.py

Output:

data/processed/phishing_email_standardized.csv

The standardized file contains:

source_file
sender
receiver
date
subject
body
urls
text_combined
label
7. Run Exploratory Data Analysis

Run:

python eda_standardized.py

Typical outputs include:

results/class_distribution.png
results/text_length_distribution.png
results/missing_values.csv
results/dataset_source_distribution.csv
results/duplicate_statistics.csv
results/top_terms_by_class.csv
8. Train and Evaluate the Models

Run:

python train_optimized.py

This command:

Loads the standardized dataset.
Creates a stratified train/test split.
Trains text-only models.
Trains text-and-metadata models.
Generates evaluation metrics.
Saves confusion matrices.
Saves ROC and Precision–Recall curves.
Checks cleaned-text duplicates.
Checks near-duplicate similarity.
Runs cleaned-deduplicated benchmarks.
Runs source-holdout experiments.
Saves model-explanation files.
Saves the trained pipelines.

Generated model files:

models/phishing_logreg_text_metadata.pkl
models/phishing_random_forest_text_metadata.pkl
models/phishing_logreg_tfidf.pkl

The training command can take longer on the full dataset because it includes multiple models and robustness checks.

9. Launch the Streamlit Application

Run:

streamlit run app.py

Open the local URL printed in the terminal.

The application normally starts at:

http://localhost:8501

Paste an email into the text area and select the analysis button.

10. Recommended Execution Order
python standardize_datasets.py
python eda_standardized.py
python train_optimized.py
streamlit run app.py
11. Troubleshooting
No raw CSV files found

Confirm that the files are inside:

data/raw/
Standardized dataset not found

Run:

python standardize_datasets.py
Final model not found

Run:

python train_optimized.py

Confirm that this file exists:

models/phishing_logreg_text_metadata.pkl
Import error

Activate the virtual environment and reinstall dependencies:

pip install -r requirements.txt
Streamlit command not found

Run:

python -m streamlit run app.py
PowerShell activation error

Run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then activate the environment again.

Model loading error after changing library versions

Delete the generated model files and retrain:

python train_optimized.py

Joblib model files may require a compatible scikit-learn version.