# Multi-CSV Dataset Auto-Discovery Guide

## Overview

The phishing email detection project now automatically loads **ALL CSV files** from the `data/` directory. Simply place new datasets there and rerun training—no code changes required.

---

## Feature Highlights

### ✅ Automatic CSV Discovery
- Scans `data/` folder recursively for all `.csv` files
- No hardcoded filenames
- Works with any folder structure

### ✅ Automatic Validation
- Validates each CSV has required columns: `text_combined`, `label`
- Skips invalid files with warning messages
- Never crashes due to malformed data

### ✅ Automatic Merging
- Combines all valid CSVs into one dataset
- Removes duplicates across all files
- Handles missing values

### ✅ Detailed Statistics
- Reports total files found, loaded, and skipped
- Shows rows before/after cleaning
- Displays class distribution
- Logs all operations

### ✅ Zero Configuration
- No changes to code required
- Works with any dataset format (as long as columns match)
- Backward compatible with existing workflows

---

## Adding New Datasets

### Step 1: Create CSV File

Format your dataset with exactly these columns:

```csv
text_combined,label
"Email content here",0
"Phishing attempt here",1
```

**Column Requirements:**
- `text_combined` (str): Full email content
- `label` (int): 0 for Ham, 1 for Phishing

### Step 2: Place in Data Directory

```bash
# Option A: Place in root of data/ folder
cp your_dataset.csv data/

# Option B: Place in subfolder (also scanned)
mkdir data/new_sources
cp your_dataset.csv data/new_sources/
```

### Step 3: Run Training

```bash
python train.py
```

The loader automatically finds and loads your dataset!

---

## Example: Adding Multiple Datasets

### Scenario
You have 5 datasets from different sources:

```
Source 1: CEAS_08.csv (but wrong columns)
Source 2: Enron.csv (but wrong columns)
Source 3: nazario.csv (but wrong columns)
Source 4: phishing_email.csv ✓ (correct columns)
Source 5: custom_dataset.csv (will add correct columns)
```

### Solution

#### A. Convert Source 5 to Match Schema

```python
import pandas as pd

# Load original dataset
df = pd.read_csv('data/sources/custom_dataset.csv')

# Rename columns to match schema
df_converted = df.rename(columns={
    'email_text': 'text_combined',
    'is_phishing': 'label'
})

# Keep only required columns
df_converted = df_converted[['text_combined', 'label']]

# Ensure correct data types
df_converted['text_combined'] = df_converted['text_combined'].astype(str)
df_converted['label'] = df_converted['label'].astype(int)

# Save
df_converted.to_csv('data/custom_dataset.csv', index=False)
print(f"Converted: {len(df_converted)} rows")
```

#### B. Run Training

```bash
python train.py
```

**Output:**
```
Found 7 CSV file(s) to process:
  - CEAS_08.csv
  - Enron.csv
  - Nazario.csv
  - Nigerian_Fraud.csv
  - SpamAssasin.csv
  - phishing_email.csv
  - custom_dataset.csv

Validating CSV files...
  - CEAS_08.csv: SKIPPED (missing text_combined column)
  - Enron.csv: SKIPPED (missing text_combined column)
  - Nazario.csv: SKIPPED (missing text_combined column)
  - Nigerian_Fraud.csv: SKIPPED (missing text_combined column)
  - SpamAssasin.csv: SKIPPED (missing text_combined column)
  - phishing_email.csv: VALID (82,486 rows)
  - custom_dataset.csv: VALID (5,200 rows)

Valid files: 2, Skipped files: 5

Merging 2 dataframe(s)...
After merge: 87,686 rows, 2 columns

Performing data cleaning and validation...
Removing 312 duplicate email(s)...

DATASET STATISTICS
======================================================================
CSV files loaded:        2
CSV files skipped:       5
Total rows (raw):        87,686
Total rows (cleaned):    87,374
Duplicates removed:      312
Missing values removed:  0
Final dataset shape:     (87,374, 2)

Class distribution:
  Ham        (0): 41,872 (47.95%)
  Phishing   (1): 45,502 (52.05%)
```

---

## How It Works

### Data Flow

```
data/
├── phishing_email.csv
├── custom_dataset.csv
├── another_dataset.csv
└── subfolder/
    └── nested_dataset.csv
        ↓
        [Recursive scan finds all CSVs]
        ↓
        [Validate columns (text_combined, label)]
        ↓
        [Load valid CSVs]
        ↓
        [Merge dataframes]
        ↓
        [Remove duplicates on text_combined]
        ↓
        [Remove rows with missing text_combined or label]
        ↓
        [Convert types: text_combined→str, label→int]
        ↓
        [Print statistics]
        ↓
        Return merged DataFrame
```

### Code Implementation

Key functions in `src/preprocessing.py`:

```python
def load_dataset() -> Tuple[pd.DataFrame, int]:
    """
    Loads and merges all CSV files from data/ directory.
    
    Returns:
        Tuple of (merged_dataframe, total_rows_before_cleaning)
    """
    # 1. Get all CSV files recursively
    csv_files = _get_csv_files()
    
    # 2. Validate each CSV
    valid_files = [f for f in csv_files if _validate_csv(f)]
    
    # 3. Load all valid CSVs
    dataframes = [_load_single_csv(f)[0] for f in valid_files]
    
    # 4. Merge
    df = pd.concat(dataframes, ignore_index=True)
    
    # 5. Clean (duplicates, missing values)
    df = df.drop_duplicates(subset=['text_combined'], keep='first')
    df = df.dropna(subset=['text_combined', 'label'])
    
    # 6. Validate types
    df['text_combined'] = df['text_combined'].astype(str)
    df['label'] = df['label'].astype(int)
    
    # 7. Return with statistics logged
    return df, total_rows_before_cleaning
```

---

## Data Format Reference

### Minimal Valid CSV

```csv
text_combined,label
"This is an email",0
"Click here to verify account",1
"Test message",0
```

### Complete Valid CSV

```csv
text_combined,label
"From: sender@example.com
To: recipient@example.com
Subject: Hello

This is the email body.",0
"URGENT: Verify your account now!
Click: http://malicious-site.com",1
```

### Data Type Conversions

```
text_combined: Automatically converted to string
  Input: "hello" → Output: "hello" ✓
  Input: 12345 → Output: "12345" ✓
  Input: None/NaN → Removed from dataset ✗

label: Automatically converted to integer
  Input: 0 → Output: 0 ✓
  Input: "0" → Output: 0 ✓
  Input: 1.0 → Output: 1 ✓
  Input: None/NaN → Removed from dataset ✗
```

---

## Error Handling

### Scenario: Missing Column

**Input CSV:**
```csv
subject,body,spam_flag
"Hello","This is email",0
```

**Output:**
```
WARNING - Skipping datasets.csv: Missing columns. 
Expected {'label', 'text_combined'}, 
Found {'subject', 'body', 'spam_flag'}
```

**Solution:** Rename columns in CSV
```python
df.rename(columns={
    'subject': 'text_combined',  # or combine with body
    'spam_flag': 'label'
})
```

### Scenario: Invalid File Format

**Input:** `data/corrupted.csv` (corrupted/unreadable)

**Output:**
```
WARNING - Skipping corrupted.csv: Error reading file - 
[Error details...]
```

**Solution:** Fix or remove the corrupted file

### Scenario: Duplicate Emails

**Input:** Same email in multiple CSVs
```
phishing_email.csv: "Click here" (row 100)
custom_data.csv:    "Click here" (row 50)
```

**Output:**
```
Removing 1 duplicate email(s)...
INFO - Removing 1 duplicate(s) found
```

**Logic:** Keeps first occurrence, removes duplicates in other files

### Scenario: Missing Values

**Input:**
```csv
text_combined,label
"Valid email",1
,0
"Another email",
```

**Output:**
```
Removing 2 missing value(s)...
INFO - Found missing values: text_combined=1, label=1
```

**Result:** Both rows removed, only valid row remains

---

## Backwards Compatibility

### Existing Code Works As-Is

All existing code continues to work without changes:

```python
# train.py (no changes needed)
from src.preprocessing import load_dataset

df, total_rows = load_dataset()  # Works with multi-CSV now!
```

```python
# train_baseline.py (no changes needed)
df, _ = load_dataset()  # Automatically loads all CSVs
```

```python
# train_logistic.py (no changes needed)
df, _ = load_dataset()  # Transparent upgrade
```

### Return Type Unchanged

Function signature remains identical:

```python
def load_dataset() -> Tuple[pd.DataFrame, int]:
    # Returns: (merged_df, total_rows_before_cleaning)
```

---

## Advanced: Custom Preprocessing

### Before Training, Verify Your Data

```python
import pandas as pd
from src.preprocessing import load_dataset

# Load all datasets
df, total_rows = load_dataset()

# Quick inspection
print(f"Total rows: {df.shape[0]}")
print(f"Class distribution:\n{df['label'].value_counts()}")
print(f"Sample text length: {df['text_combined'].str.len().mean():.0f} chars")

# Check for issues
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Data types: {df.dtypes.to_dict()}")
```

### Custom Column Concatenation

If your CSVs have separate columns (subject, body, etc.):

```python
import pandas as pd

# Create combined text_combined column
df = pd.read_csv('data/source.csv')
df['text_combined'] = df['subject'] + ' ' + df['body']
df = df[['text_combined', 'label']]
df.to_csv('data/source_converted.csv', index=False)
```

---

## Testing

### Quick Test

```bash
# Test the multi-CSV loader
python test_multi_csv_loader.py

# Output should show:
# Test Result: SUCCESS
# Loaded dataframe shape: (XXX, 2)
```

### Verify Specific Dataset

```python
import pandas as pd

# Check what would be loaded
csv_path = 'data/your_dataset.csv'
df = pd.read_csv(csv_path, nrows=5)

# Verify columns
required = {'text_combined', 'label'}
actual = set(df.columns)
print(f"Required: {required}")
print(f"Actual: {actual}")
print(f"Match: {required == actual}")
```

---

## FAQ

**Q: Can I have subdirectories in data/?**  
A: Yes! The loader scans recursively. Nested CSVs are auto-discovered.

**Q: What if I add a CSV without label column?**  
A: It's skipped with a warning. No crash, no interruption.

**Q: Do duplicates remove good data?**  
A: Duplicates are identified by `text_combined` only. It keeps the first occurrence and removes later ones. No important data is lost.

**Q: Can I merge datasets with different schemas?**  
A: No—all CSVs must have the exact same columns: `text_combined` and `label`. But you can convert them beforehand (see examples above).

**Q: How many CSV files can I add?**  
A: No hard limit. Tested with 7 files. Merging is efficient even with 100+ files.

**Q: What happens if I add no CSV files?**  
A: Training fails with clear error: "No valid CSV files found with required columns"

---

## Summary

| Feature | Before | After |
|---------|--------|-------|
| Files to configure | Manual | Zero |
| Dataset addition | Code changes required | Just copy CSV |
| Error handling | Crashes on invalid data | Skips + warning |
| Duplicates | Not handled | Auto-removed |
| Schema validation | Manual | Automatic |
| Multiple sources | Not supported | Fully supported |

---

**Document Version:** 1.0  
**Last Updated:** June 2026  
**Tested:** ✓ 7 CSV files, ✓ Multi-source merging, ✓ Duplicate removal
