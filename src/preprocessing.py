"""Data preprocessing and exploratory data analysis module."""

import re
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

from .utils import setup_logger, DATA_DIR, RESULTS_DIR, set_seed

logger = setup_logger(__name__)

# Download required NLTK data
_NLTK_DOWNLOADS = ["stopwords", "punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger"]
for resource in _NLTK_DOWNLOADS:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Initialize lemmatizer and stopwords
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# Required columns for dataset validation
REQUIRED_COLUMNS = {"text_combined", "label"}


def _get_csv_files() -> List[Path]:
    """Get all CSV files from data directory recursively.

    Returns:
        List of Path objects for all CSV files found

    Raises:
        FileNotFoundError: If data directory does not exist
    """
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found at {DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}")

    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV file(s) in {DATA_DIR}")

    return csv_files


def _validate_csv(csv_path: Path) -> bool:
    """Validate CSV file has required columns.

    Args:
        csv_path: Path to CSV file

    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_path, nrows=1)
        columns = set(df.columns)

        if not REQUIRED_COLUMNS.issubset(columns):
            logger.warning(
                f"Skipping {csv_path.name}: Missing columns. "
                f"Expected {REQUIRED_COLUMNS}, Found {columns}"
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Skipping {csv_path.name}: Error reading file - {str(e)}")
        return False


def _load_single_csv(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """Load single CSV file with error handling.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (dataframe or None, success_bool)
    """
    try:
        df = pd.read_csv(csv_path)

        # Select only required columns
        df = df[["text_combined", "label"]].copy()

        logger.info(f"Loaded {csv_path.name}: {df.shape[0]} rows")

        return df, True

    except Exception as e:
        logger.warning(f"Error loading {csv_path.name}: {str(e)}")
        return None, False


def load_dataset() -> Tuple[pd.DataFrame, int]:
    """Load and merge all labeled CSV files from data directory.

    Automatically scans data/ folder recursively for all CSV files,
    validates they contain required columns (text_combined, label),
    merges them, removes duplicates, and handles missing values.

    Returns:
        Tuple of (merged_dataframe, total_rows_before_cleaning)

    Raises:
        FileNotFoundError: If data directory not found
        ValueError: If no valid CSV files found
    """
    logger.info("=" * 70)
    logger.info("LOADING DATASET: Multi-CSV Auto-Discovery")
    logger.info("=" * 70)

    # Get all CSV files
    csv_files = _get_csv_files()

    if not csv_files:
        logger.error("No CSV files found in data directory")
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    logger.info(f"\nFound {len(csv_files)} CSV file(s) to process:")
    for csv_file in csv_files:
        logger.info(f"  - {csv_file.name}")

    # Validate CSV files
    logger.info("\nValidating CSV files...")
    valid_files = []
    skipped_files = []

    for csv_path in csv_files:
        if _validate_csv(csv_path):
            valid_files.append(csv_path)
        else:
            skipped_files.append(csv_path.name)

    logger.info(f"Valid files: {len(valid_files)}, Skipped files: {len(skipped_files)}")

    if not valid_files:
        logger.error("No valid CSV files found with required columns")
        raise ValueError(
            f"No valid CSV files found. Required columns: {REQUIRED_COLUMNS}"
        )

    # Load valid CSV files
    logger.info("\nLoading valid CSV files...")
    dataframes = []
    total_rows_before_cleaning = 0

    for csv_path in valid_files:
        df, success = _load_single_csv(csv_path)
        if success:
            dataframes.append(df)
            total_rows_before_cleaning += df.shape[0]

    if not dataframes:
        logger.error("Failed to load any CSV files")
        raise ValueError("Failed to load any CSV files")

    # Merge all dataframes
    logger.info(f"\nMerging {len(dataframes)} dataframe(s)...")
    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"After merge: {df.shape[0]} rows, {df.shape[1]} columns")

    # Data cleaning and validation
    logger.info("\nPerforming data cleaning and validation...")

    # Count duplicates before removal
    duplicate_count = df.duplicated(subset=["text_combined"]).sum()

    # Remove duplicates
    if duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} duplicate email(s)...")
        df = df.drop_duplicates(subset=["text_combined"], keep="first")

    # Count missing values before removal
    missing_text = df["text_combined"].isnull().sum()
    missing_label = df["label"].isnull().sum()

    if missing_text > 0 or missing_label > 0:
        logger.warning(
            f"Found missing values: text_combined={missing_text}, label={missing_label}"
        )
        df = df.dropna(subset=["text_combined", "label"])
        logger.info(f"After removing missing values: {df.shape[0]} rows")

    # Convert text_combined to string
    df["text_combined"] = df["text_combined"].astype(str)
    df["label"] = df["label"].astype(int)

    # Dataset statistics
    logger.info("\n" + "=" * 70)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 70)
    logger.info(f"CSV files loaded:        {len(valid_files)}")
    logger.info(f"CSV files skipped:       {len(skipped_files)}")
    logger.info(f"Total rows (raw):        {total_rows_before_cleaning:,}")
    logger.info(f"Total rows (cleaned):    {df.shape[0]:,}")
    logger.info(f"Duplicates removed:      {duplicate_count:,}")
    logger.info(f"Missing values removed:  {missing_text + missing_label:,}")
    logger.info(f"Final dataset shape:     {df.shape}")

    # Class distribution
    logger.info(f"\nClass distribution:")
    class_dist = df["label"].value_counts().sort_index()
    for label, count in class_dist.items():
        pct = (count / len(df)) * 100
        label_name = "Ham" if label == 0 else "Phishing"
        logger.info(f"  {label_name:10s} ({label}): {count:6,} ({pct:5.2f}%)")

    logger.info("=" * 70 + "\n")

    return df, total_rows_before_cleaning


def explore_dataset(df: pd.DataFrame) -> None:
    """Perform and display exploratory data analysis.

    Args:
        df: Input dataframe
    """
    logger.info("=== EXPLORATORY DATA ANALYSIS ===")

    # Basic statistics
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found")

    # Class distribution
    logger.info(f"\nClass distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(
        f"Class proportions:\n{df['label'].value_counts(normalize=True).to_string()}"
    )

    # Text statistics
    df["text_length"] = df["text_combined"].apply(len)
    df["word_count"] = df["text_combined"].apply(lambda x: len(x.split()))

    logger.info(f"\nText length statistics:")
    logger.info(f"  Mean: {df['text_length'].mean():.2f} characters")
    logger.info(f"  Median: {df['text_length'].median():.2f} characters")
    logger.info(f"  Max: {df['text_length'].max():.2f} characters")
    logger.info(f"  Min: {df['text_length'].min():.2f} characters")

    logger.info(f"\nWord count statistics:")
    logger.info(f"  Mean: {df['word_count'].mean():.2f} words")
    logger.info(f"  Median: {df['word_count'].median():.2f} words")
    logger.info(f"  Max: {df['word_count'].max():.2f} words")
    logger.info(f"  Min: {df['word_count'].min():.2f} words")


def visualize_eda(df: pd.DataFrame) -> None:
    """Generate and save EDA visualizations.

    Args:
        df: Input dataframe
    """
    set_seed()

    logger.info("Generating EDA visualizations...")

    # Prepare data
    df["text_length"] = df["text_combined"].apply(len)

    # Figure 1: Label distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    label_counts = df["label"].value_counts()
    label_names = ["Ham", "Phishing"]
    colors = ["#2ecc71", "#e74c3c"]

    ax.bar(label_names, label_counts.values, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Email Class Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(label_counts.values):
        ax.text(i, v + 1000, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_label_distribution.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: 01_label_distribution.png")
    plt.close()

    # Figure 2: Text length histogram by class
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, name, color in [(0, "Ham", "#2ecc71"), (1, "Phishing", "#e74c3c")]:
        data = df[df["label"] == label]["text_length"]
        ax.hist(
            data, bins=50, label=name, alpha=0.6, color=color, edgecolor="black"
        )

    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Email Text Length Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_text_length_histogram.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: 02_text_length_histogram.png")
    plt.close()

    logger.info("EDA visualizations generated successfully")


def _remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def _remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )


def _remove_special_urls(text: str) -> str:
    """Remove www URLs and email patterns."""
    text = re.sub(r"www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}", "", text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)
    return text


def _count_urls(text: str) -> int:
    """Count URLs in original text."""
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    www_pattern = r"www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}"
    return len(re.findall(url_pattern, text)) + len(re.findall(www_pattern, text))


def clean_email(text: str) -> str:
    """Clean and preprocess email text.

    Args:
        text: Raw email text

    Returns:
        Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = _remove_html_tags(text)

    # Remove URLs
    text = _remove_urls(text)
    text = _remove_special_urls(text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess entire dataset.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with cleaned text
    """
    logger.info("Preprocessing dataset...")

    df = df.copy()
    df["text_cleaned"] = df["text_combined"].apply(clean_email)

    # Remove empty rows
    df = df[df["text_cleaned"].str.len() > 0]

    logger.info(f"After preprocessing: {df.shape[0]} rows remaining")

    return df


def get_statistics(text: str) -> dict:
    """Extract statistical features from text.

    Args:
        text: Email text

    Returns:
        Dictionary of statistical features
    """
    text_length = len(text)
    word_count = len(text.split())
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    url_count = _count_urls(text)

    uppercase_ratio = (
        uppercase_count / text_length if text_length > 0 else 0
    )
    digit_ratio = digit_count / text_length if text_length > 0 else 0

    return {
        "email_length": text_length,
        "word_count": word_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "url_count": url_count,
    }
