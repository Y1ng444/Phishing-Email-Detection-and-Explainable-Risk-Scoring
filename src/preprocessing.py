"""Data preprocessing and exploratory data analysis module."""

import re
import logging
from typing import Tuple, List
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


def load_dataset() -> Tuple[pd.DataFrame, int]:
    """Load phishing email dataset.

    Returns:
        Tuple of (dataframe, total_rows)

    Raises:
        FileNotFoundError: If dataset file not found
    """
    csv_path = DATA_DIR / "phishing_email.csv"
    if not csv_path.exists():
        logger.error(f"Dataset not found at {csv_path}")
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df, df.shape[0]


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
