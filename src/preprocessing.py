"""Data preprocessing and exploratory data analysis module."""

import re
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from .utils import setup_logger, DATA_DIR, RESULTS_DIR, set_seed

logger = setup_logger(__name__)

# Initialize NLTK helpers without downloading resources at import time.
LEMMATIZER = WordNetLemmatizer()
_TOKENIZER_AVAILABLE = True
_LEMMATIZER_AVAILABLE = True

FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = FALLBACK_STOP_WORDS
    logger.warning(
        "NLTK stopwords data is not installed. Using a small built-in "
        "fallback stopword list. For full preprocessing, run: "
        "python -m nltk.downloader stopwords punkt wordnet"
    )


TEXT_COLUMN_CANDIDATES = [
    "text_combined",
    "text",
    "email",
    "email_text",
    "email text",
    "body",
    "message",
    "content",
    "subject",
    "mail",
    "email_body",
    "email body",
    "Body",
    "Subject",
    "Message",
    "Content",
    "Text",
    "Email Text",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "Label",
    "class",
    "Class",
    "target",
    "Target",
    "phishing",
    "Phishing",
    "category",
    "Category",
    "type",
    "Type",
]

HAM_LABELS = {
    "0",
    "ham",
    "benign",
    "legitimate",
    "legit",
    "safe",
    "normal",
    "not phishing",
    "non-phishing",
    "non phishing",
    "valid",
    "clean",
    "false",
    "no",
    "safe email",
    "ham email",
}

PHISHING_LABELS = {
    "1",
    "phishing",
    "phish",
    "spam",
    "fraud",
    "malicious",
    "scam",
    "nigerian fraud",
    "nigerian_fraud",
    "suspicious",
    "attack",
    "true",
    "yes",
    "phishing email",
    "spam email",
    "fraud email",
}


def _normalize_column_name(name: str) -> str:
    """Normalize column name for robust matching."""
    return str(name).strip().lower().replace("_", " ")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column from candidate names."""
    normalized_map = {
        _normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    return None


def _read_csv_safely(csv_file: Path) -> pd.DataFrame:
    """Read CSV file with fallback encodings."""
    try:
        return pd.read_csv(csv_file, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_file, encoding="latin1", low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {csv_file.name}: {exc}") from exc


def _build_text_combined(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Build text_combined column from available text-like columns.

    Priority:
    1. Use existing text_combined if available.
    2. Combine known text-like columns.
    3. Fallback to object columns that are likely text.
    """

    existing_text_col = _find_column(df, ["text_combined"])
    if existing_text_col:
        return df[existing_text_col].astype(str)

    matched_text_columns = []

    normalized_label_candidates = {
        _normalize_column_name(col)
        for col in LABEL_COLUMN_CANDIDATES
    }

    for col in df.columns:
        normalized_col = _normalize_column_name(col)

        if normalized_col in normalized_label_candidates:
            continue

        for candidate in TEXT_COLUMN_CANDIDATES:
            if normalized_col == _normalize_column_name(candidate):
                matched_text_columns.append(col)
                break

    matched_text_columns = list(dict.fromkeys(matched_text_columns))

    if matched_text_columns:
        return (
            df[matched_text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

    # Fallback: use object/string columns that are likely email text
    object_columns = [
        col
        for col in df.columns
        if df[col].dtype == "object"
    ]

    possible_text_columns = []

    for col in object_columns:
        normalized_col = _normalize_column_name(col)

        if normalized_col in normalized_label_candidates:
            continue

        avg_length = (
            df[col]
            .dropna()
            .astype(str)
            .str.len()
            .mean()
        )

        if pd.notna(avg_length) and avg_length >= 10:
            possible_text_columns.append(col)

    if possible_text_columns:
        return (
            df[possible_text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

    return None


def _normalize_label_value(value) -> Optional[int]:
    """Normalize one label value into binary format."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        if value == 0:
            return 0
        if value == 1:
            return 1

    if isinstance(value, (float, np.floating)):
        if value == 0.0:
            return 0
        if value == 1.0:
            return 1

    value_str = str(value).strip().lower()
    value_str = value_str.replace("-", " ")
    value_str = value_str.replace("_", " ")
    value_str = re.sub(r"\s+", " ", value_str)

    if value_str in HAM_LABELS:
        return 0

    if value_str in PHISHING_LABELS:
        return 1

    if "phish" in value_str:
        return 1

    if "fraud" in value_str:
        return 1

    if "spam" in value_str:
        return 1

    if "scam" in value_str:
        return 1

    if "malicious" in value_str:
        return 1

    if "safe" in value_str:
        return 0

    if "ham" in value_str:
        return 0

    if "benign" in value_str:
        return 0

    if "legitimate" in value_str:
        return 0

    return None


def _normalize_label_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """Find and normalize label column."""
    label_col = _find_column(df, LABEL_COLUMN_CANDIDATES)

    if not label_col:
        return None

    normalized = df[label_col].apply(_normalize_label_value)

    return normalized


def _load_single_dataset(csv_file: Path) -> Optional[pd.DataFrame]:
    """
    Load one CSV file and normalize it into:

    - text_combined
    - label
    """

    try:
        raw_df = _read_csv_safely(csv_file)

        if raw_df.empty:
            logger.warning(f"Skipping {csv_file.name}: empty CSV file")
            return None

        text_combined = _build_text_combined(raw_df)

        if text_combined is None:
            logger.warning(
                f"Skipping {csv_file.name}: no valid text column found"
            )
            return None

        labels = _normalize_label_column(raw_df)

        if labels is None:
            logger.warning(
                f"Skipping {csv_file.name}: no valid label column found"
            )
            return None

        normalized_df = pd.DataFrame(
            {
                "text_combined": text_combined,
                "label": labels,
            }
        )

        before_drop = len(normalized_df)

        normalized_df = normalized_df.dropna(
            subset=["text_combined", "label"]
        )

        normalized_df["text_combined"] = (
            normalized_df["text_combined"]
            .astype(str)
            .str.strip()
        )

        normalized_df = normalized_df[
            normalized_df["text_combined"] != ""
        ]

        normalized_df["label"] = normalized_df["label"].astype(int)

        normalized_df = normalized_df[
            normalized_df["label"].isin([0, 1])
        ]

        if normalized_df.empty:
            logger.warning(
                f"Skipping {csv_file.name}: no usable rows after normalization"
            )
            return None

        logger.info(
            f"Loaded {csv_file.name}: "
            f"{len(normalized_df):,}/{before_drop:,} usable rows"
        )

        return normalized_df

    except Exception as exc:
        logger.error(f"Error loading {csv_file.name}: {exc}")
        return None


def load_dataset(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, int]:
    """
    Load and merge all labeled CSV files from the data directory.

    The function recursively scans the data directory and loads every CSV file
    that can be normalized into two columns:

    - text_combined
    - label

    Label format:

    - 0 = Ham / Benign
    - 1 = Phishing / Spam / Fraud / Malicious

    Args:
        data_dir: Optional custom data directory. If None, DATA_DIR is used.

    Returns:
        Tuple[pd.DataFrame, int]: merged dataframe and total number of rows.

    Raises:
        FileNotFoundError: If no CSV files are found.
        ValueError: If no valid dataset can be loaded.
    """

    dataset_dir = Path(data_dir) if data_dir else DATA_DIR

    csv_files = sorted(dataset_dir.rglob("*.csv"))

    if not csv_files:
        logger.error(f"No CSV files found in {dataset_dir}")
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir}"
        )

    logger.info("=" * 70)
    logger.info("LOADING ALL LABELED CSV DATASETS")
    logger.info("=" * 70)
    logger.info(f"Data directory: {dataset_dir}")
    logger.info(f"CSV files found: {len(csv_files)}")

    dataframes = []
    skipped_files = []

    total_rows_before_cleaning = 0

    for csv_file in csv_files:
        logger.info(f"Processing file: {csv_file.name}")

        loaded_df = _load_single_dataset(csv_file)

        if loaded_df is None:
            skipped_files.append(csv_file.name)
            continue

        total_rows_before_cleaning += len(loaded_df)
        dataframes.append(loaded_df)

    if not dataframes:
        raise ValueError(
            "No valid CSV datasets could be loaded. "
            "Please make sure your CSV files contain email text and labels."
        )

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    logger.info("=" * 70)
    logger.info("MERGING AND CLEANING DATASETS")
    logger.info("=" * 70)

    before_final_cleaning = len(combined_df)

    combined_df = combined_df.dropna(
        subset=["text_combined", "label"]
    )

    combined_df["text_combined"] = (
        combined_df["text_combined"]
        .astype(str)
        .str.strip()
    )

    combined_df = combined_df[
        combined_df["text_combined"] != ""
    ]

    after_missing_removal = len(combined_df)

    combined_df = combined_df.drop_duplicates(
        subset=["text_combined"]
    )

    after_duplicate_removal = len(combined_df)

    combined_df["label"] = combined_df["label"].astype(int)

    combined_df = combined_df[["text_combined", "label"]]

    logger.info(f"CSV files found: {len(csv_files)}")
    logger.info(f"CSV files loaded: {len(dataframes)}")
    logger.info(f"CSV files skipped: {len(skipped_files)}")

    if skipped_files:
        logger.warning(f"Skipped files: {skipped_files}")

    logger.info(
        f"Rows before final cleaning: {before_final_cleaning:,}"
    )

    logger.info(
        f"Rows removed due to missing or empty values: "
        f"{before_final_cleaning - after_missing_removal:,}"
    )

    logger.info(
        f"Duplicate rows removed: "
        f"{after_missing_removal - after_duplicate_removal:,}"
    )

    logger.info(
        f"Final dataset rows: {after_duplicate_removal:,}"
    )

    logger.info("=" * 70)
    logger.info("FINAL CLASS DISTRIBUTION")
    logger.info("=" * 70)

    class_distribution = (
        combined_df["label"]
        .value_counts()
        .sort_index()
    )

    logger.info(f"\n{class_distribution.to_string()}")

    if 0 not in class_distribution.index:
        logger.warning("Warning: no Ham class found in final dataset")

    if 1 not in class_distribution.index:
        logger.warning("Warning: no Phishing class found in final dataset")

    return combined_df, len(combined_df)


def explore_dataset(df: pd.DataFrame) -> None:
    """Perform and display exploratory data analysis.

    Args:
        df: Input dataframe
    """
    logger.info("=== EXPLORATORY DATA ANALYSIS ===")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")

    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found")

    logger.info(f"\nClass distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(
        f"Class proportions:\n{df['label'].value_counts(normalize=True).to_string()}"
    )

    df["text_length"] = df["text_combined"].apply(len)
    df["word_count"] = df["text_combined"].apply(lambda x: len(x.split()))

    logger.info("\nText length statistics:")
    logger.info(f"  Mean: {df['text_length'].mean():.2f} characters")
    logger.info(f"  Median: {df['text_length'].median():.2f} characters")
    logger.info(f"  Max: {df['text_length'].max():.2f} characters")
    logger.info(f"  Min: {df['text_length'].min():.2f} characters")

    logger.info("\nWord count statistics:")
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

    df["text_length"] = df["text_combined"].apply(len)

    fig, ax = plt.subplots(figsize=(10, 6))
    label_counts = df["label"].value_counts().sort_index()

    label_names = []
    values = []

    for label in label_counts.index:
        if label == 0:
            label_names.append("Ham")
        elif label == 1:
            label_names.append("Phishing")
        else:
            label_names.append(str(label))

        values.append(label_counts[label])

    colors = ["#2ecc71", "#e74c3c"]

    ax.bar(
        label_names,
        values,
        color=colors[:len(values)],
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Email Class Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "01_label_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    logger.info("Saved: 01_label_distribution.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, name, color in [
        (0, "Ham", "#2ecc71"),
        (1, "Phishing", "#e74c3c"),
    ]:
        data = df[df["label"] == label]["text_length"]

        if len(data) == 0:
            continue

        ax.hist(
            data,
            bins=50,
            label=name,
            alpha=0.6,
            color=color,
            edgecolor="black",
        )

    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Email Text Length Distribution by Class",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "02_text_length_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
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


def _tokenize_clean_text(text: str) -> List[str]:
    """Tokenize text with a regex fallback when NLTK punkt is unavailable."""
    global _TOKENIZER_AVAILABLE

    if _TOKENIZER_AVAILABLE:
        try:
            return word_tokenize(text)
        except LookupError:
            _TOKENIZER_AVAILABLE = False
            logger.warning(
                "NLTK punkt tokenizer data is not installed. Falling back to "
                "regex tokenization. For full preprocessing, run: "
                "python -m nltk.downloader punkt"
            )

    return re.findall(r"\b\w+\b", text)


def _lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens when WordNet is available, otherwise return tokens."""
    global _LEMMATIZER_AVAILABLE

    if not _LEMMATIZER_AVAILABLE:
        return tokens

    try:
        return [LEMMATIZER.lemmatize(token) for token in tokens]
    except LookupError:
        _LEMMATIZER_AVAILABLE = False
        logger.warning(
            "NLTK WordNet data is not installed. Skipping lemmatization. "
            "For full preprocessing, run: python -m nltk.downloader wordnet"
        )
        return tokens


def clean_email(text: str) -> str:
    """Clean and preprocess email text.

    Args:
        text: Raw email text

    Returns:
        Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = _remove_html_tags(text)

    text = _remove_urls(text)
    text = _remove_special_urls(text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d+", "", text)

    tokens = _tokenize_clean_text(text)

    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    tokens = _lemmatize_tokens(tokens)

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

    uppercase_ratio = uppercase_count / text_length if text_length > 0 else 0
    digit_ratio = digit_count / text_length if text_length > 0 else 0

    return {
        "email_length": text_length,
        "word_count": word_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "url_count": url_count,
    }
