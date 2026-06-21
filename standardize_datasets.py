"""Standardize raw phishing email CSV files into one simple dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/phishing_email_standardized.csv")

OUTPUT_COLUMNS = [
    "sender",
    "receiver",
    "date",
    "subject",
    "body",
    "urls",
    "label",
    "text_combined",
]

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

COLUMN_ALIASES = {
    "sender": {"sender", "from", "from_email", "from_address"},
    "receiver": {"receiver", "recipient", "to", "to_email", "to_address"},
    "date": {"date", "sent_date", "timestamp", "time"},
    "subject": {"subject", "email_subject"},
    "body": {"body", "message", "content", "email_body"},
    "urls": {"urls", "url", "links", "link"},
    "label": {"label", "class", "target", "category", "type", "is_phishing"},
    "text_combined": {"text_combined", "text", "email_text", "email", "message_text"},
}

PHISHING_LABELS = {
    "phishing",
    "phish",
    "spam",
    "malicious",
    "bad",
    "true",
    "yes",
    "1",
}

LEGITIMATE_LABELS = {
    "legitimate",
    "legit",
    "ham",
    "benign",
    "safe",
    "false",
    "no",
    "0",
}

EMPTY_VALUES = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def normalize_column_name(name: object) -> str:
    """Normalize column names to lowercase underscore format."""
    normalized = str(name).strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized, unique column names."""
    counts: dict[str, int] = {}
    columns: list[str] = []

    for column in df.columns:
        normalized = normalize_column_name(column)
        counts[normalized] = counts.get(normalized, 0) + 1
        if counts[normalized] > 1:
            normalized = f"{normalized}_{counts[normalized]}"
        columns.append(normalized)

    normalized_df = df.copy()
    normalized_df.columns = columns
    return normalized_df


def find_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """Find the first matching column for a canonical field."""
    aliases = COLUMN_ALIASES[canonical_name]
    for column in df.columns:
        if column in aliases:
            return column
    return None


def read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    """Read a CSV using common encodings."""
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            last_error = exc
            try:
                return pd.read_csv(
                    path,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as parser_exc:
                last_error = parser_exc
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read {path}: {last_error}")


def clean_value(value: object) -> str:
    """Convert missing-like values to an empty string."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in EMPTY_VALUES:
        return ""
    return text


def normalize_label(value: object) -> int | None:
    """Convert supported labels to 0 for legitimate and 1 for phishing."""
    text = clean_value(value).lower()
    if not text:
        return None

    if text.endswith(".0"):
        text = text[:-2]

    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in PHISHING_LABELS:
        return 1
    if text in LEGITIMATE_LABELS:
        return 0
    return None


def get_text_column(df: pd.DataFrame, canonical_name: str) -> pd.Series:
    """Return a cleaned text column or empty strings when absent."""
    column = find_column(df, canonical_name)
    if column is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].map(clean_value)


def build_standardized_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the standardized columns from one normalized raw dataframe."""
    output = pd.DataFrame(index=df.index)
    for column in ["sender", "receiver", "date", "subject", "body", "urls"]:
        output[column] = get_text_column(df, column)

    text_column = find_column(df, "text_combined")
    if text_column is not None:
        output["text_combined"] = df[text_column].map(clean_value)
    else:
        output["text_combined"] = (
            output[["subject", "body", "urls"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    label_column = find_column(df, "label")
    if label_column is None:
        output["label"] = None
    else:
        output["label"] = df[label_column].map(normalize_label)

    return output[OUTPUT_COLUMNS]


def standardize_datasets(input_dir: Path, output_path: Path) -> pd.DataFrame:
    """Standardize all raw CSV files and save one processed CSV."""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.rglob("*.csv"))
    total_rows_before = 0
    frames: list[pd.DataFrame] = []

    print(f"Input directory: {input_dir}")
    print(f"CSV files found: {len(csv_files)}")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        raw_df = normalize_columns(read_csv_with_fallbacks(csv_path))
        total_rows_before += len(raw_df)
        standardized = build_standardized_frame(raw_df)
        frames.append(standardized)
        print(f"  rows loaded: {len(raw_df):,}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows_after_load = len(combined)
    combined = combined.dropna(subset=["label"]).copy()
    combined["label"] = combined["label"].astype(int)
    combined["text_combined"] = combined["text_combined"].map(clean_value)
    combined = combined[combined["text_combined"] != ""].copy()

    before_exact_dedup = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    exact_duplicates_removed = before_exact_dedup - len(combined)

    label_counts_per_text = combined.groupby("text_combined")["label"].nunique()
    conflicting_texts = label_counts_per_text[label_counts_per_text > 1].index
    conflicting_rows_removed = int(combined["text_combined"].isin(conflicting_texts).sum())
    combined = combined[~combined["text_combined"].isin(conflicting_texts)].copy()

    before_text_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text_combined"], keep="first")
    duplicate_text_rows_removed = before_text_dedup - len(combined)

    combined = combined[OUTPUT_COLUMNS].reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    duplicate_rows_removed = exact_duplicates_removed + duplicate_text_rows_removed
    rows_after_cleaning = len(combined)
    class_distribution = combined["label"].value_counts().sort_index()

    print("\nStandardization summary")
    print(f"Total rows before cleaning: {total_rows_before:,}")
    print(f"Rows after initial load: {rows_after_load:,}")
    print(f"Total rows after cleaning: {rows_after_cleaning:,}")
    print(f"Duplicate rows removed: {duplicate_rows_removed:,}")
    print(f"Conflicting duplicated texts removed: {len(conflicting_texts):,}")
    print(f"Rows removed because of conflicting labels: {conflicting_rows_removed:,}")
    print("Class distribution:")
    if class_distribution.empty:
        print("  no usable labeled rows")
    else:
        for label, count in class_distribution.items():
            print(f"  {label}: {count:,}")
    print(f"Saved standardized CSV to {output_path}")

    return combined


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Standardize phishing email CSV files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the standardization workflow."""
    args = parse_args()
    standardize_datasets(args.input_dir, args.output)


if __name__ == "__main__":
    main()
