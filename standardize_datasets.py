"""Standardize heterogeneous raw phishing email CSV datasets.

The output schema is:
source_file,row_id,sender,receiver,date,subject,body,urls,text_combined,label
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COMMON_COLUMNS = [
    "source_file",
    "row_id",
    "sender",
    "receiver",
    "date",
    "subject",
    "body",
    "urls",
    "text_combined",
    "label",
]

REPORT_COLUMNS = [
    "source_file",
    "rows_loaded",
    "rows_kept",
    "rows_dropped",
    "label_0_count",
    "label_1_count",
    "label_distribution",
    "status",
    "message",
]

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

EMPTY_VALUES = {"", "nan", "none", "null", "nat", "na", "n/a", "<na>"}

COLUMN_ALIASES = {
    "sender": ["sender", "from", "from_email", "from address", "mail from"],
    "receiver": ["receiver", "recipient", "to", "to_email", "to address", "mail to"],
    "date": ["date", "sent date", "timestamp", "time"],
    "subject": ["subject", "email subject"],
    "body": ["body", "message", "content", "email body"],
    "urls": ["urls", "url", "links", "link"],
    "text_combined": [
        "text_combined",
        "text combined",
        "text",
        "email_text",
        "email text",
        "mail text",
    ],
    "label": ["label", "class", "target", "category", "type", "is_phishing"],
}

HAM_LABELS = {
    "0",
    "ham",
    "legitimate",
    "legit",
    "benign",
    "safe",
    "non phishing",
    "nonphishing",
    "not phishing",
    "not phish",
    "valid",
    "clean",
    "normal",
    "false",
    "no",
}

PHISHING_LABELS = {
    "1",
    "phishing",
    "phish",
    "malicious",
    "spam",
    "fraud",
    "scam",
    "suspicious",
    "attack",
    "true",
    "yes",
}


class EmptyCsvError(Exception):
    """Raised when pandas finds no rows or columns in a CSV."""


def setup_logger() -> logging.Logger:
    """Create a console logger for the standardization workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("standardize_datasets")


logger = setup_logger()


def normalize_column_name(name: object) -> str:
    """Normalize a column name for case-insensitive matching."""
    text = str(name).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    """Find a source column by canonical name and aliases."""
    normalized_to_original: dict[str, str] = {}

    for column in df.columns:
        normalized = normalize_column_name(column)
        if normalized not in normalized_to_original:
            normalized_to_original[normalized] = column

    for alias in COLUMN_ALIASES[canonical_name]:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_to_original:
            return normalized_to_original[normalized_alias]

    return None


def read_csv_with_fallbacks(csv_path: Path) -> tuple[pd.DataFrame, str, str]:
    """Read a CSV using UTF-8 first, then fallback encodings."""
    last_error: Optional[Exception] = None

    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            return df, encoding, ""
        except pd.errors.EmptyDataError as exc:
            raise EmptyCsvError(f"{csv_path.name} is empty") from exc
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            last_error = exc
            try:
                df = pd.read_csv(
                    csv_path,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
                return df, encoding, "Parser issues detected; bad lines were skipped."
            except Exception as parser_exc:
                last_error = parser_exc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read {csv_path.name}: {last_error}")


def clean_text_value(value: object) -> str:
    """Convert null-like values to empty strings and strip whitespace."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text.lower() in EMPTY_VALUES:
        return ""

    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    """Clean a text-like pandas Series."""
    return series.map(clean_text_value)


def empty_text_series(length: int) -> pd.Series:
    """Create an empty string Series with a stable default index."""
    return pd.Series([""] * length)


def normalize_label(value: object) -> Optional[int]:
    """Normalize one label to 0 for ham or 1 for phishing."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return int(value)
        return None

    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return int(value)
        return None

    text = clean_text_value(value).lower()
    if not text:
        return None

    try:
        numeric_value = float(text)
        if numeric_value in (0.0, 1.0):
            return int(numeric_value)
    except ValueError:
        pass

    normalized = text.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in HAM_LABELS:
        return 0

    if normalized in PHISHING_LABELS:
        return 1

    tokens = set(normalized.split())

    if normalized.startswith(("not ", "non ")) and tokens.intersection(
        {"phishing", "phish", "spam", "fraud", "scam", "malicious"}
    ):
        return 0

    if tokens.intersection({"phishing", "phish", "spam", "fraud", "scam", "malicious"}):
        return 1

    if tokens.intersection({"ham", "legitimate", "legit", "benign", "safe"}):
        return 0

    return None


def get_clean_column(df: pd.DataFrame, canonical_name: str) -> pd.Series:
    """Return a cleaned source column or an empty column if missing."""
    source_column = find_column(df, canonical_name)
    if source_column is None:
        return empty_text_series(len(df))

    return clean_text_series(df[source_column]).reset_index(drop=True)


def build_text_combined(raw_df: pd.DataFrame, standardized_df: pd.DataFrame) -> pd.Series:
    """Keep text_combined when present, otherwise combine subject, body, and urls."""
    text_column = find_column(raw_df, "text_combined")
    if text_column is not None:
        return clean_text_series(raw_df[text_column]).reset_index(drop=True)

    return (
        standardized_df[["subject", "body", "urls"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def standardize_one_file(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load and standardize a single CSV file."""
    report = {
        "source_file": csv_path.name,
        "rows_loaded": 0,
        "rows_kept": 0,
        "rows_dropped": 0,
        "label_0_count": 0,
        "label_1_count": 0,
        "label_distribution": "0=0; 1=0",
        "status": "skipped",
        "message": "",
    }

    try:
        raw_df, encoding, read_warning = read_csv_with_fallbacks(csv_path)
        report["rows_loaded"] = len(raw_df)
        message_parts = [f"encoding={encoding}"]
        if read_warning:
            message_parts.append(read_warning)

        if raw_df.empty:
            report["status"] = "empty"
            report["message"] = "; ".join(message_parts)
            logger.warning("Skipping %s: empty CSV", csv_path.name)
            return pd.DataFrame(columns=COMMON_COLUMNS), report

        standardized_df = pd.DataFrame(
            {
                "source_file": csv_path.name,
                "row_id": raw_df.index.to_series().reset_index(drop=True) + 1,
                "sender": get_clean_column(raw_df, "sender"),
                "receiver": get_clean_column(raw_df, "receiver"),
                "date": get_clean_column(raw_df, "date"),
                "subject": get_clean_column(raw_df, "subject"),
                "body": get_clean_column(raw_df, "body"),
                "urls": get_clean_column(raw_df, "urls"),
            }
        )

        standardized_df["text_combined"] = build_text_combined(raw_df, standardized_df)

        label_column = find_column(raw_df, "label")
        if label_column is None:
            report["status"] = "missing_label"
            report["rows_dropped"] = report["rows_loaded"]
            report["message"] = "; ".join(message_parts + ["No label column found."])
            logger.warning("Skipping %s: no label column found", csv_path.name)
            return pd.DataFrame(columns=COMMON_COLUMNS), report

        standardized_df["label"] = raw_df[label_column].map(normalize_label).reset_index(
            drop=True
        )

        before_row_cleaning = len(standardized_df)
        standardized_df = standardized_df.dropna(subset=["label"])
        standardized_df["text_combined"] = clean_text_series(
            standardized_df["text_combined"]
        )
        standardized_df = standardized_df[standardized_df["text_combined"] != ""]

        standardized_df["label"] = standardized_df["label"].astype(int)
        standardized_df = standardized_df[COMMON_COLUMNS].reset_index(drop=True)

        dropped_before_dedup = before_row_cleaning - len(standardized_df)
        report["status"] = "loaded"
        report["message"] = "; ".join(
            message_parts
            + [f"Rows removed before duplicate cleanup: {dropped_before_dedup}."]
        )

        logger.info(
            "Loaded %s: %s rows, %s usable before duplicate cleanup",
            csv_path.name,
            f"{report['rows_loaded']:,}",
            f"{len(standardized_df):,}",
        )

        return standardized_df, report

    except EmptyCsvError as exc:
        report["status"] = "empty"
        report["message"] = str(exc)
        logger.warning("Skipping %s: %s", csv_path.name, exc)
        return pd.DataFrame(columns=COMMON_COLUMNS), report
    except Exception as exc:
        report["status"] = "error"
        report["message"] = str(exc)
        logger.error("Skipping %s due to error: %s", csv_path.name, exc)
        return pd.DataFrame(columns=COMMON_COLUMNS), report


def update_report_after_dedup(reports: list[dict], combined_df: pd.DataFrame) -> pd.DataFrame:
    """Fill final kept/dropped counts and label distribution in the report."""
    if combined_df.empty:
        return pd.DataFrame(reports, columns=REPORT_COLUMNS)

    grouped = combined_df.groupby("source_file")["label"].value_counts().unstack(fill_value=0)

    for report in reports:
        source_file = report["source_file"]
        label_0_count = int(grouped.loc[source_file, 0]) if source_file in grouped.index and 0 in grouped.columns else 0
        label_1_count = int(grouped.loc[source_file, 1]) if source_file in grouped.index and 1 in grouped.columns else 0
        rows_kept = label_0_count + label_1_count

        report["rows_kept"] = rows_kept
        report["rows_dropped"] = max(int(report["rows_loaded"]) - rows_kept, 0)
        report["label_0_count"] = label_0_count
        report["label_1_count"] = label_1_count
        report["label_distribution"] = f"0={label_0_count}; 1={label_1_count}"

        if report["status"] == "loaded" and rows_kept == 0:
            report["status"] = "all_rows_dropped"

    return pd.DataFrame(reports, columns=REPORT_COLUMNS)


def standardize_datasets(input_dir: Path, output_path: Path, report_path: Path) -> pd.DataFrame:
    """Standardize all CSV files from input_dir and save the dataset and report."""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_resolved = output_path.resolve()
    report_resolved = report_path.resolve()

    csv_files = [
        path
        for path in sorted(input_dir.rglob("*.csv"))
        if path.resolve() not in {output_resolved, report_resolved}
    ]

    logger.info("Input directory: %s", input_dir)
    logger.info("CSV files found: %s", len(csv_files))

    if not csv_files:
        logger.warning("No CSV files found in %s", input_dir)
        empty_df = pd.DataFrame(columns=COMMON_COLUMNS)
        empty_report = pd.DataFrame(columns=REPORT_COLUMNS)
        empty_df.to_csv(output_path, index=False)
        empty_report.to_csv(report_path, index=False)
        return empty_df

    standardized_frames = []
    reports = []

    for csv_path in csv_files:
        logger.info("Processing %s", csv_path.name)
        standardized_df, report = standardize_one_file(csv_path)
        reports.append(report)
        if not standardized_df.empty:
            standardized_frames.append(standardized_df)

    if standardized_frames:
        combined_df = pd.concat(standardized_frames, ignore_index=True)
        rows_before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=["text_combined", "label"], keep="first"
        ).reset_index(drop=True)
        duplicate_rows = rows_before_dedup - len(combined_df)
    else:
        combined_df = pd.DataFrame(columns=COMMON_COLUMNS)
        duplicate_rows = 0

    report_df = update_report_after_dedup(reports, combined_df)

    combined_df.to_csv(output_path, index=False)
    report_df.to_csv(report_path, index=False)

    logger.info("Duplicate rows removed globally: %s", f"{duplicate_rows:,}")
    logger.info("Standardized dataset rows: %s", f"{len(combined_df):,}")
    if not combined_df.empty:
        logger.info(
            "Label distribution after cleaning:\n%s",
            combined_df["label"].value_counts().sort_index().to_string(),
        )
    logger.info("Saved standardized dataset to %s", output_path)
    logger.info("Saved standardization report to %s", report_path)

    return combined_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standardize raw phishing email CSV files into one schema."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/phishing_email_standardized.csv"),
        help="Path for the standardized output CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path for the standardization report CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standardization workflow."""
    args = parse_args()
    report_path = args.report or args.output.parent / "standardization_report.csv"
    standardize_datasets(args.input_dir, args.output, report_path)


if __name__ == "__main__":
    main()
