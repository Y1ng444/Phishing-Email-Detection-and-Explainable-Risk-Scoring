"""Simple EDA for the standardized phishing email dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.text_cleaning import clean_text


DEFAULT_INPUT = Path("data/processed/phishing_email_standardized.csv")
DEFAULT_RESULTS_DIR = Path("results")


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the standardized dataset and validate required columns."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Standardized dataset not found at {input_path}. "
            "Run python standardize_datasets.py first."
        )

    df = pd.read_csv(input_path, low_memory=False)
    required = {"text_combined", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text_combined"] = df["text_combined"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Save a class distribution bar chart."""
    counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Legitimate", "Phishing"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#4c78a8", "#f58518"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Email count")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_text_length_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Save a text length distribution plot by class."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, name, color in [(0, "Legitimate", "#4c78a8"), (1, "Phishing", "#f58518")]:
        lengths = df.loc[df["label"] == label, "text_combined"].str.len()
        if lengths.empty:
            continue
        clipped = lengths.clip(upper=lengths.quantile(0.99))
        ax.hist(clipped, bins=40, alpha=0.6, label=name, color=color)

    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Characters, clipped at class 99th percentile")
    ax.set_ylabel("Email count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_missing_values(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save missing-value counts and rates for all standardized columns."""
    missing_counts = df.replace("", pd.NA).isna().sum()
    missing = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_rate": missing_counts.values / max(len(df), 1),
        }
    )
    missing.to_csv(output_path, index=False)
    return missing


def save_source_distribution(df: pd.DataFrame, output_path: Path) -> pd.DataFrame | None:
    """Save source-file distribution when source metadata is available."""
    if "source_file" not in df.columns:
        return None

    counts = df["source_file"].fillna("unknown").replace("", "unknown").value_counts()
    distribution = pd.DataFrame(
        {
            "source_file": counts.index,
            "row_count": counts.values,
            "row_rate": counts.values / max(len(df), 1),
        }
    )
    distribution.to_csv(output_path, index=False)
    return distribution


def save_duplicate_statistics(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save exact and cleaned-text duplicate statistics."""
    cleaned_text = df["text_combined"].map(clean_text)
    stats = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {
                "metric": "duplicate_rows_text_combined",
                "value": int(df.duplicated(subset=["text_combined"]).sum()),
            },
            {
                "metric": "duplicate_groups_text_combined",
                "value": int(df.loc[df["text_combined"].duplicated(keep=False), "text_combined"].nunique()),
            },
            {
                "metric": "duplicate_rows_text_cleaned",
                "value": int(cleaned_text.duplicated().sum()),
            },
            {
                "metric": "duplicate_groups_text_cleaned",
                "value": int(cleaned_text[cleaned_text.duplicated(keep=False)].nunique()),
            },
            {
                "metric": "rows_after_drop_duplicate_text_cleaned",
                "value": int(len(df.assign(text_cleaned=cleaned_text).drop_duplicates(subset=["text_cleaned"]))),
            },
        ]
    )
    stats.to_csv(output_path, index=False)
    return stats


def save_top_terms_by_class(df: pd.DataFrame, output_path: Path, top_n: int = 30) -> pd.DataFrame:
    """Save frequent unigram and bigram terms for each class."""
    rows: list[dict[str, int | str]] = []

    for label, class_name in [(0, "legitimate"), (1, "phishing")]:
        texts = df.loc[df["label"] == label, "text_combined"]
        if texts.empty:
            continue

        vectorizer = CountVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue

        counts = np.asarray(matrix.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        top_indices = np.argsort(counts)[-top_n:][::-1]

        for index in top_indices:
            rows.append(
                {
                    "label": int(label),
                    "class_name": class_name,
                    "term": str(terms[index]),
                    "count": int(counts[index]),
                }
            )

    top_terms = pd.DataFrame(rows)
    top_terms.to_csv(output_path, index=False)
    return top_terms


def run_eda(input_path: Path, results_dir: Path) -> None:
    """Generate the requested EDA plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(input_path)

    print(f"Total rows: {len(df):,}")
    print("Class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    plot_class_distribution(df, results_dir / "class_distribution.png")
    plot_text_length_distribution(df, results_dir / "text_length_distribution.png")
    missing = save_missing_values(df, results_dir / "missing_values.csv")
    source_distribution = save_source_distribution(
        df,
        results_dir / "dataset_source_distribution.csv",
    )
    duplicate_stats = save_duplicate_statistics(df, results_dir / "duplicate_statistics.csv")
    save_top_terms_by_class(df, results_dir / "top_terms_by_class.csv")

    print("Missing values:")
    print(missing.to_string(index=False))

    if source_distribution is not None:
        print("Source distribution:")
        print(source_distribution.to_string(index=False))

    print("Duplicate statistics:")
    print(duplicate_stats.to_string(index=False))

    print(f"Saved {results_dir / 'class_distribution.png'}")
    print(f"Saved {results_dir / 'text_length_distribution.png'}")
    print(f"Saved {results_dir / 'missing_values.csv'}")
    print(f"Saved {results_dir / 'duplicate_statistics.csv'}")
    print(f"Saved {results_dir / 'top_terms_by_class.csv'}")
    if source_distribution is not None:
        print(f"Saved {results_dir / 'dataset_source_distribution.csv'}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run simple EDA on standardized data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the EDA script."""
    args = parse_args()
    run_eda(args.input, args.results_dir)


if __name__ == "__main__":
    main()
