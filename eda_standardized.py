"""Simple EDA for the standardized phishing email dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def run_eda(input_path: Path, results_dir: Path) -> None:
    """Generate the requested EDA plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(input_path)

    print(f"Rows: {len(df):,}")
    print("Class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    plot_class_distribution(df, results_dir / "class_distribution.png")
    plot_text_length_distribution(df, results_dir / "text_length_distribution.png")

    print(f"Saved {results_dir / 'class_distribution.png'}")
    print(f"Saved {results_dir / 'text_length_distribution.png'}")


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
