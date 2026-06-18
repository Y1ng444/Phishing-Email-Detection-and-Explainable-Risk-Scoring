"""Exploratory data analysis for the standardized phishing email dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("data/processed/phishing_email_standardized.csv")
DEFAULT_RESULTS_DIR = Path("results")


def read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    """Read a CSV with common encoding fallbacks."""
    last_error: Exception | None = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{path} is empty") from exc
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not read {path}: {last_error}")


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load and lightly validate the standardized dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Standardized dataset not found at {input_path}. "
            "Run standardize_datasets.py first."
        )

    df = read_csv_with_fallbacks(input_path)
    required = {"text_combined", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text_combined"] = df["text_combined"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a simple long-form EDA summary table."""
    rows: list[dict[str, object]] = []

    rows.append({"section": "dataset_shape", "item": "rows", "value": len(df)})
    rows.append({"section": "dataset_shape", "item": "columns", "value": len(df.columns)})

    for column in df.columns:
        rows.append({"section": "schema", "item": column, "value": str(df[column].dtype)})

    for column, count in df.isna().sum().items():
        rows.append({"section": "missing_values", "item": column, "value": int(count)})

    label_counts = df["label"].value_counts(dropna=False).sort_index()
    for label, count in label_counts.items():
        rows.append({"section": "class_distribution", "item": str(label), "value": int(count)})

    text_lengths = df["text_combined"].str.len()
    word_counts = df["text_combined"].str.split().str.len()
    stats = {
        "text_length_min": text_lengths.min(),
        "text_length_mean": text_lengths.mean(),
        "text_length_median": text_lengths.median(),
        "text_length_max": text_lengths.max(),
        "word_count_min": word_counts.min(),
        "word_count_mean": word_counts.mean(),
        "word_count_median": word_counts.median(),
        "word_count_max": word_counts.max(),
    }
    for item, value in stats.items():
        rows.append({"section": "text_statistics", "item": item, "value": round(float(value), 3)})

    return pd.DataFrame(rows)


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Save a class distribution bar chart."""
    counts = df["label"].value_counts().sort_index()
    labels = ["Ham" if label == 0 else "Phishing" if label == 1 else str(label) for label in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, counts.values, color=["#4c78a8", "#f58518"][: len(labels)])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_text_length_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Save a text length histogram by class."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, name, color in [(0, "Ham", "#4c78a8"), (1, "Phishing", "#f58518")]:
        values = df.loc[df["label"] == label, "text_combined"].str.len()
        if values.empty:
            continue
        ax.hist(values.clip(upper=values.quantile(0.99)), bins=50, alpha=0.6, label=name, color=color)

    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Characters, clipped at each class 99th percentile")
    ax.set_ylabel("Rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_eda(input_path: Path, results_dir: Path) -> None:
    """Run EDA and save summary artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading standardized dataset: {input_path}")
    df = load_dataset(input_path)
    print(f"Dataset shape: {df.shape}")
    print("Schema:")
    print(df.dtypes.to_string())
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    summary = build_summary(df)
    summary_path = results_dir / "eda_summary.csv"
    summary.to_csv(summary_path, index=False)

    class_plot_path = results_dir / "class_distribution.png"
    text_plot_path = results_dir / "text_length_distribution.png"
    plot_class_distribution(df, class_plot_path)
    plot_text_length_distribution(df, text_plot_path)

    print(f"Saved EDA summary to {summary_path}")
    print(f"Saved class distribution plot to {class_plot_path}")
    print(f"Saved text length plot to {text_plot_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run EDA on the standardized phishing dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_eda(args.input, args.results_dir)


if __name__ == "__main__":
    main()
