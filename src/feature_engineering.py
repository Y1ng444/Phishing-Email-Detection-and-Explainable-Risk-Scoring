"""Metadata feature extraction for phishing email detection.

The model still uses TF-IDF for email text, but these numeric features capture
security signals that SOC analysts often inspect manually: links, HTML markup,
raw IP URLs, unusual punctuation, uppercase wording, and sender format.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


METADATA_FEATURES = [
    "n_links",
    "has_ip_url",
    "html_tag_count",
    "html_ratio",
    "body_len",
    "subject_len",
    "n_exclamation",
    "n_upper_words",
    "sender_has_email",
    "url_count_from_column",
]

MODEL_INPUT_COLUMNS = ["text_combined", *METADATA_FEATURES]

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>'\")]+",
    flags=re.IGNORECASE,
)
IP_URL_PATTERN = re.compile(
    r"(?:https?://|www\.)?(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?(?:/|$)",
    flags=re.IGNORECASE,
)
HTML_TAG_PATTERN = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
EMPTY_VALUES = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def _clean_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a string Series for a possibly missing standardized column."""
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    series = df[column].fillna("").astype(str)
    return series.mask(series.str.strip().str.lower().isin(EMPTY_VALUES), "")


def _count_urls_from_text(value: Any) -> int:
    """Count URL-like strings in free text."""
    text = "" if pd.isna(value) else str(value)
    return len(URL_PATTERN.findall(text))


def _count_urls_from_column(value: Any) -> int:
    """Count URLs stored in a dataset URL column.

    Many public email datasets store URLs as a stringified list, a comma-joined
    field, or a single URL. Regex hits are preferred; separator counting is a
    fallback for rows where URL text is redacted but the field still lists items.
    """
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if text.lower() in EMPTY_VALUES:
        return 0

    regex_count = _count_urls_from_text(text)
    if regex_count:
        return regex_count

    normalized = text.strip("[](){}")
    parts = [part.strip().strip("'\"") for part in re.split(r"[,;|]\s*", normalized)]
    return sum(1 for part in parts if part and part.lower() not in EMPTY_VALUES)


def _parse_email_text(email_text: Any) -> dict[str, str]:
    """Split a pasted email into lightweight standardized fields."""
    text = "" if email_text is None else str(email_text)
    subject = ""
    sender = ""
    body_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("subject:") and not subject:
            subject = stripped.split(":", 1)[1].strip()
        elif lower.startswith("from:") and not sender:
            sender = stripped.split(":", 1)[1].strip()
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip() or text
    urls = " ".join(URL_PATTERN.findall(text))
    text_combined = " ".join(part for part in [subject, body, urls] if part).strip()
    return {
        "subject": subject,
        "body": body,
        "urls": urls,
        "sender": sender,
        "text_combined": text_combined,
    }


def _as_dataframe(df_or_text: pd.DataFrame | str | Any) -> pd.DataFrame:
    """Normalize supported inputs to the standardized email-column shape."""
    if isinstance(df_or_text, pd.DataFrame):
        df = df_or_text.copy()
        for column in ["subject", "body", "urls", "sender"]:
            if column not in df.columns:
                df[column] = ""
        if "text_combined" not in df.columns:
            df["text_combined"] = (
                df[["subject", "body", "urls"]]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        return df
    return pd.DataFrame([_parse_email_text(df_or_text)])


def extract_metadata_features(df_or_text: pd.DataFrame | str | Any) -> pd.DataFrame:
    """Extract numeric metadata features from emails.

    Parameters
    ----------
    df_or_text:
        Either the standardized email DataFrame used during model training or a
        single pasted email string from the demo app.

    Returns
    -------
    pandas.DataFrame
        A numeric frame with columns listed in ``METADATA_FEATURES``. It is safe
        to feed this output into ``StandardScaler`` inside an sklearn pipeline.
    """
    df = _as_dataframe(df_or_text)
    subject = _clean_series(df, "subject")
    body = _clean_series(df, "body")
    urls = _clean_series(df, "urls")
    sender = _clean_series(df, "sender")

    if "text_combined" in df.columns:
        text_combined = _clean_series(df, "text_combined")
    else:
        text_combined = (subject + " " + body + " " + urls).str.strip()

    body_for_length = body.mask(body == "", text_combined)
    raw_text_for_counts = (subject + " " + body_for_length).str.strip()
    combined_for_security = (raw_text_for_counts + " " + urls).str.strip()

    url_count_from_column = urls.map(_count_urls_from_column)
    url_count_from_text = raw_text_for_counts.map(_count_urls_from_text)
    n_links = pd.concat([url_count_from_text, url_count_from_column], axis=1).max(axis=1)

    html_tag_count = raw_text_for_counts.map(lambda value: len(HTML_TAG_PATTERN.findall(value)))
    html_tag_chars = raw_text_for_counts.map(
        lambda value: sum(len(match.group(0)) for match in HTML_TAG_PATTERN.finditer(value))
    )
    combined_len = raw_text_for_counts.str.len().clip(lower=1)

    features = pd.DataFrame(index=df.index)
    features["n_links"] = n_links.astype(float)
    features["has_ip_url"] = combined_for_security.map(
        lambda value: 1.0 if IP_URL_PATTERN.search(value) else 0.0
    )
    features["html_tag_count"] = html_tag_count.astype(float)
    features["html_ratio"] = (html_tag_chars / combined_len).astype(float)
    features["body_len"] = body_for_length.str.len().astype(float)
    features["subject_len"] = subject.str.len().astype(float)
    features["n_exclamation"] = raw_text_for_counts.str.count("!").astype(float)
    features["n_upper_words"] = raw_text_for_counts.map(
        lambda value: float(len(UPPER_WORD_PATTERN.findall(value)))
    )
    features["sender_has_email"] = sender.map(
        lambda value: 1.0 if EMAIL_PATTERN.search(value) else 0.0
    )
    features["url_count_from_column"] = url_count_from_column.astype(float)

    return features[METADATA_FEATURES].fillna(0.0)


def build_model_input(df_or_text: pd.DataFrame | str | Any) -> pd.DataFrame:
    """Return the exact feature frame expected by text+metadata pipelines."""
    df = _as_dataframe(df_or_text)
    metadata = extract_metadata_features(df)
    text = _clean_series(df, "text_combined").rename("text_combined")
    model_input = pd.concat(
        [text.reset_index(drop=True), metadata.reset_index(drop=True)],
        axis=1,
    )
    return model_input[MODEL_INPUT_COLUMNS].fillna(0.0)
