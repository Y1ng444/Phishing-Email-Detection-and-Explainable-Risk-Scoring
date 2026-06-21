"""Regex-based email text cleaning for the phishing classifier."""

from __future__ import annotations

import html
import re


HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
SPECIAL_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Clean email text using simple, course-aligned preprocessing."""
    if text is None:
        return ""

    cleaned = html.unescape(str(text)).lower()
    cleaned = HTML_PATTERN.sub(" ", cleaned)
    cleaned = URL_PATTERN.sub(" url ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" email ", cleaned)
    cleaned = NUMBER_PATTERN.sub(" num ", cleaned)
    cleaned = SPECIAL_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def count_urls(text: object) -> int:
    """Count URL-like strings in raw email text."""
    if text is None:
        return 0
    return len(URL_PATTERN.findall(str(text)))


# Compatibility alias for older imports in notebooks or local experiments.
clean_email_text = clean_text
