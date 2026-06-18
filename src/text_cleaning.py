"""Simple text cleaning helpers for phishing email classification.

The optimized pipeline intentionally uses lightweight, course-aligned NLP:
lowercasing, pattern replacement, punctuation cleanup, and TF-IDF tokenization.
No NLTK downloads are required.
"""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_WORD_PATTERN = re.compile(r"[^a-zA-Z_]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_email_text(text: object) -> str:
    """Return a normalized text string suitable for TF-IDF."""
    if text is None:
        return ""

    text = str(text).lower()
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def count_urls(text: object) -> int:
    """Count URL-like strings in an email."""
    if text is None:
        return 0

    return len(URL_PATTERN.findall(str(text)))
