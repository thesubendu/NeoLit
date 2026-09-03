"""
Small text-processing helpers shared across agents. Kept dependency-free
(no nltk/spacy) so the project installs quickly.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with",
    "is", "are", "properties", "property", "literature", "find", "recent",
    "papers", "paper", "about", "what", "which", "that", "high", "confidence",
    "research", "study", "studies", "review", "describe", "describing", "does",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def keyword_overlap_score(query: str, *fields: str) -> float:
    """Cheap fallback relevance score (0-10) when the LLM ranker is unavailable."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    text_tokens = set()
    for f in fields:
        text_tokens.update(tokenize(f or ""))
    if not text_tokens:
        return 0.0
    overlap = len(q_tokens & text_tokens)
    return round(10 * overlap / max(len(q_tokens), 1), 2)


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
