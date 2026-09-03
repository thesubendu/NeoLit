"""
Shared data structures passed between agents.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Paper:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    s2_paper_id: str = ""
    url: str = ""
    citation_count: Optional[int] = None
    sources: set[str] = field(default_factory=set)  # e.g. {"PubMed", "SemanticScholar"}

    # filled in by later agents
    relevance_score: float = 0.0
    relevance_reason: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.strip().lower()}"
        if self.pmid:
            return f"pmid:{self.pmid.strip()}"
        return f"title:{_normalize_title(self.title)}"

    def short_citation(self) -> str:
        first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        suffix = " et al." if len(self.authors) > 1 else ""
        year = self.year or "n.d."
        return f"{first_author}{suffix} ({year})"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["sources"] = sorted(self.sources)
        return d


def _normalize_title(title: str) -> str:
    import re

    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass
class RunLog:
    """Human-readable trace of what the pipeline did, for transparency/debugging."""

    events: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.events.append(message)

    def dump(self) -> str:
        return "\n".join(self.events)
