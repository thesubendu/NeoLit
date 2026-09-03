"""
Semantic Scholar Search Agent — retrieves papers via the Semantic Scholar
Graph API. Docs: https://api.semanticscholar.org/api-docs/

Works without an API key (shared low-rate pool); set SEMANTIC_SCHOLAR_API_KEY
for a higher personal rate limit.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from config import settings
from models import Paper

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,authors,venue,externalIds,url,citationCount"


class SemanticScholarSearchAgent:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or settings.s2_api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 15) -> list[Paper]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {"query": query, "limit": max_results, "fields": FIELDS}

        for attempt in range(3):
            try:
                r = requests.get(
                    f"{BASE}/paper/search",
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json().get("data", [])
                return [self._parse(item) for item in data]
            except requests.RequestException:
                time.sleep(1 * (attempt + 1))
        return []

    @staticmethod
    def _parse(item: dict) -> Paper:
        external = item.get("externalIds") or {}
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        return Paper(
            title=item.get("title") or "(untitled)",
            abstract=item.get("abstract") or "",
            authors=authors,
            year=item.get("year"),
            journal=item.get("venue") or "",
            doi=(external.get("DOI") or "").strip(),
            pmid=str(external.get("PubMed") or "").strip(),
            s2_paper_id=item.get("paperId") or "",
            url=item.get("url") or "",
            citation_count=item.get("citationCount"),
            sources={"SemanticScholar"},
        )
