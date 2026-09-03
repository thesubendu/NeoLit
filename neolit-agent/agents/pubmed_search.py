"""
PubMed Search Agent — retrieves papers via NCBI E-utilities (esearch + efetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
No API key is required, but supplying NCBI_EMAIL (and optionally
NCBI_API_KEY) is good practice and raises your rate limit from 3 to 10
requests/second.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from config import settings
from models import Paper

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSearchAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        tool: Optional[str] = None,
        timeout: int = 20,
    ):
        self.api_key = api_key or settings.ncbi_api_key
        self.email = email or settings.ncbi_email
        self.tool = tool or settings.ncbi_tool_name
        self.timeout = timeout

    def _common_params(self) -> dict:
        params = {"tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, max_results: int = 15) -> list[Paper]:
        pmids = self._esearch(query, max_results)
        if not pmids:
            return []
        return self._efetch(pmids)

    # ------------------------------------------------------------------
    def _esearch(self, query: str, max_results: int) -> list[str]:
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
        try:
            r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("esearchresult", {}).get("idlist", [])
        except requests.RequestException:
            return []

    def _efetch(self, pmids: list[str]) -> list[Paper]:
        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        try:
            r = requests.get(f"{BASE}/efetch.fcgi", params=params, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException:
            return []

        papers: list[Paper] = []
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            return []

        for article in root.iter("PubmedArticle"):
            papers.append(self._parse_article(article))
        return papers

    @staticmethod
    def _parse_article(article: ET.Element) -> Paper:
        def text(el: Optional[ET.Element]) -> str:
            return (el.text or "").strip() if el is not None else ""

        pmid = text(article.find(".//MedlineCitation/PMID"))
        title = text(article.find(".//ArticleTitle"))

        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            fragment = (ab.text or "").strip()
            if not fragment:
                continue
            abstract_parts.append(f"{label}: {fragment}" if label else fragment)
        abstract = " ".join(abstract_parts)

        journal = text(article.find(".//Journal/Title"))

        year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
        year = None
        if year_el is not None and year_el.text and year_el.text.strip().isdigit():
            year = int(year_el.text.strip())
        else:
            medline_date = text(article.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
            digits = "".join(c for c in medline_date[:4] if c.isdigit())
            if len(digits) == 4:
                year = int(digits)

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = text(author.find("LastName"))
            initials = text(author.find("Initials"))
            collective = text(author.find("CollectiveName"))
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective:
                authors.append(collective)

        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()

        return Paper(
            title=title or "(untitled)",
            abstract=abstract,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            pmid=pmid,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            sources={"PubMed"},
        )
