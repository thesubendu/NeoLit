"""
Deduplication + Relevance Ranking Agent.

Dedup: merges papers found by multiple sources/queries (matched by DOI,
then PMID, then normalized title), keeping the richest abstract available.

Ranking: asks Gemini to score each paper's relevance to the research
question on a 0-10 scale, in batches (cheaper and more consistent than
one call per paper). Falls back to a keyword-overlap heuristic if the LLM
is unavailable or a batch fails to parse.
"""
from __future__ import annotations

from typing import Optional

from llm_client import GeminiClient
from models import Paper
from utils import keyword_overlap_score, truncate

SYSTEM_INSTRUCTION = (
    "You are a meticulous biomedical literature reviewer. You judge how "
    "relevant a paper is to a research question based only on its title and "
    "abstract. Respond with JSON only."
)

_BATCH_SIZE = 12


class DedupRankAgent:
    def __init__(self, llm: Optional[GeminiClient] = None):
        self.llm = llm

    # ------------------------------------------------------------------
    def dedup(self, papers: list[Paper]) -> list[Paper]:
        merged: dict[str, Paper] = {}
        for p in papers:
            key = p.dedup_key()
            if key not in merged:
                merged[key] = p
                continue
            existing = merged[key]
            existing.sources |= p.sources
            if len(p.abstract) > len(existing.abstract):
                existing.abstract = p.abstract
            existing.doi = existing.doi or p.doi
            existing.pmid = existing.pmid or p.pmid
            existing.s2_paper_id = existing.s2_paper_id or p.s2_paper_id
            existing.citation_count = existing.citation_count or p.citation_count
            existing.url = existing.url or p.url
        return list(merged.values())

    # ------------------------------------------------------------------
    def rank(self, papers: list[Paper], research_question: str) -> list[Paper]:
        if not papers:
            return []
        if self.llm is not None:
            try:
                self._rank_with_llm(papers, research_question)
            except Exception:
                self._rank_with_heuristic(papers, research_question)
        else:
            self._rank_with_heuristic(papers, research_question)

        return sorted(papers, key=lambda p: p.relevance_score, reverse=True)

    # ------------------------------------------------------------------
    def _rank_with_llm(self, papers: list[Paper], research_question: str) -> None:
        for start in range(0, len(papers), _BATCH_SIZE):
            batch = papers[start : start + _BATCH_SIZE]
            items = [
                {
                    "index": i,
                    "title": p.title,
                    "abstract": truncate(p.abstract, 600),
                }
                for i, p in enumerate(batch)
            ]
            prompt = f"""
Research question: "{research_question}"

Score how relevant each paper below is to answering this research question,
on a 0-10 scale (10 = directly and substantively relevant, 0 = unrelated).
Consider whether the paper actually discusses neoantigen properties
(binding, expression, immunogenicity, clonality, validation, etc.), not just
whether it mentions cancer in general.

Papers:
{items}

Return ONLY a JSON array, one object per paper, in this exact form:
[{{"index": 0, "score": 7.5, "reason": "<12 words max"}}, ...]
""".strip()
            result = self.llm.generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
            if isinstance(result, dict) and "scores" in result:
                result = result["scores"]
            for entry in result:
                idx = int(entry["index"])
                if 0 <= idx < len(batch):
                    batch[idx].relevance_score = float(entry.get("score", 0))
                    batch[idx].relevance_reason = str(entry.get("reason", ""))[:120]

    # ------------------------------------------------------------------
    @staticmethod
    def _rank_with_heuristic(papers: list[Paper], research_question: str) -> None:
        for p in papers:
            p.relevance_score = keyword_overlap_score(research_question, p.title, p.abstract)
            p.relevance_reason = "keyword-overlap fallback score (LLM unavailable)"
