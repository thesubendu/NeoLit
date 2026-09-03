"""
Query Planner Agent.

Turns a natural-language research question into several diverse search
queries covering the different facets of neoantigen biology (origin,
presentation, immunogenicity, prediction, validation, ...), instead of
searching the literal question string. Can also be re-invoked with a
"gap_note" to target a coverage gap found by the pipeline's feedback loop.
"""
from __future__ import annotations

import json
from typing import Optional

from llm_client import GeminiClient
from utils import tokenize

_CATEGORIES = [
    "core terminology (neoantigen, neoepitope, tumor antigen)",
    "mutation origin (somatic mutation, SNV, indel, frameshift, fusion)",
    "presentation (MHC-I, MHC-II, HLA binding, antigen processing)",
    "immunogenicity (T-cell response, immunogenicity, T-cell recognition)",
    "prediction tools (NetMHCpan, MHCflurry, neoantigen prediction pipelines)",
    "quality / confidence (high-confidence neoantigen, prioritization criteria)",
    "biological properties (expression, clonality, tumor specificity, stability)",
    "validation methods (immunopeptidomics, mass spectrometry, T-cell assays)",
]

_FALLBACK_TEMPLATES = [
    "neoantigen {topic}",
    "neoepitope {topic}",
    "tumor neoantigen prediction {topic}",
    "neoantigen immunogenicity HLA binding {topic}",
    "neoantigen expression clonality {topic}",
    "high confidence neoantigen validation {topic}",
]

SYSTEM_INSTRUCTION = (
    "You are a biomedical literature search strategist specializing in cancer "
    "immunology and neoantigen research. You design precise, diverse search "
    "queries suitable for PubMed and Semantic Scholar. Respond with JSON only."
)


class QueryPlannerAgent:
    def __init__(self, llm: Optional[GeminiClient] = None):
        self.llm = llm

    def plan(
        self,
        research_question: str,
        prior_queries: Optional[list[str]] = None,
        gap_note: Optional[str] = None,
        n_queries: int = 6,
    ) -> list[str]:
        if self.llm is not None:
            try:
                return self._plan_with_llm(research_question, prior_queries, gap_note, n_queries)
            except Exception:
                pass  # fall through to deterministic fallback
        return self._plan_fallback(research_question, n_queries)

    # ------------------------------------------------------------------
    def _plan_with_llm(
        self,
        research_question: str,
        prior_queries: Optional[list[str]],
        gap_note: Optional[str],
        n_queries: int,
    ) -> list[str]:
        categories = "\n".join(f"- {c}" for c in _CATEGORIES)
        prior_block = ""
        if prior_queries:
            prior_block = (
                "\nQueries already tried (do not repeat these, propose different angles):\n"
                + "\n".join(f"- {q}" for q in prior_queries)
            )
        gap_block = f"\nCoverage gap to address: {gap_note}\n" if gap_note else ""

        prompt = f"""
Research question: "{research_question}"

Generate {n_queries} distinct, high-precision search-engine queries (for PubMed
and Semantic Scholar) that together cover the different facets below relevant
to this question. Do not just restate the question. Prefer concrete
biomedical terminology over generic phrasing. Do not include boolean
operators (AND/OR) or quotation marks — plain keyword strings only.

Facets to draw on:
{categories}
{gap_block}{prior_block}

Return ONLY a JSON array of {n_queries} strings, e.g.:
["neoantigen HLA binding affinity glioblastoma", "..."]
""".strip()

        raw = self.llm.generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
        if isinstance(raw, dict) and "queries" in raw:
            raw = raw["queries"]
        queries = [str(q).strip() for q in raw if str(q).strip()]
        if not queries:
            raise ValueError("empty query list from LLM")
        return queries[:n_queries]

    # ------------------------------------------------------------------
    def _plan_fallback(self, research_question: str, n_queries: int) -> list[str]:
        """Deterministic template expansion used if the LLM call fails or no key is set."""
        topic_words = tokenize(research_question)[:4]
        topic = " ".join(topic_words) if topic_words else "cancer"
        queries = [t.format(topic=topic) for t in _FALLBACK_TEMPLATES]
        return queries[:n_queries]
