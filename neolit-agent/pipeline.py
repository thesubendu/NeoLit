"""
NeoLitPipeline — orchestrates the agent chain:

  QueryPlanner -> [PubMed + Semantic Scholar] -> Dedup -> Rank
      -> (feedback loop: enough relevant papers? if not, expand & re-search)
      -> PropertyExtraction (top N)
      -> ReportGenerator

This is the MVP scope (PubMed + Semantic Scholar). Europe PMC, an explicit
Evidence Verification agent, and LangGraph-based control flow are natural
Phase 2 extensions — see README.md.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agents import (
    DedupRankAgent,
    PropertyExtractionAgent,
    PubMedSearchAgent,
    QueryPlannerAgent,
    ReportGeneratorAgent,
    SemanticScholarSearchAgent,
)
from config import settings
from llm_client import GeminiClient
from models import Paper, RunLog


@dataclass
class PipelineResult:
    report_markdown: str
    ranked_papers: list[Paper]
    run_log: RunLog


class NeoLitPipeline:
    def __init__(self, use_llm: bool = True):
        self.llm = GeminiClient() if use_llm else None
        self.query_planner = QueryPlannerAgent(self.llm)
        self.pubmed = PubMedSearchAgent()
        self.s2 = SemanticScholarSearchAgent()
        self.dedup_rank = DedupRankAgent(self.llm)
        self.extractor = PropertyExtractionAgent(self.llm)
        self.reporter = ReportGeneratorAgent(self.llm)

    def run(
        self,
        research_question: str,
        max_papers_per_source: int | None = None,
        top_n_extract: int | None = None,
        min_relevant: int | None = None,
        max_expansion_rounds: int | None = None,
        on_event=print,
    ) -> PipelineResult:
        max_papers_per_source = max_papers_per_source or settings.max_papers_per_source
        top_n_extract = top_n_extract or settings.top_n_for_extraction
        min_relevant = min_relevant or settings.min_relevant_papers
        max_expansion_rounds = max_expansion_rounds or settings.max_query_expansion_rounds

        log = RunLog()

        def emit(msg: str):
            log.add(msg)
            if on_event:
                on_event(msg)

        queries = self.query_planner.plan(research_question)
        emit(f"[QueryPlanner] Generated {len(queries)} search queries: {queries}")

        all_papers: list[Paper] = []
        tried_queries: list[str] = []
        ranked: list[Paper] = []

        for round_idx in range(1, max_expansion_rounds + 1):
            new_papers = self._search_all(queries, max_papers_per_source, emit)
            all_papers.extend(new_papers)
            tried_queries.extend(queries)

            deduped = self.dedup_rank.dedup(all_papers)
            emit(f"[Dedup] {len(all_papers)} raw results -> {len(deduped)} unique papers")

            ranked = self.dedup_rank.rank(deduped, research_question)
            relevant = [p for p in ranked if p.relevance_score >= settings.relevance_score_floor]
            emit(
                f"[Rank] Round {round_idx}: {len(relevant)} papers scored >= "
                f"{settings.relevance_score_floor}/10"
            )

            if len(relevant) >= min_relevant or round_idx == max_expansion_rounds:
                break

            emit(
                f"[QueryPlanner] Only {len(relevant)} relevant papers found — "
                "expanding search with new query angles"
            )
            queries = self.query_planner.plan(
                research_question,
                prior_queries=tried_queries,
                gap_note=f"only {len(relevant)} clearly relevant papers found so far",
            )

        top_papers = ranked[:top_n_extract]
        emit(f"[Extraction] Extracting neoantigen properties from top {len(top_papers)} papers")
        self.extractor.extract_batch(top_papers, research_question)

        emit("[Report] Synthesizing final literature report")
        report = self.reporter.generate(research_question, top_papers)

        emit("[Done] Pipeline complete")
        return PipelineResult(report_markdown=report, ranked_papers=top_papers, run_log=log)

    # ------------------------------------------------------------------
    def _search_all(self, queries: list[str], max_per_source: int, emit) -> list[Paper]:
        results: list[Paper] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            pubmed_futures = {pool.submit(self.pubmed.search, q, max_per_source): q for q in queries}
            s2_futures = {pool.submit(self.s2.search, q, max_per_source): q for q in queries}

            for fut, q in pubmed_futures.items():
                papers = fut.result()
                emit(f"[PubMed] '{q}' -> {len(papers)} papers")
                results.extend(papers)

            for fut, q in s2_futures.items():
                papers = fut.result()
                emit(f"[SemanticScholar] '{q}' -> {len(papers)} papers")
                results.extend(papers)
        return results
