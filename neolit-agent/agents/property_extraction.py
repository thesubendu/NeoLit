"""
Neoantigen Property Extraction Agent.

For each relevant paper, converts the unstructured title/abstract into the
structured neoantigen taxonomy the project cares about: publication info,
neoantigen characteristics, biological properties, immunological properties,
and prediction/validation evidence. Grounded strictly in what the abstract
actually states — the prompt explicitly forbids inferring unstated facts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import settings
from llm_client import GeminiClient
from models import Paper
from utils import truncate

SYSTEM_INSTRUCTION = (
    "You are a cancer immunology research assistant. You extract structured "
    "facts about tumor neoantigens strictly from the text provided. Never "
    "invent or infer values that are not stated or clearly implied in the "
    "abstract. Respond with JSON only."
)

_SCHEMA_FIELDS = """
{
  "mutation_type": "e.g. SNV / indel / frameshift / fusion / gene fusion / not reported",
  "hla_allele": "specific HLA allele(s) mentioned, or not reported",
  "source_protein": "gene/protein of origin, or not reported",
  "tumor_specificity": "evidence the antigen is tumor-specific vs shared, or not reported",
  "gene_expression": "expression evidence (e.g. RNA-seq support), or not reported",
  "clonality": "clonal vs subclonal evidence, or not reported",
  "mhc_binding_affinity": "binding affinity findings/thresholds, or not reported",
  "mhc_presentation": "evidence of MHC presentation (e.g. immunopeptidomics), or not reported",
  "tcell_recognition": "evidence of T-cell recognition, or not reported",
  "immunogenicity": "summary of immunogenicity findings, or not reported",
  "prediction_tool": "computational tool(s) used (e.g. NetMHCpan), or not reported",
  "experimental_validation": "validation method used (e.g. ELISPOT, tetramer, mass spec), or not reported",
  "cancer_type": "cancer type(s) studied, or not reported",
  "evidence_level": "one of: Experimental, Computational, Both, Not reported",
  "key_properties_summary": "1-2 sentence summary, in your own words, of the neoantigen properties this paper supports"
}
""".strip()


class PropertyExtractionAgent:
    def __init__(self, llm: Optional[GeminiClient] = None, workers: Optional[int] = None):
        self.llm = llm
        self.workers = workers or settings.extraction_workers

    def extract_batch(self, papers: list[Paper], research_question: str) -> None:
        """Populates paper.properties in place, using a small thread pool."""
        if not self.llm or not papers:
            for p in papers:
                p.properties = {"evidence_level": "Not extracted (no LLM configured)"}
            return

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._extract_one, p, research_question): p for p in papers
            }
            for fut in as_completed(futures):
                paper = futures[fut]
                try:
                    paper.properties = fut.result()
                except Exception as e:  # noqa: BLE001
                    paper.properties = {
                        "evidence_level": "Extraction failed",
                        "key_properties_summary": f"(extraction error: {e})",
                    }

    # ------------------------------------------------------------------
    def _extract_one(self, paper: Paper, research_question: str) -> dict:
        prompt = f"""
Research question the review is answering: "{research_question}"

Paper title: {paper.title}
Journal / year: {paper.journal or "unknown"} / {paper.year or "unknown"}
Abstract:
\"\"\"{truncate(paper.abstract, 3000) or "(no abstract available)"}\"\"\"

Extract the following fields strictly from the abstract above. If a field is
not addressed, use the string "not reported" (or "not reported" inside the
relevant list). Do not copy long verbatim spans from the abstract — paraphrase.

Return ONLY a JSON object with exactly these keys:
{_SCHEMA_FIELDS}
""".strip()
        return self.llm.generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
