"""
Research Report Agent.

Synthesizes the ranked, property-extracted papers into a structured
markdown literature review. The reference list is built deterministically
in Python (from paper metadata) so it can never be hallucinated; only the
narrative synthesis is delegated to the LLM, and it is instructed to cite
using the fixed [n] numbering supplied to it.
"""
from __future__ import annotations

from typing import Optional

from llm_client import GeminiClient
from models import Paper

SYSTEM_INSTRUCTION = (
    "You are a scientific writer producing a rigorous, evidence-grounded "
    "literature review on tumor neoantigens for a computational biology "
    "researcher. Every claim you make must be traceable to the numbered "
    "source data provided — cite sources inline as [n]. Never invent "
    "findings, papers, or citation numbers that are not in the provided data."
)


class ReportGeneratorAgent:
    def __init__(self, llm: Optional[GeminiClient] = None):
        self.llm = llm

    def generate(self, research_question: str, papers: list[Paper]) -> str:
        if not papers:
            return (
                f"# Literature Report: {research_question}\n\n"
                "No sufficiently relevant papers were found. Try broadening "
                "the research question or increasing --max-per-source."
            )

        numbered = list(enumerate(papers, start=1))
        source_block = self._build_source_block(numbered)
        references = self._build_reference_list(numbered)

        if self.llm is None:
            body = self._fallback_body(numbered)
        else:
            try:
                body = self._generate_with_llm(research_question, source_block)
            except Exception as e:  # noqa: BLE001
                body = self._fallback_body(numbered)
                body += f"\n\n> _Note: LLM synthesis failed ({e}); showing extracted data only._\n"

        return f"# Literature Report: {research_question}\n\n{body}\n\n## References\n\n{references}\n"

    # ------------------------------------------------------------------
    @staticmethod
    def _build_source_block(numbered: list[tuple[int, Paper]]) -> str:
        lines = []
        for n, p in numbered:
            props = p.properties or {}
            lines.append(
                f"[{n}] {p.title} ({p.year or 'n.d.'}) — relevance {p.relevance_score}/10\n"
                f"    cancer_type: {props.get('cancer_type', 'not reported')}\n"
                f"    mutation_type: {props.get('mutation_type', 'not reported')}\n"
                f"    hla_allele: {props.get('hla_allele', 'not reported')}\n"
                f"    mhc_binding_affinity: {props.get('mhc_binding_affinity', 'not reported')}\n"
                f"    mhc_presentation: {props.get('mhc_presentation', 'not reported')}\n"
                f"    tcell_recognition: {props.get('tcell_recognition', 'not reported')}\n"
                f"    immunogenicity: {props.get('immunogenicity', 'not reported')}\n"
                f"    gene_expression: {props.get('gene_expression', 'not reported')}\n"
                f"    clonality: {props.get('clonality', 'not reported')}\n"
                f"    tumor_specificity: {props.get('tumor_specificity', 'not reported')}\n"
                f"    prediction_tool: {props.get('prediction_tool', 'not reported')}\n"
                f"    experimental_validation: {props.get('experimental_validation', 'not reported')}\n"
                f"    evidence_level: {props.get('evidence_level', 'not reported')}\n"
                f"    summary: {props.get('key_properties_summary', 'not reported')}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_reference_list(numbered: list[tuple[int, Paper]]) -> str:
        lines = []
        for n, p in numbered:
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            link = p.url or (f"https://doi.org/{p.doi}" if p.doi else "")
            lines.append(
                f"{n}. {authors or 'Unknown authors'}. **{p.title}**. "
                f"{p.journal or 'Journal not recorded'}, {p.year or 'n.d.'}. "
                f"{f'[Link]({link})' if link else ''}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _generate_with_llm(self, research_question: str, source_block: str) -> str:
        prompt = f"""
Research question: "{research_question}"

You have {source_block.count('[')} numbered source papers below, each with
extracted neoantigen properties. Write a structured markdown literature
report with these sections (use these exact headers, as ## headers):

## 1. Definition of Neoantigen
## 2. Sources of Neoantigens
## 3. Neoantigen Properties
(group findings by: HLA/MHC binding, presentation, expression, clonality,
tumor specificity, immunogenicity/T-cell recognition — cite [n] for each claim)
## 4. Prediction Methods Reported
## 5. Experimental Validation Approaches Reported
## 6. Cancer-Specific Evidence
## 7. Proposed High-Confidence Neoantigen Criteria
(synthesize across sources; be explicit these are literature-derived, not your own opinion)
## 8. Research Gaps
(what the retrieved set does NOT cover well)
## 9. Key Papers At a Glance
(a short markdown table: # | Paper | Cancer type | Key property | Evidence level)

Rules:
- Cite every substantive claim with [n] matching the source numbers below.
- If evidence conflicts between sources, say so explicitly.
- Do not cite a source number that is not listed below.
- Do not add a References section yourself — it will be appended separately.

Numbered sources:
{source_block}
""".strip()
        return self.llm.generate_text(prompt, system_instruction=SYSTEM_INSTRUCTION, temperature=0.4)

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_body(numbered: list[tuple[int, Paper]]) -> str:
        """Used when no LLM is configured — dumps extracted data as a table."""
        rows = ["| # | Paper | Cancer type | Evidence level | Summary |", "|---|---|---|---|---|"]
        for n, p in numbered:
            props = p.properties or {}
            rows.append(
                f"| {n} | {p.title} | {props.get('cancer_type', 'n/a')} | "
                f"{props.get('evidence_level', 'n/a')} | {props.get('key_properties_summary', 'n/a')} |"
            )
        return "## Extracted Data (no LLM synthesis configured)\n\n" + "\n".join(rows)
