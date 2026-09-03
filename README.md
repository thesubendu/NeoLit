# NeoLit-Agent

An agentic literature retrieval and neoantigen property extraction system.
Instead of a single chatbot call, a research question flows through a chain
of specialized agents that search, deduplicate, rank, extract structured
biology from, and finally synthesize a cited literature report — with a
feedback loop that automatically expands the search if coverage is thin.

This is the **MVP** scope: PubMed + Semantic Scholar retrieval, Gemini for
reasoning steps, sequential/threaded orchestration (no LangGraph yet). See
[Roadmap](#roadmap) for what Phase 2 adds.

## Pipeline

```
research question
      │
      ▼
Query Planner Agent  ──────────► generates 6+ diverse search queries
      │                          (HLA binding, immunogenicity, prediction,
      │                           validation, cancer type, ...)
      ▼
┌─────────────┬──────────────────┐
│  PubMed      │ Semantic Scholar │   (run concurrently, per query)
└─────────────┴──────────────────┘
      │
      ▼
Dedup Agent  ───────────────────► merges by DOI / PMID / normalized title
      │
      ▼
Ranking Agent ──────────────────► Gemini scores 0-10 relevance per paper
      │
      ▼
 enough relevant papers? ──NO──► Query Planner expands search (max N rounds)
      │ YES
      ▼
Property Extraction Agent  ─────► structured JSON per paper:
      │                           mutation type, HLA allele, MHC binding,
      │                           presentation, immunogenicity, clonality,
      │                           expression, prediction tool, validation...
      ▼
Report Generator Agent ─────────► cited markdown literature report
      │                           (reference list built deterministically —
      ▼                            never hallucinated)
  report.md
```

## Setup

```bash
cd neolit-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GEMINI_API_KEY
# (free key: https://aistudio.google.com/apikey)
```

`NCBI_EMAIL` and `NCBI_API_KEY` are optional but recommended — PubMed asks
for an email with E-utilities requests and the API key raises your rate
limit. `SEMANTIC_SCHOLAR_API_KEY` is optional too.

## Usage

### Command line

```bash
python main.py "Find recent literature on the properties of high-confidence neoantigens in glioblastoma"
```

Options:

```bash
python main.py "your research question" \
  --max-per-source 15 \      # papers fetched per query per source
  --top-n 12 \                # papers sent to property extraction
  --min-relevant 8 \          # stop expanding search once this many are relevant
  --max-rounds 2 \            # cap on query-expansion feedback loop
  --output report.md \
  --json-output data.json     # optional: dump raw structured data too
```

Run without an LLM key (search + heuristic ranking only, no extraction/report synthesis):

```bash
python main.py "your research question" --no-llm
```

### Web UI

```bash
streamlit run app.py
```

## Output

`report.md` follows this structure (see `agents/report_generator.py`):

1. Definition of neoantigen
2. Sources of neoantigens
3. Neoantigen properties (HLA/MHC binding, presentation, expression,
   clonality, tumor specificity, immunogenicity) — every claim cited `[n]`
4. Prediction methods reported in the literature
5. Experimental validation approaches reported
6. Cancer-specific evidence
7. Proposed high-confidence neoantigen criteria (synthesized across sources)
8. Research gaps in the retrieved set
9. Key papers at a glance (table)
10. References (built from paper metadata, not LLM-generated)

## Project layout

```
neolit-agent/
├── main.py                        CLI entry point
├── app.py                         Streamlit UI
├── pipeline.py                    Orchestrator + feedback loop
├── config.py                      Settings from env vars / .env
├── models.py                      Paper / RunLog dataclasses
├── llm_client.py                  Gemini wrapper (text + JSON mode, retries)
├── utils.py                       Tokenization / keyword-overlap fallback
└── agents/
    ├── query_planner.py           Expands question -> diverse queries
    ├── pubmed_search.py           NCBI E-utilities client
    ├── semantic_scholar_search.py Semantic Scholar Graph API client
    ├── dedup_rank.py              Merge duplicates + relevance scoring
    ├── property_extraction.py     Paper -> structured neoantigen JSON
    └── report_generator.py        Structured JSON -> cited markdown report
```

## Design notes

- **Grounding over fluency**: the property-extraction prompt explicitly
  forbids inferring unstated facts, and every field defaults to
  `"not reported"` rather than being guessed. The report generator is given
  a fixed, Python-built reference list and told never to cite a number that
  isn't in it — this is what keeps "each property linked to supporting
  papers" honest rather than LLM-hallucinated.
- **Graceful degradation**: every LLM call has a non-LLM fallback (keyword
  overlap for ranking, template expansion for query planning, a raw data
  table if report synthesis fails) so a flaky API call degrades the output
  quality rather than crashing the run.
- **Abstract-level extraction**: the MVP extracts from titles/abstracts
  only (no full-text PDF parsing), since PubMed/Semantic Scholar mostly
  return abstracts for free. This is usually sufficient for property-level
  synthesis but will miss detail that's only in the results section.

## Roadmap

Not built yet, but the code is structured so each of these is an additive
agent/module rather than a rewrite:

- **Europe PMC agent** — adds open-access full-text retrieval, useful when
  an abstract under-reports validation methods.
- **Evidence Verification agent** — a second pass that checks each claim in
  the draft report actually appears in the source's extracted properties
  before the report is finalized (catches LLM drift during synthesis).
- **LangGraph orchestration** — replace the linear `pipeline.py` loop with
  an explicit graph so the search/verify/expand cycle can branch more
  richly (e.g. per-property gap-filling instead of whole-query expansion).
- **Vector store (ChromaDB/FAISS)** — semantic re-ranking and cross-paper
  similarity once the corpus grows past what fits in a single ranking call.
- **PDF/full-text ingestion** for papers with open-access PDFs, extending
  extraction beyond the abstract.

## Notes on the source design doc

This implementation follows the multi-agent architecture from the attached
design write-up (Query Planner → Search → Dedup → Rank → Property
Extraction → Report, with a feedback loop), narrowed to the MVP (Phase 1)
scope: PubMed + Semantic Scholar, no Europe PMC, no explicit Evidence
Verification agent yet, and Gemini instead of OpenAI as the LLM backend.
