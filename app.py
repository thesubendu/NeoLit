"""
Optional Streamlit front-end for NeoLit-Agent.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from config import settings
from pipeline import NeoLitPipeline

st.set_page_config(page_title="NeoLit-Agent", page_icon="🧬", layout="wide")
st.title("🧬 NeoLit-Agent")
st.caption("Agentic literature retrieval + neoantigen property extraction")

with st.sidebar:
    st.header("Settings")
    max_per_source = st.slider("Max papers per source per query", 5, 30, settings.max_papers_per_source)
    top_n = st.slider("Papers to extract properties from", 3, 25, settings.top_n_for_extraction)
    min_relevant = st.slider("Min relevant papers before stopping", 3, 20, settings.min_relevant_papers)
    max_rounds = st.slider("Max query-expansion rounds", 1, 4, settings.max_query_expansion_rounds)
    if not settings.gemini_api_key:
        st.warning("GEMINI_API_KEY not set — see .env.example. Running in heuristic-only mode.")

question = st.text_area(
    "Research question",
    placeholder="e.g. Find recent literature on the properties of high-confidence "
    "neoantigens in glioblastoma",
    height=80,
)

run = st.button("Run agent", type="primary", disabled=not question.strip())

if run:
    log_box = st.empty()
    log_lines: list[str] = []

    def on_event(msg: str):
        log_lines.append(msg)
        log_box.code("\n".join(log_lines[-20:]))

    with st.spinner("Running multi-agent pipeline..."):
        pipeline = NeoLitPipeline(use_llm=bool(settings.gemini_api_key))
        result = pipeline.run(
            research_question=question,
            max_papers_per_source=max_per_source,
            top_n_extract=top_n,
            min_relevant=min_relevant,
            max_expansion_rounds=max_rounds,
            on_event=on_event,
        )

    st.success(f"Done — synthesized from {len(result.ranked_papers)} papers")
    st.download_button(
        "Download report (.md)", result.report_markdown, file_name="neolit_report.md"
    )
    st.markdown(result.report_markdown)

    with st.expander("Ranked source papers (raw data)"):
        st.json([p.to_dict() for p in result.ranked_papers])
