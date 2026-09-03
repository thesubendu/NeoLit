"""
Central configuration for NeoLit-Agent.

All values can be overridden via environment variables (or a .env file
loaded with python-dotenv). See .env.example for the full list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file is present


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Gemini / LLM ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # --- PubMed / NCBI E-utilities ---
    ncbi_api_key: str = os.getenv("NCBI_API_KEY", "")
    ncbi_email: str = os.getenv("NCBI_EMAIL", "")
    ncbi_tool_name: str = os.getenv("NCBI_TOOL_NAME", "neolit-agent")

    # --- Semantic Scholar ---
    s2_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    # --- Pipeline behaviour ---
    max_papers_per_source: int = int(os.getenv("MAX_PAPERS_PER_SOURCE", "15"))
    top_n_for_extraction: int = int(os.getenv("TOP_N_FOR_EXTRACTION", "12"))
    min_relevant_papers: int = int(os.getenv("MIN_RELEVANT_PAPERS", "8"))
    max_query_expansion_rounds: int = int(os.getenv("MAX_QUERY_EXPANSION_ROUNDS", "2"))
    relevance_score_floor: float = float(os.getenv("RELEVANCE_SCORE_FLOOR", "5.0"))
    extraction_workers: int = int(os.getenv("EXTRACTION_WORKERS", "3"))

    verbose: bool = _get_bool("NEOLIT_VERBOSE", True)


settings = Settings()


def require_gemini_key() -> str:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or export GEMINI_API_KEY in your shell."
        )
    return settings.gemini_api_key
