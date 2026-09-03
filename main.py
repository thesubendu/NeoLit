#!/usr/bin/env python3
"""
NeoLit-Agent CLI.

Usage:
    python main.py "Find recent literature on properties of high-confidence
    neoantigens in glioblastoma" --output report.md

Run `python main.py --help` for all options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import settings
from pipeline import NeoLitPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NeoLit-Agent: neoantigen literature research agent")
    p.add_argument("research_question", help="The research question to investigate")
    p.add_argument(
        "--max-per-source",
        type=int,
        default=settings.max_papers_per_source,
        help="Max papers to fetch per query per source (default: %(default)s)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=settings.top_n_for_extraction,
        help="Number of top-ranked papers to run property extraction on (default: %(default)s)",
    )
    p.add_argument(
        "--min-relevant",
        type=int,
        default=settings.min_relevant_papers,
        help="Minimum relevant papers before stopping query expansion (default: %(default)s)",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=settings.max_query_expansion_rounds,
        help="Max query-expansion rounds if coverage is thin (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default="report.md",
        help="Output markdown file path (default: %(default)s)",
    )
    p.add_argument(
        "--json-output",
        default=None,
        help="Optional path to also dump structured paper+property data as JSON",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without Gemini (heuristic ranking only, no property extraction/synthesis). "
        "Useful for testing search connectivity without an API key.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        pipeline = NeoLitPipeline(use_llm=not args.no_llm)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = pipeline.run(
        research_question=args.research_question,
        max_papers_per_source=args.max_per_source,
        top_n_extract=args.top_n,
        min_relevant=args.min_relevant,
        max_expansion_rounds=args.max_rounds,
    )

    out_path = Path(args.output)
    out_path.write_text(result.report_markdown, encoding="utf-8")
    print(f"\nReport written to {out_path.resolve()}")

    if args.json_output:
        json_path = Path(args.json_output)
        payload = {
            "research_question": args.research_question,
            "papers": [p.to_dict() for p in result.ranked_papers],
            "run_log": result.run_log.events,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Structured data written to {json_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
