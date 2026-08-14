#!/usr/bin/env python3
"""
Resume Screening Agent - command line entry point.

    python run.py                          # deterministic, no API key needed
    python run.py --mode llm               # adds LLM extraction + judgment
    python run.py --mode llm --provider openai
    python run.py --similarity bm25        # swap the lexical ranking function
    python run.py --top 5                  # print only the top 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agent import ResumeScreeningAgent, write_outputs
from src.config import DEFAULT_JD_PATH, OUTPUT_DIR, RESUME_DIR
from src.llm import LLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank resumes against a job description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--jd", type=Path, default=DEFAULT_JD_PATH,
                        help="job description file (.md/.txt/.pdf/.docx)")
    parser.add_argument("--resumes", type=Path, default=RESUME_DIR,
                        help="folder containing resume files")
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline",
                        help="offline = deterministic only, no API key required "
                             "(default); llm = also use an LLM for extraction "
                             "and qualitative judgment")
    parser.add_argument("--provider", choices=["groq", "openai", "anthropic", "ollama"],
                        default=None, help="override LLM_PROVIDER from .env")
    parser.add_argument("--similarity", choices=["tfidf", "bm25"], default="tfidf",
                        help="lexical similarity function (default: tfidf)")
    parser.add_argument("--top", type=int, default=0,
                        help="print only the top N (0 = all). All candidates are "
                             "still written to the output files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prefix", default="ranked_candidates",
                        help="output filename prefix")
    parser.add_argument("--quiet", action="store_true", help="suppress progress log")
    return parser


def print_table(ranked, top: int = 0) -> None:
    rows = ranked[:top] if top else ranked
    header = (f"{'#':>2}  {'Candidate':<22} {'Score':>6}  {'Skills':>6} "
              f"{'Sem':>5} {'Exp':>5} {'Edu':>5}  {'Recommendation':<24} Missing required")
    print("\n" + header)
    print("-" * len(header))
    for c in rows:
        comp = c.score.components
        missing = ", ".join(c.score.missing_required) or "-"
        print(f"{c.rank:>2}  {c.profile.name[:22]:<22} {c.score.final_score:>6.3f}  "
              f"{comp['skill_coverage']:>6.2f} {comp['semantic']:>5.2f} "
              f"{comp['experience']:>5.2f} {comp['education']:>5.2f}  "
              f"{c.recommendation:<24} {missing}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda m: None) if args.quiet else (lambda m: print(m))

    try:
        agent = ResumeScreeningAgent(mode=args.mode, provider=args.provider,
                                     similarity=args.similarity, log=log)
    except LLMError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    try:
        jd = agent.load_job_description(args.jd)
        profiles = agent.load_profiles(args.resumes)
        ranked = agent.rank(jd, profiles)
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    print_table(ranked, args.top)

    paths = write_outputs(ranked, jd, args.output_dir, args.prefix)
    print("\nWrote:")
    for kind, path in paths.items():
        print(f"  {kind:<9} {path}")

    shortlisted = sum(1 for c in ranked if not c.recommendation.startswith("Reject"))
    print(f"\n{shortlisted}/{len(ranked)} candidates recommended for a screen or "
          f"interview. Mode: {args.mode}, similarity: {args.similarity}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
