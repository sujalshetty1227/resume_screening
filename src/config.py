"""Central configuration. Every tunable number in the system lives here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESUME_DIR = DATA_DIR / "resumes"
OUTPUT_DIR = ROOT / "outputs"
TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"
DEFAULT_JD_PATH = DATA_DIR / "job_description.md"

# --------------------------------------------------------------------------
# Scoring weights.
#
# These are a hiring-policy decision, not a mathematical truth, which is
# exactly why they are a single editable block rather than magic numbers
# scattered through scoring.py. Rationale for the defaults:
#
#   skill_coverage (0.35) - the highest-signal, most falsifiable component.
#       Either the candidate demonstrates the required skill or they do not.
#   semantic (0.20) - catches domain fit that a keyword list misses
#       (adjacent tooling, relevant phrasing) but is easily gamed by
#       keyword stuffing, so it is deliberately not dominant.
#   experience (0.20) - strong but saturating signal (see scoring.py).
#   llm_judgment (0.15) - genuine reasoning about *quality* of experience,
#       held to a minority share because it is the least reproducible input.
#   education (0.10) - weakest predictor of on-the-job performance; kept
#       small on purpose.
# --------------------------------------------------------------------------
WEIGHTS = {
    "skill_coverage": 0.35,
    "semantic": 0.20,
    "experience": 0.20,
    "llm_judgment": 0.15,
    "education": 0.10,
}

# Within skill_coverage, how much a "nice to have" counts against a "must have".
PREFERRED_SKILL_WEIGHT = 0.35

# Experience curve: score saturates once the candidate exceeds the JD minimum.
# Someone with 12 years is not 4x better than someone with 3 for a 3-year role.
EXPERIENCE_SATURATION_BONUS = 0.25   # max extra credit above the minimum
EXPERIENCE_OVERSHOOT_YEARS = 4.0     # years above minimum needed to max it out

DEGREE_LEVELS = {"none": 0, "diploma": 1, "associate": 1, "bachelor": 2,
                 "master": 3, "phd": 4, "doctorate": 4}

# How the raw lexical similarity is mapped into the blend.
#   "pool" - min-max rescale across this run's candidates (default)
#   "raw"  - use the bare cosine
# Raw JD-to-resume cosine lands in a narrow 0.04-0.15 band: a JD is mostly
# prose about the company and a resume is mostly dense skill listing, so the
# two share little vocabulary in absolute terms even for a perfect candidate.
# Left raw, the component varies by ~0.1 across the whole pool and its 0.20
# weight does almost nothing. Rescaling restores its discriminating power at
# the cost of making the value POOL-RELATIVE - see docs/TRADEOFFS.md.
SEMANTIC_SCALING = os.getenv("SEMANTIC_SCALING", "pool").lower()
RESUME_SCALING_NOTE = ("note: SEMANTIC_SCALING applies to the tfidf backend "
                       "only; BM25 is unbounded and is always pool-scaled.")

# The recommendation thresholds below are calibrated for SEMANTIC_SCALING=pool.
# Under "raw" every score drops by roughly 0.15-0.20 and nobody reaches the
# "Strong match" band -- the thresholds would need recalibrating with it.

# --------------------------------------------------------------------------
# Recommendation bands.
#
# MIN_REQUIRED_COVERAGE is a HARD GATE applied before the bands, and it also
# sinks the candidate in the sort order: matching fewer than 60% of the
# required skills is a rejection no matter how high the weighted score is.
#
# The structural argument is that a weighted average always lets one strong
# component paper over a disqualifying weak one. On the sample corpus the
# clearest case is Karthik Reddy, a data engineer who scores 0.604 -- above the
# 0.60 shortlist threshold -- on tenure, degree and adjacent tooling while
# missing 5 of 9 required skills including PyTorch and NLP. He is the one
# candidate whose outcome the gate actually changes here; on a larger or more
# senior pool that class of false positive gets much more common, because
# tenure and education saturate while skill coverage does not.
# --------------------------------------------------------------------------
MIN_REQUIRED_COVERAGE = 0.60
SHORTLIST_THRESHOLD = 0.60
STRONG_MATCH_THRESHOLD = 0.78

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
# Temperature 0: scoring must be as reproducible as the vendor allows.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

PROVIDERS = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions",
             "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
             "default_model": "llama-3.3-70b-versatile"},
    "openai": {"url": "https://api.openai.com/v1/chat/completions",
               "key_env": "OPENAI_API_KEY", "model_env": "OPENAI_MODEL",
               "default_model": "gpt-4o-mini"},
    "anthropic": {"url": "https://api.anthropic.com/v1/messages",
                  "key_env": "ANTHROPIC_API_KEY", "model_env": "ANTHROPIC_MODEL",
                  "default_model": "claude-3-5-haiku-20241022"},
    "ollama": {"url": os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1/chat/completions",
               "key_env": None, "model_env": "OLLAMA_MODEL",
               "default_model": "llama3.1:8b"},
}
