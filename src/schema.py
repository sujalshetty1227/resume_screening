"""Typed records passed between pipeline stages.

Dataclasses rather than free-form dicts: the LLM returns untrusted JSON, and
coercing it into a declared shape at the boundary means a malformed model
response fails loudly at parse time instead of silently producing a wrong score.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class JobDescription:
    title: str = "Unspecified Role"
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_years_experience: float = 0.0
    min_degree_level: int = 0
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_text")
        return d


@dataclass
class ResumeProfile:
    candidate_id: str
    name: str = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    years_experience: float = 0.0
    degree_level: int = 0
    degree_field: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    job_titles: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    source_file: str = ""
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_text")
        return d


@dataclass
class ScoreBreakdown:
    """Every number the final score is made of, kept for auditability.

    A hiring decision that cannot be explained back to the candidate (or to a
    regulator) is not usable, so no component is ever collapsed away.
    """
    components: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    matched_required: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    matched_preferred: List[str] = field(default_factory=list)
    required_coverage: float = 0.0   # fraction of REQUIRED skills matched
    reasoning: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankedCandidate:
    rank: int
    profile: ResumeProfile
    score: ScoreBreakdown

    @property
    def recommendation(self) -> str:
        """Hard gate first, then the score bands.

        Order matters: the gate is not a tie-breaker, it is a veto.
        """
        from .config import (MIN_REQUIRED_COVERAGE, SHORTLIST_THRESHOLD,
                             STRONG_MATCH_THRESHOLD)
        if self.score.required_coverage < MIN_REQUIRED_COVERAGE:
            return "Reject - missing core requirements"
        if self.score.final_score >= STRONG_MATCH_THRESHOLD:
            return "Strong match - interview"
        if self.score.final_score >= SHORTLIST_THRESHOLD:
            return "Shortlist - phone screen"
        return "Reject - below threshold"

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "recommendation": self.recommendation,
                "profile": self.profile.to_dict(), "score": self.score.to_dict()}
