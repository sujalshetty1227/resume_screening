"""
The scoring model: five independent components combined by a weighted average.

Design principle: every component is bounded to [0, 1] and computed
independently, so a component can be removed (LLM unavailable) or reweighted
without any other component changing meaning. The weights live in config.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import (EXPERIENCE_OVERSHOOT_YEARS, EXPERIENCE_SATURATION_BONUS,
                     PREFERRED_SKILL_WEIGHT, WEIGHTS)
from .schema import JobDescription, ResumeProfile, ScoreBreakdown


def skill_coverage(jd: JobDescription, profile: ResumeProfile
                   ) -> Tuple[float, List[str], List[str], List[str]]:
    """Weighted fraction of the JD's skills evidenced in the resume.

    Required and preferred skills share one denominator so that a candidate
    cannot compensate for a missing must-have by stacking nice-to-haves:
    preferred skills contribute only PREFERRED_SKILL_WEIGHT (0.35) each.
    """
    have = set(profile.skills)
    matched_req = [s for s in jd.required_skills if s in have]
    missing_req = [s for s in jd.required_skills if s not in have]
    matched_pref = [s for s in jd.preferred_skills if s in have]

    denominator = len(jd.required_skills) + PREFERRED_SKILL_WEIGHT * len(jd.preferred_skills)
    if denominator == 0:
        # Fail CLOSED, not open. Returning 1.0 here meant that a JD the
        # taxonomy does not cover (say, a chef vacancy) gave EVERY resume full
        # skill marks -- an empty resume scored 0.88 and "Strong match".
        # Under-scoring everyone is a visibly broken run; over-scoring everyone
        # looks like a working shortlist and is far more dangerous.
        # ResumeScreeningAgent.load_job_description refuses to proceed in this
        # case; this branch is the belt-and-braces for direct API callers.
        return 0.0, matched_req, missing_req, matched_pref

    numerator = len(matched_req) + PREFERRED_SKILL_WEIGHT * len(matched_pref)
    return round(numerator / denominator, 4), matched_req, missing_req, matched_pref


def experience_fit(years: float, min_years: float) -> float:
    """Saturating experience curve.

    Meeting the stated minimum scores 0.75, not 1.0, leaving headroom to
    reward genuine additional depth -- but that headroom caps out four years
    past the bar. A 15-year candidate for a 3-year role is not five times
    better; frequently they are a worse fit. Below the bar the penalty is
    linear rather than a hard cut, because a strong 2.5-year candidate should
    still be able to out-rank a mediocre 4-year one.
    """
    if min_years <= 0:
        return 1.0
    base = 1.0 - EXPERIENCE_SATURATION_BONUS      # 0.75
    if years >= min_years:
        overshoot = min(1.0, (years - min_years) / EXPERIENCE_OVERSHOOT_YEARS)
        return round(base + EXPERIENCE_SATURATION_BONUS * overshoot, 4)
    return round(base * max(0.0, years / min_years), 4)


def education_fit(level: int, min_level: int) -> float:
    """Meeting the requirement scores full marks; each level short costs 0.4.

    No bonus for exceeding it: a PhD is not evidence of being better at a
    hands-on engineering job, and rewarding it would bake in a credential
    bias the role does not ask for.
    """
    if min_level <= 0 or level >= min_level:
        return 1.0
    return round(max(0.0, 1.0 - 0.4 * (min_level - level)), 4)


def _describe(components: Dict[str, float], matched_req: List[str],
              missing_req: List[str], matched_pref: List[str],
              profile: ResumeProfile, jd: JobDescription) -> str:
    """Deterministic, human-readable rationale.

    Generated from the numbers rather than by the LLM, so the explanation is
    guaranteed to match the score it explains -- an LLM asked to justify a
    score it did not compute will confabulate.
    """
    bits = [
        f"Matched {len(matched_req)}/{len(jd.required_skills)} required skills"
        + (f" (missing: {', '.join(missing_req)})" if missing_req else " (all present)"),
        f"{len(matched_pref)}/{len(jd.preferred_skills)} preferred skills",
        f"{profile.years_experience:g}y experience vs {jd.min_years_experience:g}y required",
        f"semantic similarity to JD {components['semantic']:.2f}",
    ]
    return "; ".join(bits) + "."


def score_candidate(jd: JobDescription, profile: ResumeProfile,
                    semantic: float,
                    llm_judgment: Optional[float] = None,
                    llm_reasoning: str = "") -> ScoreBreakdown:
    """Combine all components into a final [0, 1] score.

    Weights are renormalised over whichever components are actually present,
    so running offline (no llm_judgment) rescales the remaining four to sum to
    1.0 rather than silently capping every candidate at 0.85.
    """
    coverage, matched_req, missing_req, matched_pref = skill_coverage(jd, profile)

    components: Dict[str, float] = {
        "skill_coverage": coverage,
        "semantic": round(semantic, 4),
        "experience": experience_fit(profile.years_experience, jd.min_years_experience),
        "education": education_fit(profile.degree_level, jd.min_degree_level),
    }
    if llm_judgment is not None:
        components["llm_judgment"] = round(float(llm_judgment), 4)

    weights = {k: WEIGHTS[k] for k in components}
    total_weight = sum(weights.values())
    final = sum(components[k] * weights[k] for k in components) / total_weight

    reasoning = _describe(components, matched_req, missing_req, matched_pref,
                          profile, jd)
    if llm_reasoning:
        reasoning += f" Recruiter assessment: {llm_reasoning}"

    n_required = len(jd.required_skills)
    return ScoreBreakdown(
        # Fails closed for the same reason skill_coverage does: an empty
        # requirement set must not hand every candidate a passing gate.
        required_coverage=round(len(matched_req) / n_required, 4) if n_required else 0.0,
        components=components,
        weights_used={k: round(w / total_weight, 4) for k, w in weights.items()},
        final_score=round(final, 4),
        matched_required=matched_req,
        missing_required=missing_req,
        matched_preferred=matched_pref,
        reasoning=reasoning,
        llm_used=llm_judgment is not None,
    )
