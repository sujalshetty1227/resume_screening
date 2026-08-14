"""
Orchestration: the Input -> Retrieve -> Think -> Act -> Output loop.

    JD file + resume folder
        -> parse every document to text
        -> extract structured records (rules, optionally + LLM)
        -> fit one TF-IDF space over the JD and all resumes
        -> score each candidate on five components
        -> rank and write CSV / JSON / Markdown
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from .config import (DEFAULT_JD_PATH, MIN_REQUIRED_COVERAGE, OUTPUT_DIR,
                     RESUME_DIR, RESUME_SCALING_NOTE, SEMANTIC_SCALING)
from .extraction import (assess_candidate_llm, extract_jd_rule,
                         extract_resume_llm, extract_resume_rule)
from .llm import LLMClient
from .parsing import discover_resumes, extract_text
from .schema import JobDescription, RankedCandidate, ResumeProfile
from .scoring import score_candidate
from .similarity import BM25, TfidfIndex, cosine_similarity


def _rescale(values: List[float]) -> List[float]:
    """Min-max a list into [0, 1]; returns 0.5 everywhere if all values tie."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class ResumeScreeningAgent:
    def __init__(self, mode: str = "offline", provider: Optional[str] = None,
                 similarity: str = "tfidf",
                 log: Optional[Callable[[str], None]] = None) -> None:
        """mode: 'offline' (deterministic only) or 'llm' (deterministic + LLM)."""
        self.mode = mode
        self.similarity = similarity
        self.log = log or (lambda msg: None)
        self.client: Optional[LLMClient] = None
        if mode == "llm":
            # Constructed eagerly so a missing API key fails in the first
            # second rather than after parsing 200 resumes.
            self.client = LLMClient(provider)
            self.log(f"LLM backend: {self.client.provider} / {self.client.model}")

    # -- stages ------------------------------------------------------------
    def load_job_description(self, path: Path = DEFAULT_JD_PATH) -> JobDescription:
        text, warnings = extract_text(Path(path))
        if not text.strip():
            raise ValueError(f"job description is empty or unreadable: {path}")
        for w in warnings:
            self.log(f"  ! JD: {w}")
        jd = extract_jd_rule(text)
        if not jd.required_skills:
            raise ValueError(
                f"no required skills recognised in {path}. The skills taxonomy "
                f"(data/skills_taxonomy.json) does not cover this job "
                f"description, so every candidate would score identically and "
                f"the ranking would be meaningless. Add the relevant skills to "
                f"the taxonomy, or check the JD uses a 'Required "
                f"Qualifications' heading.")
        self.log(f"JD: {jd.title} | {len(jd.required_skills)} required, "
                 f"{len(jd.preferred_skills)} preferred skills | "
                 f"{jd.min_years_experience:g}y minimum")
        return jd

    def load_profiles(self, directory: Path = RESUME_DIR) -> List[ResumeProfile]:
        paths = discover_resumes(Path(directory))
        if not paths:
            raise ValueError(f"no supported resume files found in {directory}")
        self.log(f"Found {len(paths)} resumes in {directory}")

        profiles: List[ResumeProfile] = []
        for i, path in enumerate(paths, start=1):
            text, warnings = extract_text(path)
            candidate_id = f"C{i:03d}"
            if self.client is not None:
                profile = extract_resume_llm(self.client, text, candidate_id, path.name)
            else:
                profile = extract_resume_rule(text, candidate_id, path.name)
            profile.warnings.extend(warnings)
            profiles.append(profile)
            flag = "  !" if profile.warnings else "   "
            self.log(f"{flag} [{candidate_id}] {profile.name:<24} {path.name:<28} "
                     f"{len(profile.skills):>2} skills, {profile.years_experience:g}y")
            for w in profile.warnings:
                self.log(f"      ! {w}")
        return profiles

    def rank(self, jd: JobDescription, profiles: List[ResumeProfile]
             ) -> List[RankedCandidate]:
        """Score and order candidates.

        The TF-IDF space is fitted over the JD plus every resume at once, so
        IDF is calibrated to this applicant pool: a term every candidate uses
        carries no discriminative weight, which is the behaviour we want.
        """
        corpus = [jd.raw_text] + [p.raw_text for p in profiles]
        if self.similarity == "bm25" and SEMANTIC_SCALING != "pool":
            self.log(RESUME_SCALING_NOTE)

        if self.similarity == "bm25":
            bm25 = BM25(corpus)
            # Index 0 is the JD itself. It must be dropped BEFORE rescaling:
            # the JD always scores highest against its own query, so leaving it
            # in anchors the max and squashes every real candidate toward 0.
            raw = [float(x) for x in bm25.scores(jd.raw_text)[1:]]
            # BM25 is unbounded, so it CANNOT be blended raw alongside four
            # components defined on [0, 1]. Pool scaling is mandatory here;
            # SEMANTIC_SCALING only applies to the TF-IDF backend, whose cosine
            # is already bounded. Stated explicitly rather than silently
            # ignoring the setting.
            semantic_scores = _rescale(raw)
        else:
            index = TfidfIndex.fit(corpus)
            jd_vec = index.transform(jd.raw_text)
            raw = [cosine_similarity(jd_vec, index.transform(p.raw_text))
                   for p in profiles]
            semantic_scores = _rescale(raw) if SEMANTIC_SCALING == "pool" else raw

        scored = []
        for profile, semantic in zip(profiles, semantic_scores):
            judgment, reasoning = None, ""
            if self.client is not None:
                assessment = assess_candidate_llm(self.client, jd, profile)
                judgment = assessment["judgment_score"]
                reasoning = assessment["reasoning"]
                if not assessment["ok"]:
                    profile.warnings.append("LLM assessment failed; used neutral 0.5")
            scored.append((profile, score_candidate(jd, profile, semantic,
                                                    judgment, reasoning)))

        # Gated candidates sort BELOW every un-gated one, whatever their
        # weighted score. Without this the gate only changed the label: a
        # candidate rejected for missing core skills could still sit above a
        # shortlisted one in the ranked list, which is incoherent output for a
        # human reading top-down. Within each group: score, then required-skill
        # count, then ID -- so ties break deterministically and reruns are
        # byte-identical.
        scored.sort(key=lambda t: (t[1].required_coverage < MIN_REQUIRED_COVERAGE,
                                   -t[1].final_score, -len(t[1].matched_required),
                                   t[0].candidate_id))
        return [RankedCandidate(rank=i, profile=p, score=s)
                for i, (p, s) in enumerate(scored, start=1)]

    def run(self, jd_path: Path = DEFAULT_JD_PATH,
            resume_dir: Path = RESUME_DIR) -> List[RankedCandidate]:
        jd = self.load_job_description(jd_path)
        profiles = self.load_profiles(resume_dir)
        return self.rank(jd, profiles)


# -- output writers --------------------------------------------------------
def write_outputs(ranked: List[RankedCandidate], jd: JobDescription,
                  output_dir: Path = OUTPUT_DIR, prefix: str = "ranked_candidates"
                  ) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate in ranked:
        profile, score = candidate.profile, candidate.score
        rows.append({
            "rank": candidate.rank,
            "candidate_id": profile.candidate_id,
            "name": profile.name,
            "source_file": profile.source_file,
            "final_score": score.final_score,
            "recommendation": candidate.recommendation,
            "required_coverage": score.required_coverage,
            "skill_coverage": score.components.get("skill_coverage"),
            "semantic": score.components.get("semantic"),
            "experience": score.components.get("experience"),
            "education": score.components.get("education"),
            "llm_judgment": score.components.get("llm_judgment"),
            "years_experience": profile.years_experience,
            "degree_level": profile.degree_level,
            "degree_field": profile.degree_field,
            "matched_required": "; ".join(score.matched_required),
            "missing_required": "; ".join(score.missing_required),
            "matched_preferred": "; ".join(score.matched_preferred),
            "email": profile.email,
            "reasoning": score.reasoning,
            "warnings": "; ".join(profile.warnings),
        })

    csv_path = output_dir / f"{prefix}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = output_dir / f"{prefix}.json"
    json_path.write_text(json.dumps(
        {"job_description": jd.to_dict(),
         "candidate_count": len(ranked),
         "candidates": [c.to_dict() for c in ranked]},
        indent=2), encoding="utf-8")

    md_path = output_dir / f"{prefix}_report.md"
    md_path.write_text(_render_report(ranked, jd), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def _render_report(ranked: List[RankedCandidate], jd: JobDescription) -> str:
    lines = [f"# Shortlist - {jd.title}", "",
             f"{len(ranked)} candidates screened. "
             f"Required skills: {', '.join(jd.required_skills)}.", "",
             "| Rank | Candidate | Score | Recommendation | Missing required |",
             "|-----:|-----------|------:|----------------|------------------|"]
    for c in ranked:
        missing = ", ".join(c.score.missing_required) or "-"
        lines.append(f"| {c.rank} | {c.profile.name} | {c.score.final_score:.3f} "
                     f"| {c.recommendation} | {missing} |")

    lines += ["", "## Per-candidate detail", ""]
    for c in ranked:
        comp = " | ".join(f"{k}={v:.2f}" for k, v in c.score.components.items())
        lines += [f"### {c.rank}. {c.profile.name} ({c.profile.candidate_id}) "
                  f"- {c.score.final_score:.3f} - {c.recommendation}",
                  f"*{c.profile.source_file}* | {c.profile.years_experience:g}y experience",
                  "", f"Components: {comp}", "", c.score.reasoning, ""]
        if c.profile.warnings:
            lines += ["> Warnings: " + "; ".join(c.profile.warnings), ""]
    return "\n".join(lines)
