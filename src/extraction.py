"""
Turn raw text into structured JobDescription / ResumeProfile records.

Two independent backends:

  rule  - regex + taxonomy. Deterministic, free, no network. Always runs.
  llm   - an LLM reads the document and returns JSON. Runs *in addition to*
          the rule backend, and the two are merged (see `_merge_profile`).

The rule backend is never skipped, even in LLM mode. That is the central
design decision of this agent: the LLM enriches a deterministic baseline
rather than replacing it, so a hallucinated or empty model response degrades
the result instead of destroying it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DEGREE_LEVELS
from .llm import LLMClient, LLMError
from .schema import JobDescription, ResumeProfile
from .taxonomy import canonicalize, find_skills
from .text_utils import normalize

# --------------------------------------------------------------------------
# Regex primitives
# --------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# Matching phone shapes with one regex is a losing game: the previous pattern
# hard-coded a trailing 3+4 grouping and silently missed "+91 98450 11234"
# (5+5), i.e. 9 of the 13 sample resumes. Instead: match loosely, then accept
# only if the digit count is plausible for an international number.
PHONE_CANDIDATE_RE = re.compile(
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?){1,3}\d{3,6}")


# A bare run of digits with no separator is an ID, not a phone number; a
# hyphenated pair of 4-digit numbers is a date range.
_YEAR_RANGE_RE = re.compile(r"(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}")


def extract_phone(text: str) -> Optional[str]:
    """First substring that plausibly denotes a phone number.

    Three filters, each earning its place against a real false positive seen
    in testing:

      * 10-15 digits total - rejects ZIP codes and "CGPA 9.1/10".
      * no embedded year range - rejects "(2016-2020)", which otherwise
        contributes 8 digits and passes the count check.
      * a '+' or at least one separator - rejects "AWS account 123456789012",
        an unbroken 12-digit run that is never how a phone number is written.

    This is a heuristic and it will still be wrong on adversarial input; it is
    correct on all 12 readable resumes in the sample corpus, where the previous
    regex found only 4.
    """
    for match in PHONE_CANDIDATE_RE.finditer(text):
        candidate = match.group().strip()
        digits = re.sub(r"\D", "", candidate)
        if not 10 <= len(digits) <= 15:
            continue
        if _YEAR_RANGE_RE.search(candidate):
            continue
        if "+" not in candidate and not re.search(r"[\s.\-()]", candidate):
            continue
        return candidate
    return None
# (?<!\d) is load-bearing: without it "120 years" matches as "20 years".
YEARS_RE = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b",
                      re.IGNORECASE)
# "2019 - 2023", "Jan 2019 – Present", "03/2019 to 06/2022"
DATE_RANGE_RE = re.compile(
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*|\d{1,2}[/-])?"
    r"(19|20)(\d{2})\s*(?:-|–|—|to|until)\s*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*|\d{1,2}[/-])?"
    r"((?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE)

# Abbreviations REQUIRE their dots. The obvious-looking `b\.?e\.?` (dots
# optional) matches the bare English word "be", so "I would be delighted to
# apply" awarded a bachelor's degree; `m\.?s\.?c?` likewise matched "MS
# Office". Every prose resume scored 1.0 on education and the component did no
# ranking work at all. Spelled-out words keep \b; abbreviations must be
# punctuated or unambiguous ("btech", "msc"). Pinned by
# test_common_english_words_are_not_degrees.
# Three tiers of abbreviation, and the distinction is the whole design:
#
#   unambiguous  - "btech", "msc", "mba" are never ordinary English, so a bare
#                  \b boundary is safe.
#   punctuated   - "b.e.", "m.s." are safe *only* with their dots. The
#                  obvious-looking `b\.?e\.?` (dots optional) matches the bare
#                  word "be", so "I would be delighted to apply" awarded a
#                  bachelor's and every prose resume scored 1.0 on education.
#   context-guarded - "MS in Data Science" is a degree; "MS Office" and "50 ms
#                  in latency" are not. Bare MS/BS/BA/MA are accepted only when
#                  directly followed by "in" and not preceded by a digit.
#
# "BE"/"BA" undotted are deliberately NOT context-guarded: "will be in charge"
# is far more common than "BE in Computer Science". See docs/TRADEOFFS.md.
# "in" only, not "of": degrees are written "BS in Computer Science", whereas
# "the BS of day-to-day delivery" is not a qualification. Caught by
# test_common_english_words_are_not_degrees.
_CTX = r"(?=\s+in\b)"
DEGREE_PATTERNS = [
    (4, r"(\bph\.?d\b|\bdoctorate\b|\bd\.phil\b)"),
    (3, r"(\bmaster'?s?\b|\bm\.\s?s\.|\bm\.?sc\b|\bm\.?tech\b"
        r"|\bmba\b|\bm\.?eng\b|\bm\.?c\.?a\b|\bm\.a\."
        rf"|(?<!\d\s)\b(?:ms|ma)\b{_CTX})"),
    (2, r"(\bbachelor'?s?\b|\bb\.\s?s\.|\bb\.?sc\b|\bb\.?tech\b"
        r"|\bb\.\s?e\.|\bb\.?c\.?a\b|\bb\.a\.|\bb\.?eng\b"
        rf"|\bundergraduate degree\b|(?<!\d\s)\b(?:bs|ba)\b{_CTX})"),
    (1, r"(\bdiploma\b|\bassociate degree\b|\bpolytechnic\b)"),
]
# \b before the preposition: without it "University of Austin" is read as
# a degree field of "Austin".
DEGREE_FIELD_RE = re.compile(
    r"\b(?:in|of)\s+([A-Za-z&\s]{3,40}?)(?:\s*[,\n(]|\s+from\s|\s+at\s|$)")

CURRENT_YEAR = 2026


# --------------------------------------------------------------------------
# Rule-based backend
# --------------------------------------------------------------------------
def detect_degree_level(text: str) -> int:
    """Highest degree mentioned. Ordered highest-first so PhD wins over the
    Bachelor's the same candidate also lists."""
    haystack = normalize(text)
    for level, pattern in DEGREE_PATTERNS:
        if re.search(pattern, haystack, re.IGNORECASE):
            return level
    return DEGREE_LEVELS["none"]


def detect_degree_field(text: str) -> Optional[str]:
    """Field of study following the highest degree mentioned.

    Runs on the same normalised text as detect_degree_level so the two can
    never disagree about which degree they are describing.
    """
    text = normalize(text)
    for _, pattern in DEGREE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            tail = text[match.end():match.end() + 80]
            field = DEGREE_FIELD_RE.search(tail)
            if field:
                return field.group(1).strip().title() or None
    return None


def estimate_years_experience(text: str) -> float:
    """Estimate professional experience, preferring employment dates over claims.

    Two sources, and they disagree often:

      1. Employment date ranges -- we take the *union span* from earliest start
         to latest end, NOT the sum of each role. Summing double-counts
         overlapping or concurrent positions and reliably inflates candidates
         who list freelance work alongside a full-time job.
      2. A self-reported "N years of experience" line.

    Dates win when available because they are evidence; the self-reported
    figure is a claim. When only the claim exists we use it, capped at 45 to
    reject parse artefacts.
    """
    spans: List[tuple] = []
    for match in DATE_RANGE_RE.finditer(text):
        start = int(match.group(1) + match.group(2))
        end_raw = match.group(3).lower()
        end = CURRENT_YEAR if end_raw in {"present", "current", "now"} else int(end_raw)
        if 1960 <= start <= CURRENT_YEAR and start <= end <= CURRENT_YEAR + 1:
            spans.append((start, end))

    if spans:
        span_years = float(max(e for _, e in spans) - min(s for s, _ in spans))
        if 0 < span_years <= 45:
            return round(span_years, 1)

    claims = [float(m.group(1)) for m in YEARS_RE.finditer(text)]
    claims = [c for c in claims if 0 < c <= 45]
    return round(max(claims), 1) if claims else 0.0


def guess_name(text: str) -> str:
    """First plausible line of the document.

    Resumes overwhelmingly lead with the candidate's name. Anything with an
    @, a digit, or more than five words is a header/address, not a name.
    """
    for line in text.splitlines():
        line = line.strip(" \t|-–—*#")
        if not line or "@" in line or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 1 < len(words) <= 5 and sum(w[:1].isupper() for w in words) >= 2:
            return line
    return "Unknown"


def _section(text: str, *headings: str) -> str:
    """Text under a markdown/plain heading, up to the next heading of any kind."""
    for heading in headings:
        pattern = rf"^#{{0,4}}\s*{re.escape(heading)}.*?$"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        rest = text[match.end():]
        nxt = re.search(r"^#{1,4}\s+\S", rest, re.MULTILINE)
        return rest[:nxt.start()] if nxt else rest
    return ""


# Same bug as DEGREE_PATTERNS: `b\.?e\.?` here deleted any JD line containing
# the word "be", silently dropping real requirements before skill harvesting.
EDUCATION_LINE_RE = re.compile(
    r"(\bdegree\b|\bbachelor|\bmaster'?s?\b|\bph\.?d\b|\bdoctorate\b"
    r"|\bb\.?tech\b|\bm\.?tech\b|\bb\.\s?e\.|\bm\.?sc\b|\bb\.?sc\b"
    r"|\bdiploma\b|\bgraduate\b|\bmajor(?:ing)?\b)", re.IGNORECASE)


def strip_education_lines(text: str) -> str:
    """Remove education-requirement lines before harvesting skills from a JD.

    Without this, "Bachelor's degree in Computer Science, Engineering,
    Statistics or a related field" makes `statistics` a REQUIRED SKILL. Every
    candidate then carries a phantom missing skill, which flattens the ranking
    and produces an explanation a recruiter would immediately call wrong.
    Degree requirements are captured separately by detect_degree_level.
    """
    keep = [ln for ln in text.splitlines() if not EDUCATION_LINE_RE.search(ln)]
    return "\n".join(keep)


def extract_jd_rule(text: str) -> JobDescription:
    title_match = re.search(r"^#\s*(.+)$", text, re.MULTILINE)
    required_block = _section(text, "Required Qualifications", "Requirements",
                              "Must Have", "Required")
    preferred_block = _section(text, "Preferred Qualifications", "Nice to Have",
                               "Preferred", "Bonus")

    # Degree lines are stripped for SKILL harvesting only; the untouched block
    # is still used below to read the degree and experience requirements.
    required = find_skills(strip_education_lines(required_block or text))
    preferred = [s for s in find_skills(strip_education_lines(preferred_block))
                 if s not in required]

    return JobDescription(
        title=title_match.group(1).strip() if title_match else "Unspecified Role",
        required_skills=required,
        preferred_skills=preferred,
        min_years_experience=estimate_years_experience(required_block or text),
        min_degree_level=detect_degree_level(required_block or text),
        raw_text=text,
    )


def extract_resume_rule(text: str, candidate_id: str, source_file: str) -> ResumeProfile:
    emails = EMAIL_RE.findall(text)
    return ResumeProfile(
        candidate_id=candidate_id,
        name=guess_name(text),
        email=emails[0] if emails else None,
        phone=extract_phone(text),
        years_experience=estimate_years_experience(text),
        degree_level=detect_degree_level(text),
        degree_field=detect_degree_field(text),
        skills=find_skills(text),
        source_file=source_file,
        raw_text=text,
    )


# --------------------------------------------------------------------------
# LLM backend
# --------------------------------------------------------------------------
RESUME_SYSTEM_PROMPT = """\
You are a resume parser. You extract facts from resume text into JSON.

Rules you must follow:
- Extract ONLY what is present in the text. Never infer, guess or embellish.
- If a field is absent, use null (or an empty list). Do not invent placeholders.
- "years_experience" means professional post-education work experience. Compute
  it from employment dates when present; do not count internships as full years.
- "skills" must be technologies, tools and techniques the candidate actually
  used or claimed - not words that merely appear in the document.
- Return ONLY a JSON object. No prose, no markdown fences.

Schema:
{"name": string|null, "email": string|null, "phone": string|null,
 "years_experience": number, "highest_degree": "none"|"diploma"|"bachelor"|"master"|"phd",
 "degree_field": string|null, "skills": [string], "job_titles": [string],
 "companies": [string]}"""

ASSESSMENT_SYSTEM_PROMPT = """\
You are an experienced technical recruiter assessing one candidate against one
job description.

You are NOT scoring keyword overlap - a separate deterministic system already
does that, and it is better at it than you are. Your job is to judge what
keywords cannot see:
- Is the experience at the right depth and seniority for this role?
- Is it in a relevant domain, or superficially similar but actually different?
- Is there evidence of production/shipping work, or only coursework and demos?
- Are there red flags: unexplained gaps, job hopping, vague unverifiable claims?

Be sceptical and concrete. Cite specifics from the resume. If the resume is
thin, say so and score low - a generous score for a weak candidate is a
failure, not a kindness.

Return ONLY JSON:
{"judgment_score": number between 0 and 1,
 "reasoning": "2-3 sentences citing specific evidence from the resume",
 "strengths": [string], "concerns": [string]}"""


def extract_resume_llm(client: LLMClient, text: str, candidate_id: str,
                       source_file: str) -> ResumeProfile:
    """LLM extraction merged over the rule-based baseline.

    Resume text is truncated to ~12k characters: enough for any realistic
    resume, and it bounds cost and latency on a pathological input.
    """
    base = extract_resume_rule(text, candidate_id, source_file)
    try:
        data = client.complete_json(RESUME_SYSTEM_PROMPT,
                                    f"Resume text:\n\n{text[:12000]}")
    except (LLMError, ValueError) as exc:
        base.warnings.append(f"LLM extraction failed, used rule-based only: {exc}")
        return base
    return _merge_profile(base, data)


def _merge_profile(base: ResumeProfile, data: Dict[str, Any]) -> ResumeProfile:
    """Reconcile LLM output with the deterministic baseline.

    Merge policy, and the reasoning behind each choice:

      skills          -> UNION. Both backends have complementary blind spots:
                         regex misses skills described in prose, the LLM misses
                         ones buried in a dense tech-stack line.
      years           -> the deterministic value ALWAYS wins when it exists;
                         the LLM value is used only when date parsing found
                         nothing at all. A disagreement larger than 2 years is
                         recorded as a warning but does not change the score,
                         because the common cause is the LLM summing
                         overlapping roles. The warning exists so a human can
                         see the conflict on the resumes where it matters.
      contact/degree  -> regex wins outright. These are exactly the fields where
                         a deterministic matcher is more reliable than a model.
      titles/companies-> LLM only; regex has no good handle on them.
    """
    llm_skills = canonicalize([s for s in (data.get("skills") or []) if s])
    base.skills = sorted(set(base.skills) | set(llm_skills))

    llm_years = data.get("years_experience")
    if isinstance(llm_years, (int, float)) and 0 < float(llm_years) <= 45:
        llm_years = float(llm_years)
        if base.years_experience == 0.0:
            base.years_experience = round(llm_years, 1)
        elif abs(llm_years - base.years_experience) > 2.0:
            base.warnings.append(
                f"experience disagreement: dates imply {base.years_experience}y, "
                f"LLM read {llm_years}y - kept the date-derived value")

    if base.degree_level == 0:
        base.degree_level = DEGREE_LEVELS.get(
            str(data.get("highest_degree", "none")).lower(), 0)
    base.degree_field = base.degree_field or data.get("degree_field")
    if base.name == "Unknown" and data.get("name"):
        base.name = str(data["name"])
    base.email = base.email or data.get("email")
    base.phone = base.phone or data.get("phone")
    base.job_titles = [str(t) for t in (data.get("job_titles") or [])][:6]
    base.companies = [str(c) for c in (data.get("companies") or [])][:6]
    return base


def assess_candidate_llm(client: LLMClient, jd: JobDescription,
                         profile: ResumeProfile) -> Dict[str, Any]:
    """Qualitative judgment component. Returns a neutral 0.5 on failure.

    0.5 rather than 0.0 on failure is deliberate: an infrastructure error must
    not be recorded as evidence against the candidate.
    """
    user = (f"JOB DESCRIPTION\n{jd.raw_text[:6000]}\n\n"
            f"---\n\nCANDIDATE RESUME\n{profile.raw_text[:8000]}")
    try:
        data = client.complete_json(ASSESSMENT_SYSTEM_PROMPT, user)
        score = float(data.get("judgment_score", 0.5))
        return {"judgment_score": min(max(score, 0.0), 1.0),
                "reasoning": str(data.get("reasoning", "")).strip(),
                "strengths": data.get("strengths", []) or [],
                "concerns": data.get("concerns", []) or [],
                "ok": True}
    except (LLMError, ValueError, TypeError) as exc:
        return {"judgment_score": 0.5, "reasoning": f"LLM assessment unavailable: {exc}",
                "strengths": [], "concerns": [], "ok": False}
