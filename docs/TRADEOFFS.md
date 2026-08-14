# Tradeoffs, Known Failures and What I'd Do Next

Written honestly. Everything below is a real limitation of the code in this
repository, and several are visible in the committed sample output.

---

## The central design decision: the LLM enriches, it does not decide

The obvious build is: paste the JD and each resume into an LLM, ask for a score
out of 10, sort. I deliberately did not do that.

**Why not.** LLM-only scoring fails as a screening system in four specific ways:

1. **Not reproducible.** Two runs give two different shortlists. A hiring
   process that cannot be re-run identically cannot be audited or defended.
2. **Not falsifiable.** "7/10, strong candidate" cannot be checked. "Missing
   Docker" can be checked in ten seconds by anyone.
3. **Position and verbosity bias.** LLM scores drift with resume length and
   ordering — a longer resume tends to score higher independent of content.
4. **It fails closed on infrastructure.** No API key, rate limit, or outage
   means no shortlist at all.

**What I built instead.** Four deterministic components carry 85% of the
weight; the LLM carries 15% and is asked only for the judgment that keyword
matching genuinely cannot provide (depth, seniority, domain relevance, red
flags). The rule-based extractor **always runs**, even in LLM mode, and the LLM
output is merged over it rather than replacing it — so a hallucinated or empty
model response degrades the result instead of destroying it.

**What this costs.** A pure-LLM agent would understand phrasing my taxonomy
misses. A candidate who writes "built encoder-decoder architectures for
sequence labelling" without ever writing the word "NLP" is under-credited by
the deterministic path. The LLM component partially recovers this, at 15%
weight. This is a real accuracy cost paid deliberately for auditability.

---

## Known failure modes

### 1. Scanned PDFs produce nothing (demonstrated in the sample output)

`data/resumes/scanned_unreadable.pdf` is an image-only PDF, included on
purpose. `pypdf` extracts no text, and the candidate scores 0.024 and ranks
last. **The agent warns rather than failing silently** — but the score is still
wrong, and in a real pipeline this would need routing to human review, not
rejection. OCR (Tesseract, or a vision model) is the fix and is not implemented.

### 2. Experience years are inflated by education-era dates

Wei Zhang's resume lists "Research Assistant, NUS 2019 – 2023" (his PhD) and
"Research Scientist 2023 – Present". The union-span heuristic reports **7
years**; his actual industry experience is **3**. The union span is the right
call against the worse failure (summing overlapping roles), but it cannot tell
a doctorate from a job. Distinguishing them needs the entry to be classified as
education or employment first — which is exactly the kind of judgment the LLM
component is good at, and a good next step would be to let the LLM's
`years_experience` win when it disagrees downward on a resume containing a
doctorate.

### 3. Skill detection proves mention, not competence

`find_skills` proves a token appeared in the document. It cannot distinguish
"led the migration to Kubernetes" from "exposure to Kubernetes" from
"Kubernetes" in a comma-separated skills dump. Vikram Desai is credited with
`transformers` because he "integrated a third-party LLM API" — defensible, but
it is not the same skill as fine-tuning one. Proficiency weighting by
surrounding context is the fix.

### 4. Pool-relative semantic scores are not comparable across runs

With the default `SEMANTIC_SCALING=pool`, the semantic component is min-max
rescaled within each run. **The best candidate in the pool always scores 1.0
on this component, even if the pool is uniformly weak**, and adding one
candidate changes every other candidate's semantic score.

This is a genuine trade: raw cosine is stable across runs but occupies a
0.04–0.15 band and does almost no ranking work at 0.20 weight (I measured this
— it is why the rescaling exists). Rescaling makes the component useful within
a run at the cost of cross-run comparability. `SEMANTIC_SCALING=raw` restores
the stable behaviour, but it applies to the TF-IDF backend only (BM25 is
unbounded and must be pool-scaled to be blendable at all), and the 0.60/0.78
recommendation thresholds are calibrated for pool mode — under `raw` scores drop by
0.06–0.21 (median 0.11) and nothing reaches the "Strong match" band without
recalibration. The correct fix is neither: it is sentence embeddings,
which produce meaningful absolute cosines. See "next steps".

### 5. The taxonomy is hand-written and will go stale

`skills_taxonomy.json` covers ~40 skills for software/ML roles. A skill absent
from the taxonomy is invisible to `skill_coverage` no matter how prominent it
is in the resume, so candidates are **under**-credited. The LLM path partially
covers this — `canonicalize` keeps unrecognised skills the LLM reports rather
than dropping them — but the deterministic path cannot.

The dangerous version of this is a JD from a different domain entirely, where
the taxonomy recognises *nothing*. `skill_coverage` used to return 1.0 on an
empty requirement set, so every resume — including a blank one — scored ~0.88
and "Strong match". It now fails closed: `load_job_description` refuses to run
and says why. The system under-serves an unsupported domain loudly instead of
producing a confident, meaningless ranking.

### 6. Section-heading parsing is brittle

Required and preferred skills are separated by locating headings ("Required
Qualifications", "Preferred Qualifications", and a handful of synonyms). A JD
that uses different wording falls back to harvesting skills from the whole
document, which collapses the required/preferred distinction and makes the hard
gate more aggressive than intended. Safe, but crude.

### 7. Undotted "BE" and "BA" are not detected as degrees

The degree matcher accepts bare `MS`/`BS`/`MA` only when directly followed by
"in" ("MS in Data Science"), and accepts `BE`/`BA` only when punctuated
("B.E.", "B.A."). This is a deliberate accuracy trade in the *other* direction
from the bug it replaced: "we will **be in** charge" and "the **BS of**
day-to-day delivery" are far more common in resume prose than "BE in Computer
Science" is, and a false degree on every prose resume is worse than a missed
one on a few. A candidate writing "BE in Mechanical" without dots is
under-credited by 0.10 × 0.4 ≈ 0.04 on the final score. The LLM path recovers
it, since the extraction prompt asks for `highest_degree` directly.

### 8. Name extraction is a heuristic

`guess_name` takes the first line that looks like a name (2–5 words, no digits,
no `@`, mostly capitalised). It handles the sample corpus, and it will fail on
resumes that lead with a letterhead or a photo caption.

### 9. The weights are asserted, not learned

The five weights are my hiring judgment, not fitted parameters. They are
defensible and documented, but they are not *validated* — I have no labelled
outcome data to fit them against. See "next steps".

---

## Bugs the tests caught (and what they cost)

Listed because "what broke and how I found it" is more informative than a clean
history. All four are fixed and each has a regression test.

1. **`b\.?e\.?` matched the English word "be".** Making the dots optional in the
   degree abbreviations meant "I would be delighted to apply" awarded a
   bachelor's, and "MS Office" awarded a master's. Every prose resume scored
   1.0 on education, so the component silently became a constant and did no
   ranking work at all. The same regex was duplicated in `EDUCATION_LINE_RE`,
   where it deleted any JD line containing "be" before skill extraction.
   → `test_common_english_words_are_not_degrees`.

2. **`120 years` parsed as `20 years`.** A missing `(?<!\d)` on the left of the
   experience regex. Caught by a test written specifically to probe absurd
   inputs, which is the reason to write those tests.
   → `test_absurd_values_rejected`.

3. **The phone regex hard-coded a 3+4 trailing grouping**, so `+91 98450 11234`
   (5+5) never matched — 9 of the 13 sample resumes. Replaced with a loose
   match plus a 10–15 digit plausibility check.
   → `test_international_groupings`.

4. **BM25 min-max included the JD's own self-similarity.** The JD always scores
   highest against its own query, so it anchored the maximum and squashed every
   real candidate toward zero. Dropping index 0 *before* rescaling fixed it.

And one bad test, worth flagging because it was worse than no test: the
substring-safety test asserted that `"r"` and `"go"` do not match "Reporting"
and "Google" — but neither bare alias existed in the taxonomy, so it passed
against any regex whatsoever, while `SCORING.md` cited it as proof. Both
aliases are now real (resumes do list "R" and "Go"), which surfaced two genuine
false positives the loose boundaries allowed: `R&D` matched `r`, and `MLS`
matched `ml` via the plural rule. Fixed by excluding `&` from the boundaries
and disabling pluralisation for aliases under three characters.

---

## Fairness

Real, and not fully solved here:

* **Not addressed:** name, gender, nationality and university prestige are all
  present in `raw_text` and therefore reachable by the `semantic` component and
  by the LLM. I have not implemented anonymisation.
* **Partially addressed:** education is weighted lowest (0.10) and exceeding
  the requirement earns *no* bonus, so the model cannot reward credentialism.
* **Structurally addressed:** every score decomposes into named components with
  matched/missing skill lists, so a disparate-impact audit is possible on the
  output as it stands. An opaque LLM score would not permit that.

A production version should strip names, addresses and institution names before
scoring, then re-attach them only for the human reviewing the shortlist.

---

## What I would do with more time

Roughly in order of value per hour:

1. **Sentence embeddings alongside TF-IDF.** TF-IDF is purely lexical: "NLP"
   and "natural language processing" are one skill only because I hand-wrote
   that alias. A bi-encoder (`all-MiniLM-L6-v2`) captures it for free and gives
   absolute cosines, removing the pool-rescaling compromise entirely. I skipped
   it in the time available because it adds a ~2GB PyTorch dependency and a
   model download, which directly harms the "reviewer can run this in minutes"
   requirement. That is a setup-cost trade, not a technical objection.
2. **A labelled evaluation set.** ~50 resumes with recruiter-assigned
   labels, reported as nDCG@10 and Spearman correlation against the human
   ranking. This turns every weight in `config.py` from an assertion into a
   measurement, and would let me fit the weights properly.
3. **OCR fallback** for scanned PDFs (failure #1).
4. **Proficiency weighting**: score a skill by the strength of its surrounding
   verbs ("architected" > "used" > listed in a skills dump) (failure #3).
5. **Bias audit**: rerun with names and universities stripped and measure how
   much the ranking moves. If it moves at all, that is a finding worth acting on.
6. **Structured section parsing** of resumes (experience / education /
   projects), so "3 years of projects" is never mistaken for 3 years of work.

## Robustness check: the ranking is not an artefact of one similarity function

Swapping the lexical component from TF-IDF cosine to Okapi BM25
(`--similarity bm25`) changes the *scores* but barely moves the *ranking*:

```
Spearman rho (TF-IDF vs BM25 ordering) = 1.000
Top-5 overlap                          = 5/5
```

That is worth knowing. If the two ranking functions disagreed sharply, the
shortlist would be an artefact of an arbitrary implementation choice rather
than a property of the candidates. They do not, which is weak but real evidence
that the ordering is driven by the deterministic skill/experience components
rather than by lexical noise. It is not a substitute for a labelled evaluation
set (see "next steps"), because both functions could be wrong in the same way.

## Model choice

Default is **Groq + `llama-3.3-70b-versatile`**: a free tier means a reviewer
can run LLM mode without a paid account, and it is fast enough that screening
13 resumes takes seconds. The client is raw `requests` against an
OpenAI-compatible schema rather than a vendor SDK, so `--provider openai`,
`anthropic` or `ollama` all work through the same ~80 lines — no vendor lock-in
and four fewer dependencies. Temperature is pinned at 0 throughout.

The model choice barely matters here, which is itself the point: the LLM
carries 15% of the weight and performs bounded, structured tasks (extract this
JSON, judge this fit). Swapping the model changes the output slightly. Removing
it entirely (`--mode offline`) still produces a complete, defensible shortlist.
