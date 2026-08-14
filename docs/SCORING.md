# Scoring Method

The required "note explaining the scoring method". This document describes
exactly how a number between 0 and 1 is produced for each candidate.

## Summary

Each candidate receives a **weighted average of five independent components**,
each bounded to `[0, 1]`, followed by a **hard gate** on required-skill
coverage that can veto the result.

```
final_score = Σ (component_i × weight_i) / Σ weight_i
```

| Component | Weight | What it measures | Source |
|---|---:|---|---|
| `skill_coverage` | 0.35 | Fraction of the JD's skills evidenced in the resume | Deterministic (taxonomy) |
| `semantic` | 0.20 | Vector-space similarity of the whole resume to the whole JD | Deterministic (TF-IDF) |
| `experience` | 0.20 | Years of experience against the JD minimum | Deterministic (dates) |
| `llm_judgment` | 0.15 | Depth, seniority, domain relevance, red flags | LLM (`--mode llm` only) |
| `education` | 0.10 | Highest degree against the JD minimum | Deterministic (regex) |

In `--mode offline` the `llm_judgment` component is absent and the remaining
four weights are **renormalised to sum to 1.0** — offline runs are not
uniformly penalised by 0.15.

---

## 1. `skill_coverage` — 0.35

The highest-weighted component because it is the most *falsifiable*: either
the resume evidences Docker or it does not, and a human can check.

Skills are detected against `data/skills_taxonomy.json`, which maps a
canonical skill to the surface forms it appears as in real resumes
(`pytorch` ← "PyTorch", "Torch", "py-torch"). Matching uses word-boundary
regex with two deliberate refinements:

* **Technical punctuation is preserved.** A plain `\b` cannot match `c++` —
  `+` is already a non-word character, so the boundary after it never fires.
  Explicit lookarounds over `[a-z0-9+#.&]` handle `c++`, `c#`, `node.js`, `ci/cd`.
* **Short aliases get strict boundaries and case sensitivity.** One- and
  two-letter skill names are where every false positive lives. `R` and `Go`
  additionally exclude `-` and `.` from their boundaries (killing
  "R-squared", "R. K. Sharma", "Go-to-market") and are matched
  **case-sensitively against the original text**, because "I go to work" and
  "Go" are the same token and only capitalisation separates them. Longer
  aliases keep the permissive boundary so "Python-based" still matches.
* **Pluralisation** is enabled for aliases of 3+ characters ("vector
  databases" → `vector database`) and disabled below that, where it made `ml`
  match "MLS". All of this is pinned in `tests/test_extraction.py`.

Required and preferred skills share one denominator:

```
coverage = (|matched_required| + 0.35 × |matched_preferred|)
           / (|required| + 0.35 × |preferred|)
```

Preferred skills are worth 0.35 of a required one, so **stacking nice-to-haves
can never compensate for a missing must-have**.

## 2. `semantic` — 0.20

A TF-IDF vector space is fitted over the JD **and every resume together**, then
the cosine of the angle between the JD vector and each resume vector is taken.

Fitting over the whole pool is the point: IDF is then calibrated to *this*
applicant set. If every candidate mentions Python, `python` carries
almost no weight, because it does not discriminate between them.

Implementation notes (`src/similarity.py`, written from scratch on numpy):

* **Smoothed IDF**: `ln((1+N)/(1+df)) + 1`.
* **Sublinear TF**: `1 + log(count)`. A resume repeating "Python" twenty times
  does not out-score one that demonstrates it once in context — keyword
  stuffing is a real resume behaviour, and this is the defence against it.
  Pinned by `test_sublinear_tf_resists_keyword_stuffing`.
* **Unigrams + bigrams**: "machine learning" is one concept. Unigrams alone
  let a resume containing "machine" and "learning" separately score as though
  it had the skill.
* **L2 normalisation** before the cosine, so document length does not inflate
  the score.

**Pool rescaling.** Raw JD-to-resume cosine lands in a narrow `0.04–0.15` band
even for a perfect candidate — a JD is mostly prose about the company, a resume
is mostly a dense skill listing, so absolute lexical overlap is low. Left raw,
this component varies by ~0.1 across the entire pool and its 0.20 weight does
essentially nothing. Scores are therefore min-max rescaled across the run
(`SEMANTIC_SCALING=pool`, the default). `SEMANTIC_SCALING=raw` disables it, but
note two things: it applies to the TF-IDF backend only (BM25 is unbounded and
*must* be pool-scaled to be blendable), and **the recommendation thresholds
below are calibrated for pool mode** — under `raw` scores drop by 0.06–0.21
(median 0.11) and nothing reaches the "Strong match" band. The trade-off is
documented in `TRADEOFFS.md`.

## 3. `experience` — 0.20

```
min_years = 0                    → 1.0
years ≥ min_years                → 0.75 + 0.25 × min(1, (years − min) / 4)
years < min_years                → 0.75 × (years / min_years)
```

Two decisions worth defending:

* **Meeting the bar scores 0.75, not 1.0.** This leaves headroom to reward
  genuine additional depth, but that headroom **caps out four years past the
  requirement**. A 15-year candidate for a 3-year role is not five times better
  and frequently is a worse fit.
* **Below the bar the penalty is linear, not a cliff.** A strong 2.5-year
  candidate should still be able to out-rank a mediocre 4-year one on the
  strength of the other components.

Years themselves come from employment date ranges, taking the **union span**
(earliest start → latest end) rather than the sum of individual roles.
Summing double-counts overlapping and concurrent positions and reliably
inflates anyone listing freelance work alongside a full-time job. A
self-reported "N years of experience" line is used only as a fallback, because
dates are evidence and the sentence is a claim.

## 4. `llm_judgment` — 0.15 (LLM mode only)

The LLM is explicitly told **not** to score keyword overlap — a deterministic
system already does that better. It is asked only for what keywords cannot see:
seniority and depth, domain relevance versus superficial similarity, evidence
of shipped production work versus coursework, and red flags.

It is capped at 0.15 because it is the least reproducible input in the system.
If the call fails it returns a neutral **0.5, not 0.0** — an infrastructure
error must never be recorded as evidence against a candidate.

## 5. `education` — 0.10

Meeting the requirement scores 1.0; each level short costs 0.4. **Exceeding it
earns no bonus** — a PhD is not evidence of being better at a hands-on
engineering job, and rewarding it would bake in a credential bias the role
never asked for. Weighted lowest of all five components because degree level is
the weakest predictor of on-the-job performance among the signals available.

Worth stating plainly: on the sample corpus this component scores 1.0 for every
readable candidate, because they all hold at least the bachelor's the JD asks
for. It therefore does no ranking work *here*. That is a property of the sample,
not of the model — it separates candidates on a pool containing non-graduates,
and its 0.10 weight is deliberately small precisely because it usually will not.

---

## The hard gate

A weighted average has one structural flaw: a strong component can always
paper over a disqualifying weak one.

The concrete case on this corpus is **Karthik Reddy**, a data engineer. Tenure
(0.94), a degree (1.00) and adjacent tooling (Spark, Airflow, Docker, SQL,
Kubernetes) carry him to **0.604** — *above* the 0.60 shortlist threshold —
while he is missing 5 of the 9 required skills, including PyTorch, NLP and
Transformers. He is not a plausible ML engineer hire, and no amount of weight
tuning fixes him without also demoting genuinely good candidates, because the
components he scores well on are real.

So before the score bands are applied:

```
required_coverage = |matched_required| / |required|
if required_coverage < 0.60:  → "Reject - missing core requirements"
```

The gate is a **veto, not a tie-breaker**. It is evaluated first, *and it also
sorts*: gated candidates are placed below every un-gated candidate regardless
of their weighted score. Relabelling alone would leave a "Reject" row sitting
above a "Shortlist" row, which is incoherent for a human reading top-down.

There is a matching fail-closed rule at the other end. If the taxonomy
recognises **no** required skills in the JD at all, the agent refuses to run
rather than returning `coverage = 1.0` for everyone. That earlier behaviour
meant an unrelated JD (a chef vacancy, say) scored a blank resume at 0.88 and
"Strong match". Under-scoring everyone is a visibly broken run; over-scoring
everyone looks like a working shortlist, and is far more dangerous.

One false positive of this kind destroys a screener's trust in the tool faster
than several missed candidates.

Bands, once the gate is passed:

| Final score | Recommendation |
|---|---|
| ≥ 0.78 | Strong match — interview |
| ≥ 0.60 | Shortlist — phone screen |
| < 0.60 | Reject — below threshold |

## Reproducibility

* Every deterministic component is a pure function of the input text.
* LLM calls use `temperature=0`.
* Ties break on `(−score, −matched_required, candidate_id)`, so reruns are
  byte-identical. Pinned by `test_run_is_deterministic`.
* Every component, the weights used, and the matched/missing skill lists are
  written to the output JSON. No number in the final score is unexplained.
