# Resume Screening Agent

Ranks a folder of resumes against a job description and outputs a scored,
ordered shortlist with per-candidate reasoning.

Built for the Rooman 24-Hour AI Agent Challenge — *Resume Screening Agent
(Intermediate)*.

> **My agent takes a job description and a folder of resumes (PDF / DOCX / TXT)
> and produces a ranked shortlist with a transparent, auditable score and a
> reason for every candidate's position.**

**It runs with no API key.** The default mode is fully deterministic and
offline. LLM mode is an opt-in enhancement, not a dependency.

---

## Quick start (about 60 seconds)

```bash
git clone https://github.com/<your-username>/resume-screening-agent.git   # <- your fork/clone URL
cd resume-screening-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python run.py
```

That's it. No API key, no `.env`, no model download. The sample job description
and 13 sample resumes are committed to the repo, so the command above produces
a complete ranked shortlist immediately.

Requires **Python 3.10+** (`numpy==2.2.6` does not build on 3.9).

### Verify it works

```bash
python -m unittest discover -s tests -v
```

68 tests, all offline, no network required. Several are regression tests for
bugs found during development and described in [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md).

---

## Actual output

```
$ python run.py --quiet

 #  Candidate               Score  Skills   Sem   Exp   Edu  Recommendation           Missing required
------------------------------------------------------------------------------------------------------
 1  Priya Raghavan          0.945    0.94  1.00  0.88  1.00  Strong match - interview -
 2  Elena Petrova           0.844    0.85  0.60  1.00  1.00  Strong match - interview rest api
 3  Arjun Menon             0.813    0.85  0.60  0.88  1.00  Strong match - interview machine learning
 4  Meera Nair              0.810    0.81  0.71  0.81  1.00  Strong match - interview -
 5  Daniel Okafor           0.786    0.76  0.52  1.00  1.00  Strong match - interview nlp, transformers
 6  Fatima Sheikh           0.704    0.66  0.46  0.88  1.00  Shortlist - phone screen nlp, transformers
 7  Wei Zhang               0.698    0.63  0.36  1.00  1.00  Shortlist - phone screen docker, rest api
 8  Vikram Desai            0.666    0.66  0.30  0.88  1.00  Shortlist - phone screen nlp, pytorch
 9  Ananya Iyer             0.652    0.81  0.60  0.25  1.00  Shortlist - phone screen -
10  Karthik Reddy           0.604    0.46  0.33  0.94  1.00  Reject - missing core requirements machine learning, nlp, pytorch, rest api, transformers
11  Rohit Sharma            0.588    0.39  0.31  1.00  1.00  Reject - missing core requirements machine learning, nlp, python, pytorch, transformers
12  Sneha Kulkarni          0.556    0.36  0.30  0.94  1.00  Reject - missing core requirements docker, nlp, pytorch, rest api, transformers
13  Unknown                 0.024    0.00  0.00  0.00  0.20  Reject - missing core requirements docker, git, machine learning, nlp, python, pytorch, rest api, sql, transformers

Wrote:
  csv       outputs/ranked_candidates.csv
  json      outputs/ranked_candidates.json
  markdown  outputs/ranked_candidates_report.md

9/13 candidates recommended for a screen or interview. Mode: offline, similarity: tfidf.
```

Rows 10–13 are vetoed by the **hard gate** on required-skill coverage. Karthik
Reddy (row 10) is the case that motivates it: he scores 0.604, *above* the
0.60 shortlist threshold, on tenure and adjacent tooling while missing PyTorch,
NLP and 3 other required skills. The gate both relabels him and sinks him below
every un-gated candidate.

Row 13 is a deliberately unreadable scanned PDF — the agent flags it in the
`warnings` column rather than silently scoring the candidate as having no
skills.

Committed outputs are in [`outputs/`](outputs/).

---

## Optional: enable LLM mode

```bash
cp .env.example .env
# edit .env, add a key for ONE provider
python run.py --mode llm
```

Get a **free** Groq key at <https://console.groq.com/keys> (the default
provider). `openai`, `anthropic` and `ollama` are also supported — set
`LLM_PROVIDER` in `.env`, or pass `--provider openai` on the command line.

LLM mode adds two things: a second extraction pass merged over the rule-based
one, and a qualitative recruiter assessment worth 15% of the score. It costs
roughly 2 API calls per resume.

---

## All options

```
python run.py --help

--jd PATH                job description file (.md/.txt/.pdf/.docx)
--resumes PATH           folder of resume files
--mode {offline,llm}     default: offline (no API key required)
--provider {groq,openai,anthropic,ollama}
--similarity {tfidf,bm25}   lexical ranking function, default: tfidf
--top N                  print only the top N (all are still written to file)
--output-dir PATH
--prefix NAME            output filename prefix
--quiet
```

Run it on your own data:

```bash
python run.py --jd path/to/your_jd.pdf --resumes path/to/your_resumes/ --top 10
```

---

## How it works

```
JD file + resume folder
   │
   ├─ parse ........... PDF (pypdf) / DOCX (python-docx) / TXT → plain text
   ├─ extract ......... regex + skills taxonomy → structured records
   │                    (+ an LLM pass merged on top, in --mode llm)
   ├─ vectorise ....... one TF-IDF space fitted over the JD and all resumes
   ├─ score ........... 5 bounded components → weighted average
   ├─ gate ............ veto anyone below 60% required-skill coverage
   └─ rank ............ → CSV + JSON + Markdown report
```

### Scoring at a glance

| Component | Weight | Source |
|---|---:|---|
| Skill coverage | 0.35 | Deterministic — taxonomy match against the JD's skills |
| Semantic similarity | 0.20 | Deterministic — TF-IDF cosine, implemented from scratch |
| Experience fit | 0.20 | Deterministic — saturating curve over employment dates |
| LLM judgment | 0.15 | LLM — depth, seniority, red flags (`--mode llm` only) |
| Education fit | 0.10 | Deterministic — degree level vs requirement |

**In offline mode the score is 100% deterministic** — the four remaining
weights renormalise to 1.0, so offline runs are not uniformly penalised.

In `--mode llm` the LLM holds 15% of the weight directly. Being precise about
this: it also contributes indirectly, because LLM-extracted skills are unioned
into the candidate's skill list and so can move `skill_coverage` too. The
deterministic *floor* — what the system produces with the LLM switched off
entirely — is a complete, defensible shortlist, and that is the property that
matters.

Full detail, including the maths and the reasoning behind every weight:
**[`docs/SCORING.md`](docs/SCORING.md)**.

### Why not just ask an LLM to score each resume?

Because the output has to be defensible. LLM-only scoring is not reproducible
across runs, not falsifiable ("7/10" cannot be checked; "missing Docker" can),
biased by resume length and ordering, and unavailable the moment an API key or
rate limit fails.

So the LLM here is a *component*, not the system: it carries 15% of the weight
and is asked only for the judgment that keyword matching genuinely cannot
supply. The rule-based extractor always runs, even in LLM mode, and LLM output
is merged over that baseline — a hallucinated or empty response degrades the
result instead of destroying it.

Full reasoning, plus every known failure mode:
**[`docs/TRADEOFFS.md`](docs/TRADEOFFS.md)**.

---

## Repository layout

```
├── run.py                       CLI entry point
├── src/
│   ├── config.py                every tunable number, in one place
│   ├── schema.py                typed records (JobDescription, ResumeProfile, …)
│   ├── text_utils.py            tokenisation shared by every stage
│   ├── parsing.py               PDF / DOCX / TXT → text
│   ├── taxonomy.py              canonical skill matching
│   ├── similarity.py            TF-IDF + BM25, from scratch on numpy
│   ├── extraction.py            rule-based + LLM extraction, and the merge policy
│   ├── llm.py                   provider-agnostic client (4 backends, no SDKs)
│   ├── scoring.py               the five components and their combination
│   └── agent.py                 orchestration + output writers
├── data/
│   ├── job_description.md       the sample JD
│   ├── skills_taxonomy.json     ~40 canonical skills and their surface forms
│   └── resumes/                 13 sample resumes (5 PDF, 4 DOCX, 4 TXT)
├── outputs/                     committed sample output
├── docs/
│   ├── SCORING.md               the scoring method note
│   └── TRADEOFFS.md             design decisions, failures, next steps
├── scripts/
│   └── make_sample_resumes.py   regenerates the sample corpus
└── tests/                       68 unittest tests, no network
```

## Deliverables checklist

| Required | Where |
|---|---|
| A Job Description | [`data/job_description.md`](data/job_description.md) |
| A folder of sample resumes | [`data/resumes/`](data/resumes/) — 13 files, 3 formats |
| Ranked output (CSV/JSON) | [`outputs/ranked_candidates.csv`](outputs/ranked_candidates.csv) · [`.json`](outputs/ranked_candidates.json) |
| Note explaining the scoring method | [`docs/SCORING.md`](docs/SCORING.md) |
| Handles 10+ resumes in one run | 13 in the sample corpus |
| Parses PDF / DOCX / Text | pypdf · python-docx · plain text |
| Tradeoff notes | [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) |

## Dependencies

`numpy`, `pandas`, `pypdf`, `python-docx`, `requests`, `python-dotenv`.
No scikit-learn — TF-IDF, cosine similarity and BM25 are implemented directly
in [`src/similarity.py`](src/similarity.py), which keeps the ranking maths
inspectable and drops a large transitive dependency. `reportlab` is a dev-only
dependency used to regenerate the sample PDFs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
| `GROQ_API_KEY is not set` | Either add it to `.env`, or just use the default `python run.py` (offline) |
| `no supported resume files found` | `--resumes` must point at a folder containing `.pdf`/`.docx`/`.txt`/`.md` |
| A candidate shows 0 skills | Likely a scanned image PDF — check the `warnings` column in the CSV; OCR is not implemented |
