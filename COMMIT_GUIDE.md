# How to commit this (read before you push)

The challenge scores a **steady commit history inside the 24-hour window**, and
one giant "initial commit" reads badly. Commit in the order below — it mirrors
the order the project was actually built, and each step is independently
runnable.

```bash
git init && git add .gitignore requirements.txt .env.example
git commit -m "chore: project scaffold, pinned deps, env template"

git add data/job_description.md data/skills_taxonomy.json scripts/
git commit -m "data: job description, skills taxonomy, resume corpus generator"

git add data/resumes/
git commit -m "data: 13 sample resumes across PDF/DOCX/TXT"

git add src/text_utils.py src/similarity.py tests/test_similarity.py
git commit -m "feat: TF-IDF and BM25 from scratch on numpy"

git add src/config.py src/schema.py src/parsing.py src/taxonomy.py
git commit -m "feat: config, typed records, multi-format parsing, skill matching"

git add src/extraction.py src/llm.py tests/test_extraction.py
git commit -m "feat: rule-based extraction + provider-agnostic LLM backend"

git add src/scoring.py tests/test_scoring.py
git commit -m "feat: five-component weighted scoring with hard requirement gate"

git add src/agent.py run.py tests/test_pipeline.py src/__init__.py tests/__init__.py
git commit -m "feat: orchestration, CLI, CSV/JSON/Markdown output"

git add outputs/
git commit -m "docs: committed sample output for the reference corpus"

git add README.md docs/
git commit -m "docs: README, scoring method note, tradeoff notes"

git branch -M main
git remote add origin https://github.com/<your-username>/resume-screening-agent.git
git push -u origin main
```

Delete this file before pushing: `rm COMMIT_GUIDE.md`

---

## Before you submit — you must be able to explain every line

The rules say so, and a reviewer will test it. Read these five in order; they
are where the actual thinking is:

1. `src/similarity.py` — why smoothed IDF, why sublinear TF, why L2 before cosine.
2. `src/scoring.py` — why five components, why the experience curve saturates.
3. `src/config.py` — the weights and the hard gate, with the reasoning inline.
4. `src/extraction.py` — union-span vs summing years; the degree-regex comments.
5. `docs/TRADEOFFS.md` — every claim in here should be one you'd defend out loud.

Three questions a sharp interviewer will ask, all answered in the code comments:

- *"Why not just ask an LLM to score each resume?"* → README, "Why not just ask
  an LLM", and TRADEOFFS §"The central design decision".
- *"Your semantic scores are pool-relative. What breaks?"* → TRADEOFFS §4.
- *"Walk me through `detect_degree_level` on 'I would be delighted to apply.'"*
  → `src/extraction.py`, the DEGREE_PATTERNS comment block.
