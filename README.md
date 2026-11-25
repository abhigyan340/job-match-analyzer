# AI Job Match Analyzer

Compares a resume against a job description, produces an explainable match
score broken into four sub-scores, identifies matched and missing skills, and
suggests targeted resume rewrites grounded in the candidate's existing
experience.

## Features

- **Resume + job description upload** — resume accepts PDF or DOCX; job
  description can be pasted as text or uploaded as PDF or DOCX. Files are
  parsed by extension (not browser MIME type). Size limit: 5 MB per file.
- **Gemini structured skill extraction** — one Gemini call returns the
  skills evidenced in the resume, the required and preferred skills from the
  job description, experience and education sub-scores, and a feedback
  paragraph, all as strict JSON. If Gemini is unavailable for any reason the
  service falls back to a local taxonomy extractor and still returns HTTP 200
  with the similarity scores.
- **Four sub-scores** — skills match (computed in code from the extracted
  skill lists), semantic similarity (sentence-transformer embeddings), and
  experience and education fit (LLM-assessed). These are combined into a
  single overall score. TF-IDF cosine similarity is also returned as a
  diagnostic but is not part of the overall score.
- **Resume rewrite suggestions** — a second Gemini call proposes up to five
  rewrites that surface missing skills using only experience already present
  in the resume. A code-side filter verifies that the "original" line is
  found verbatim in the resume text and that the target skill is in the
  code-computed missing list. Suggestions that fail either check are adjusted
  or dropped before being returned.

## Architecture

This project uses a polyglot architecture: a Next.js frontend and a Python
ML microservice communicating over HTTP.

```
job-match-analyzer/
├── frontend/                        # Next.js 14, Tailwind CSS
│   └── app/
│       ├── page.tsx                 # Upload form (resume + JD paste/upload)
│       ├── components/
│       │   └── ResultsPanel.tsx     # Score ring, sub-score bars, skill pills,
│       │                            #   rewrite suggestion cards, feedback
│       ├── api/analyze/route.ts     # Proxy to the ML microservice
│       └── lib/
│           ├── types.ts             # Shared TypeScript types (AnalysisResult, etc.)
│           └── supabaseClient.ts    # Supabase client (not yet wired up)
├── ml-service/                      # Python 3.13, FastAPI
│   ├── main.py                      # POST /analyze endpoint + Pydantic response models
│   ├── parsing.py                   # PDF/DOCX text extraction (PyMuPDF, python-docx)
│   ├── scoring.py                   # TF-IDF + sentence-transformer embedding scores
│   ├── skills.py                    # Skill normalisation, alias map, match_skills(),
│   │                                #   compute_overall_score(), rewrite-suggestion filters
│   ├── feedback.py                  # Gemini calls (analysis + rewrite suggestions)
│   ├── requirements.txt
│   ├── requirements-dev.txt         # pytest (dev only)
│   └── tests/
│       └── test_skills.py           # 55 unit tests, no network required
└── supabase_schema.sql              # Schema stub (not yet in use)
```

**Why two services?** The scoring logic (embeddings, TF-IDF, skill matching)
is Python ML work. The LLM calls are kept in a separate module so the
scoring path works independently and can be tested without a Gemini key.

## Setup

### 1. ML microservice

```bash
cd ml-service
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
uvicorn main:app --reload --port 8000
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# ML_SERVICE_URL defaults to http://localhost:8000 — adjust if needed
npm run dev
```

Visit http://localhost:3000.

## How it works

1. The user uploads a resume (PDF/DOCX) and provides a job description —
   either pasted as text or uploaded as a file.
2. The Next.js API route forwards the multipart form to the FastAPI service.
3. `parsing.py` extracts text from the resume and, if uploaded, the JD file.
   The parser is chosen by file extension, not by the browser-sent MIME type.
4. `scoring.py` computes two similarity scores independently of Gemini:
   - TF-IDF cosine similarity (keyword-overlap baseline, diagnostic only)
   - Sentence-embedding cosine similarity (semantic match)
5. `feedback.py` makes one Gemini call that returns, as strict JSON: the
   skills evidenced in the resume, required and preferred skills from the JD,
   experience and education sub-scores with one-sentence explanations, and a
   feedback paragraph.
6. `skills.py` runs skill matching entirely in code: it normalises skill
   names, applies an alias map (e.g. "JS" → "javascript"), and checks for
   whole-word matches in the resume text. The LLM never makes the
   matched/missing decision.
7. If Gemini is available and there are missing skills, `feedback.py` makes
   a second call to generate rewrite suggestions. The raw output is passed
   through `filter_rewrite_suggestions()` before use.
8. The frontend displays the overall score ring, four sub-score bars with
   explanations, matched/missing/preferred skill pills, rewrite suggestion
   cards, and the feedback paragraph.

## Scoring formula

```
overall_score = skills × 0.40
              + semantic × 0.30
              + experience × 0.20
              + education × 0.10
```

All components are on a 0–100 scale.

| Component  | Weight | Source |
|---|---|---|
| Skills match | 40% | `matched_required / total_required × 100` (code) |
| Semantic similarity | 30% | Sentence-transformer cosine similarity × 100 |
| Experience fit | 20% | Gemini score (0–100) |
| Education fit | 10% | Gemini score (0–100) |

If any component is `null` (e.g. Gemini unavailable, or the JD lists no
required skills), that component is dropped and the remaining weights are
re-normalized so they still sum to 100%. For example, if both experience and
education are null, the surviving weights are skills 57% and semantic 43%.

The overall score is computed in `skills.py` → `compute_overall_score()`.

## Fallback behavior when Gemini is unavailable

If the Gemini call fails for any reason (API key missing, 429 rate limit,
404 model not found, invalid JSON, network timeout):

- A 429 triggers one retry after a 5-second backoff. All other errors fall
  back immediately.
- A 404 logs a human-readable hint:
  `Model <name> is not available. Set GEMINI_MODEL in ml-service/.env to a current model.`
- Skill lists fall back to `SKILLS_TAXONOMY` in `skills.py` — a curated set
  of ~40 common technology skills matched by whole-word regex.
- Experience and education sub-scores are set to `null` and excluded from
  the overall score calculation.
- Feedback is set to `"AI feedback is temporarily unavailable."`
- Rewrite suggestions are set to `[]`.
- The endpoint still returns HTTP 200 with all similarity scores.
- The frontend shows an amber notice: "AI analysis was unavailable —
  showing similarity scores only."

The model name is read from the `GEMINI_MODEL` environment variable
(default `"gemini-3.6-flash"`). To change it, set `GEMINI_MODEL` in
`ml-service/.env` without touching any code.

## Running the tests

The unit tests cover skill normalisation, alias resolution, `match_skills()`,
`compute_overall_score()` with and without null components,
`verify_original()`, and `filter_rewrite_suggestions()`. No network, Gemini
key, or ML model is required.

```bash
cd ml-service
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements-dev.txt
pytest tests/test_skills.py -v
# Expected: 55 passed
```

## Known limitations

- **Semantic score is uncalibrated.** The sentence-transformer cosine
  similarity is returned as-is (multiplied by 100). It is not calibrated
  against human judgements, so a score of 65 does not mean "65% of the way
  to a perfect match" in any absolute sense.
- **Experience and education sub-scores are LLM-assessed.** They reflect
  the model's interpretation of the resume and job description texts and can
  be inconsistent across runs or when the JD is vague.
- **Rewrite suggestions require user judgement.** The filter confirms that
  the "original" line appears verbatim in the resume and that the target
  skill is in the code-computed missing list, but it cannot verify whether
  the candidate actually used the suggested skill in that work. Always review
  suggestions before adding them to your resume.
