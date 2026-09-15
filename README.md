# AI Job Match Analyzer

Compares a resume against a job description, computes a match score, and
identifies missing skills and areas for improvement.

## Architecture

This project uses a polyglot architecture: a Next.js frontend and a Python
ML microservice, communicating over HTTP.

```
job-match-analyzer/
├── frontend/          # Next.js + Tailwind CSS + Supabase
│   └── app/
│       ├── page.tsx               # Upload form + results
│       ├── components/
│       │   └── ResultsPanel.tsx   # Score bars, skill pills, feedback
│       ├── api/analyze/route.ts   # Proxies requests to the ML microservice
│       └── lib/supabaseClient.ts  # Supabase client (for history/auth)
├── ml-service/        # Python FastAPI microservice
│   ├── main.py         # API endpoint
│   ├── parsing.py      # PDF/DOCX text extraction (PyMuPDF, python-docx)
│   ├── scoring.py       # TF-IDF + sentence-transformer embedding scoring, skill extraction
│   └── feedback.py     # Gemini API call for natural-language feedback
└── supabase_schema.sql # Table + RLS policy for storing analysis history
```

**Why two services?** The scoring logic (embeddings, TF-IDF, skill matching)
is genuine ML work best done in Python. The feedback/explanation text is
generated separately via the Gemini API. Keeping these as distinct modules
makes it easy to explain and evaluate each piece independently — and to swap
either one out later.

## Setup

### 1. ML microservice

```bash
cd ml-service
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# add your GEMINI_API_KEY to .env
uvicorn main:app --reload --port 8000
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# ML_SERVICE_URL defaults to http://localhost:8000, adjust if needed
# add your Supabase project URL + anon key
npm run dev
```

Visit http://localhost:3000.

### 3. Supabase (optional, for history/auth)

Create a project at https://supabase.com, then run `supabase_schema.sql`
in the SQL editor to create the `analyses` table.

## How it works

1. User uploads a resume (PDF/DOCX) and pastes a job description.
2. The Next.js API route forwards the file + text to the FastAPI service.
3. `parsing.py` extracts raw text from the resume.
4. `scoring.py` computes two scores:
   - TF-IDF + cosine similarity (classical keyword-overlap baseline)
   - Sentence-embedding cosine similarity (semantic match, primary score)
   It also extracts skills from both texts against a curated taxonomy to
   find matched and missing skills.
5. `feedback.py` sends the scores + skill gaps to Gemini, which generates
   a short, specific, encouraging feedback paragraph.
6. Results are displayed with score bars, skill pills, and the feedback text.

## Possible extensions

- Expand `SKILLS_TAXONOMY` in `scoring.py` with a real skills dataset
  (O*NET, ESCO, or a scraped list) instead of the small starter set.
- Add Supabase auth so users can save and revisit past analyses.
- Break the match score into sub-scores (skills, experience, education)
  instead of one overall number.
- Deploy the ML microservice on Render/Railway/HF Spaces and the frontend
  on Vercel.
