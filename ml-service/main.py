"""
AI Job Match Analyzer - ML Microservice
POST /analyze  — resume + JD (pasted text or uploaded file) → full analysis.
GET  /health   — liveness probe.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from parsing import extract_text_from_file
from scoring import compute_match_score
from skills import (
    match_skills,
    normalize_skill,
    compute_overall_score,
    _extract_skills_from_text_taxonomy,
    filter_rewrite_suggestions,
)
from feedback import call_gemini_analysis, generate_rewrite_suggestions

app = FastAPI(title="Job Match Analyzer ML Service")

# Allow the Next.js frontend to call this service.
# Tighten allow_origins to your deployed frontend URL before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models (mirror the TypeScript types in frontend/app/lib/types.ts)
# ---------------------------------------------------------------------------

class SubScore(BaseModel):
    score: Optional[float]
    explanation: str


class SubScores(BaseModel):
    skills:     SubScore
    semantic:   SubScore
    experience: SubScore
    education:  SubScore


class RewriteSuggestion(BaseModel):
    target_skill: str
    original:     str   # empty string means the original could not be verified
    suggested:    str
    reason:       str


class AnalyzeResponse(BaseModel):
    overall_score:            float
    sub_scores:               SubScores
    tfidf_score:              float
    matched_skills:           list[str]
    missing_skills:           list[str]
    missing_preferred_skills: list[str]
    rewrite_suggestions:      list[RewriteSuggestion]
    feedback:                 str
    ai_used:                  bool
    job_description_text:     str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume_file:     UploadFile          = File(...),
    job_description: Optional[str]       = Form(default=None),
    jd_file:         Optional[UploadFile] = File(default=None),
):
    """
    Accepts a resume (PDF/DOCX) plus EITHER a pasted job_description string
    OR an uploaded jd_file (PDF/DOCX). If both arrive, jd_file takes priority.
    Returns the full analysis response defined in AnalyzeResponse.
    """

    # ------------------------------------------------------------------
    # 1. Parse resume
    # ------------------------------------------------------------------
    resume_bytes = await resume_file.read()
    try:
        resume_text = extract_text_from_file(
            resume_bytes, resume_file.filename or "", label="resume"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not read any text from the resume. "
                   "Make sure the PDF/DOCX contains selectable text.",
        )

    # ------------------------------------------------------------------
    # 2. Parse job description
    # ------------------------------------------------------------------
    jd_text: str = ""

    if jd_file is not None:
        jd_bytes = await jd_file.read()
        try:
            jd_text = extract_text_from_file(
                jd_bytes, jd_file.filename or "", label="job description"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not jd_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not read any text from the job description file. "
                       "Make sure the PDF/DOCX contains selectable text.",
            )
    elif job_description and job_description.strip():
        jd_text = job_description.strip()
    else:
        raise HTTPException(
            status_code=400,
            detail="A job description is required. Paste text into job_description "
                   "or upload a file via jd_file.",
        )

    # ------------------------------------------------------------------
    # 3. Similarity scores (always computed, never depend on Gemini)
    # ------------------------------------------------------------------
    scores = compute_match_score(resume_text, jd_text)
    semantic_score: float = scores["embedding_score"]
    tfidf_score:    float = scores["tfidf_score"]

    # ------------------------------------------------------------------
    # 4. Gemini structured extraction (with fallback)
    # ------------------------------------------------------------------
    gemini_result, ai_used = call_gemini_analysis(resume_text, jd_text)

    if ai_used:
        # Gemini returned skill lists — use them for matching
        required_skills  = gemini_result["required_skills"]
        preferred_skills = gemini_result["preferred_skills"]
        resume_skills    = gemini_result["resume_skills"]

        exp_data = gemini_result["experience"]
        edu_data = gemini_result["education"]
        experience_score: Optional[float] = float(exp_data["score"])
        education_score:  Optional[float] = float(edu_data["score"])
        experience_explanation = exp_data["explanation"]
        education_explanation  = edu_data["explanation"]
        feedback = gemini_result["feedback"]
    else:
        # Fallback: use the taxonomy extractor for skills
        taxonomy_resume = _extract_skills_from_text_taxonomy(resume_text)
        taxonomy_jd     = _extract_skills_from_text_taxonomy(jd_text)

        required_skills  = sorted(taxonomy_jd)
        preferred_skills = []
        resume_skills    = sorted(taxonomy_resume)

        experience_score = None
        education_score  = None
        experience_explanation = "Not available (AI analysis unavailable)"
        education_explanation  = "Not available (AI analysis unavailable)"
        feedback = gemini_result["feedback"]  # "AI feedback is temporarily unavailable."

    # ------------------------------------------------------------------
    # 5. Skill matching (always done in code, never by the LLM)
    # ------------------------------------------------------------------
    matched, missing = match_skills(required_skills, resume_skills, resume_text)

    # Preferred skills: missing ones the resume doesn't cover
    _, missing_preferred = match_skills(preferred_skills, resume_skills, resume_text)

    # ------------------------------------------------------------------
    # 6. Skills sub-score
    # ------------------------------------------------------------------
    if required_skills:
        skills_score: Optional[float] = round(
            len(matched) / len(required_skills) * 100, 1
        )
        skills_explanation = (
            f"Matched {len(matched)} of {len(required_skills)} required skills"
        )
    else:
        skills_score = None
        skills_explanation = "No required skills listed in the job description"

    # ------------------------------------------------------------------
    # 7. Overall score (weighted, drops None components and re-normalizes)
    # ------------------------------------------------------------------
    overall = compute_overall_score(
        skills=skills_score,
        semantic=semantic_score,
        experience=experience_score,
        education=education_score,
    )

    # ------------------------------------------------------------------
    # 8. Rewrite suggestions (second Gemini call, only when AI is available
    #    and there are skills to address — failure returns [] silently)
    # ------------------------------------------------------------------
    all_missing = missing + missing_preferred   # required + preferred gaps
    rewrite_suggestions: list[RewriteSuggestion] = []

    if ai_used and all_missing:
        raw_suggestions = generate_rewrite_suggestions(
            resume_text=resume_text,
            missing_skills=all_missing,
        )
        # filter_rewrite_suggestions enforces honesty rules and caps at 5
        filtered = filter_rewrite_suggestions(
            raw_items=raw_suggestions,
            missing_skills=all_missing,
            resume_text=resume_text,
        )
        rewrite_suggestions = [RewriteSuggestion(**s) for s in filtered]

    # ------------------------------------------------------------------
    # 9. Build and return response
    # ------------------------------------------------------------------
    return AnalyzeResponse(
        overall_score=overall,
        sub_scores=SubScores(
            skills=SubScore(
                score=skills_score,
                explanation=skills_explanation,
            ),
            semantic=SubScore(
                score=round(semantic_score, 1),
                explanation="Embedding similarity of the full resume vs job description",
            ),
            experience=SubScore(
                score=experience_score,
                explanation=experience_explanation,
            ),
            education=SubScore(
                score=education_score,
                explanation=education_explanation,
            ),
        ),
        tfidf_score=round(tfidf_score, 1),
        matched_skills=matched,
        missing_skills=missing,
        missing_preferred_skills=missing_preferred,
        rewrite_suggestions=rewrite_suggestions,
        feedback=feedback,
        ai_used=ai_used,
        job_description_text=jd_text,
    )
