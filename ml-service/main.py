"""
AI Job Match Analyzer - ML Microservice
POST /analyze  — accepts a resume file plus either pasted JD text or a JD file.
GET  /health   — liveness probe.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from parsing import extract_text_from_file
from scoring import compute_match_score, extract_skills
from feedback import generate_feedback

app = FastAPI(title="Job Match Analyzer ML Service")

# Allow the Next.js frontend to call this service.
# Tighten allow_origins to your deployed frontend URL before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    match_score: float
    tfidf_score: float
    embedding_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    feedback: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume_file: UploadFile = File(...),
    job_description: Optional[str] = Form(default=None),
    jd_file: Optional[UploadFile] = File(default=None),
):
    """
    Accepts a resume file (PDF or DOCX) plus EITHER a pasted job_description
    string OR an uploaded jd_file (PDF or DOCX).

    If both jd_file and job_description are provided, jd_file takes priority.
    """

    # --- Parse resume ---
    # Use the file extension to decide the parser, NOT content_type.
    # Windows browsers send inconsistent MIME types for the same file.
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

    # --- Parse job description ---
    jd_text: str = ""

    if jd_file is not None:
        # Uploaded JD file takes priority over pasted text
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
            detail="A job description is required. "
                   "Either paste text into the job_description field "
                   "or upload a file via jd_file.",
        )

    # --- Score + skills ---
    scores = compute_match_score(resume_text, jd_text)
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    # --- Feedback (Gemini) ---
    try:
        feedback = generate_feedback(
            resume_text=resume_text,
            job_description=jd_text,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_score=scores["embedding_score"],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Feedback generation failed: {e}")

    return AnalyzeResponse(
        match_score=scores["embedding_score"],
        tfidf_score=scores["tfidf_score"],
        embedding_score=scores["embedding_score"],
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        feedback=feedback,
    )
