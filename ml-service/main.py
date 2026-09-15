"""
AI Job Match Analyzer - ML Microservice
Handles resume/JD parsing, embedding-based match scoring, and skill-gap analysis.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parsing import extract_text_from_resume
from scoring import compute_match_score, extract_skills
from feedback import generate_feedback

app = FastAPI(title="Job Match Analyzer ML Service")

# Allow the Next.js frontend (localhost during dev, your deployed domain in prod) to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend URL before deploying
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
    job_description: str = Form(...),
):
    """
    Accepts a resume file (PDF or DOCX) + a job description string.
    Returns a match score, matched/missing skills, and LLM-generated feedback.
    """
    if resume_file.content_type not in (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        raise HTTPException(status_code=400, detail="Resume must be a PDF or DOCX file")

    resume_bytes = await resume_file.read()
    resume_text = extract_text_from_resume(resume_bytes, resume_file.filename)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from resume")

    scores = compute_match_score(resume_text, job_description)
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    try:
        feedback = generate_feedback(
            resume_text=resume_text,
            job_description=job_description,
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
