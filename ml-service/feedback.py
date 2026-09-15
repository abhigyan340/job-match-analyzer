"""
Generates natural-language feedback (why the score is what it is, and how
to improve) using the Gemini API. Keeps the ML scoring (scoring.py) separate
from this LLM-generated explanation layer, by design.
"""

import os
import json
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"

PROMPT_TEMPLATE = """You are a career coach helping a candidate improve their resume for a specific job.

Resume text:
{resume_text}

Job description:
{job_description}

The candidate's resume already matches these skills the job asks for: {matched_skills}
The job asks for these skills that were NOT found in the resume: {missing_skills}
The computed semantic match score is {match_score}/100.

Write a short, encouraging, and specific feedback paragraph (4-6 sentences) that:
1. Explains what's working well in the match
2. Highlights the 2-3 most important missing skills to address first
3. Gives one concrete, actionable suggestion for improving the resume for this specific role

Do not repeat the raw score back numerically. Write in plain text, no markdown."""


def generate_feedback(
    resume_text: str,
    job_description: str,
    matched_skills: list[str],
    missing_skills: list[str],
    match_score: float,
) -> str:
    prompt = PROMPT_TEMPLATE.format(
        resume_text=resume_text[:3000],  # keep prompt size reasonable
        job_description=job_description[:2000],
        matched_skills=", ".join(matched_skills) or "none detected",
        missing_skills=", ".join(missing_skills) or "none detected",
        match_score=match_score,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()
