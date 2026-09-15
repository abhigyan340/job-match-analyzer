"""
Computes resume-vs-JD match scores using two approaches:
1. TF-IDF + cosine similarity (classical baseline)
2. Sentence embeddings + cosine similarity (semantic match, primary score)

Also does lightweight skill extraction against a curated skills list.
"""

import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# A small starter skills taxonomy. Expand this with a real skills dataset
# (e.g. O*NET, LinkedIn Skills, or a scraped list) for production use.
SKILLS_TAXONOMY = {
    "python", "java", "javascript", "typescript", "c++", "sql", "r",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "react", "next.js", "node.js", "fastapi", "django", "flask",
    "docker", "kubernetes", "aws", "gcp", "azure", "git",
    "postgresql", "mongodb", "redis", "supabase", "firebase",
    "rest api", "graphql", "microservices", "ci/cd", "agile",
    "data analysis", "data engineering", "mlops", "llm", "rag",
}


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Loaded once and cached — this is the slow part, so don't reload per-request
    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_tfidf_score(resume_text: str, job_description: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)


def compute_embedding_score(resume_text: str, job_description: str) -> float:
    model = get_embedding_model()
    embeddings = model.encode([resume_text, job_description])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score) * 100, 2)


def compute_match_score(resume_text: str, job_description: str) -> dict:
    return {
        "tfidf_score": compute_tfidf_score(resume_text, job_description),
        "embedding_score": compute_embedding_score(resume_text, job_description),
    }


def extract_skills(text: str) -> set[str]:
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_TAXONOMY:
        # word-boundary-ish match so "r" doesn't match inside "director"
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found
