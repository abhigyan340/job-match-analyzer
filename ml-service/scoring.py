"""
Computes resume-vs-JD similarity scores using two approaches:
1. TF-IDF cosine similarity  — classical keyword-overlap baseline (diagnostic).
2. Sentence-embedding cosine similarity — semantic match, used in overall score.

Skill extraction using the taxonomy fallback lives in skills.py.
"""

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Loaded once and cached — model download/load is slow; don't repeat it
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
        "tfidf_score":     compute_tfidf_score(resume_text, job_description),
        "embedding_score": compute_embedding_score(resume_text, job_description),
    }
