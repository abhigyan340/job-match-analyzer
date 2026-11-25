"""
Unit tests for skills.py.

No network, no Gemini, no sentence-transformers — pure Python only.
Run with:  pytest ml-service/tests/test_skills.py  (from repo root, venv active)
Or:        python -m pytest  (from ml-service/ with venv active)
"""

import sys
import os

# Allow running directly with `python test_skills.py` from the tests/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills import (
    normalize_skill,
    match_skills,
    compute_overall_score,
    WEIGHT_SKILLS,
    WEIGHT_SEMANTIC,
    WEIGHT_EXPERIENCE,
    WEIGHT_EDUCATION,
)


# ---------------------------------------------------------------------------
# normalize_skill
# ---------------------------------------------------------------------------

class TestNormalizeSkill:
    def test_lowercase(self):
        assert normalize_skill("Python") == "python"

    def test_strips_whitespace(self):
        assert normalize_skill("  react  ") == "react"

    def test_strips_leading_trailing_punctuation(self):
        # A period at the start/end should be stripped
        assert normalize_skill(".python.") == "python"

    def test_keeps_internal_dot(self):
        # "next.js" internal dot must be preserved
        assert normalize_skill("Next.js") == "next.js"

    def test_alias_js(self):
        assert normalize_skill("JS") == "javascript"

    def test_alias_ts(self):
        assert normalize_skill("TS") == "typescript"

    def test_alias_node(self):
        assert normalize_skill("node") == "node.js"

    def test_alias_nodejs(self):
        assert normalize_skill("NodeJS") == "node.js"

    def test_alias_react_js(self):
        assert normalize_skill("react.js") == "react"

    def test_alias_postgres(self):
        assert normalize_skill("Postgres") == "postgresql"

    def test_alias_ml(self):
        assert normalize_skill("ML") == "machine learning"

    def test_alias_sklearn(self):
        assert normalize_skill("sklearn") == "scikit-learn"

    def test_alias_k8s(self):
        assert normalize_skill("k8s") == "kubernetes"

    def test_no_alias_passthrough(self):
        assert normalize_skill("docker") == "docker"

    def test_unknown_skill_passthrough(self):
        assert normalize_skill("somedomain-specific-tool") == "somedomain-specific-tool"


# ---------------------------------------------------------------------------
# match_skills
# ---------------------------------------------------------------------------

class TestMatchSkills:
    def test_exact_match_via_resume_skills(self):
        matched, missing = match_skills(
            required_skills=["python"],
            resume_skills=["python"],
            resume_text="",
        )
        assert "python" in matched
        assert missing == []

    def test_alias_match(self):
        # JD says "JS", resume says "javascript" — they should normalize to same
        matched, missing = match_skills(
            required_skills=["JS"],
            resume_skills=["javascript"],
            resume_text="",
        )
        assert "JS" in matched
        assert missing == []

    def test_whole_word_text_match(self):
        # Skill not in resume_skills list but present in resume text
        matched, missing = match_skills(
            required_skills=["docker"],
            resume_skills=[],
            resume_text="Deployed services using Docker containers.",
        )
        assert "docker" in matched
        assert missing == []

    def test_no_partial_word_match(self):
        # "r" should NOT match inside "director"
        matched, missing = match_skills(
            required_skills=["r"],
            resume_skills=[],
            resume_text="I was the director of engineering.",
        )
        # "r" appears only inside "director", not as a standalone word
        assert "r" in missing

    def test_missing_skill(self):
        matched, missing = match_skills(
            required_skills=["kubernetes"],
            resume_skills=["docker"],
            resume_text="Worked with Docker.",
        )
        assert "kubernetes" in missing
        assert matched == []

    def test_mixed_matched_and_missing(self):
        matched, missing = match_skills(
            required_skills=["python", "kubernetes", "react"],
            resume_skills=["python", "react"],
            resume_text="Proficient in Python and React.",
        )
        assert "python" in matched
        assert "react" in matched
        assert "kubernetes" in missing

    def test_empty_required(self):
        matched, missing = match_skills(
            required_skills=[],
            resume_skills=["python"],
            resume_text="Python developer.",
        )
        assert matched == []
        assert missing == []

    def test_results_are_sorted(self):
        matched, missing = match_skills(
            required_skills=["react", "python", "aws"],
            resume_skills=["python", "react"],
            resume_text="",
        )
        assert matched == sorted(matched)
        assert missing == sorted(missing)

    def test_phrase_skill_match_in_text(self):
        # Multi-word skill like "machine learning" matched in resume text
        matched, missing = match_skills(
            required_skills=["machine learning"],
            resume_skills=[],
            resume_text="Developed machine learning models for NLP tasks.",
        )
        assert "machine learning" in matched

    def test_alias_via_text(self):
        # JD says "postgres"; resume text says "PostgreSQL" (lowercase match)
        matched, missing = match_skills(
            required_skills=["postgres"],
            resume_skills=[],
            resume_text="Managed a PostgreSQL database.",
        )
        # normalize("postgres") -> "postgresql"; check whole-word in text
        assert "postgres" in matched


# ---------------------------------------------------------------------------
# compute_overall_score
# ---------------------------------------------------------------------------

class TestComputeOverallScore:
    def test_all_components_present(self):
        # With all components: 40*80 + 30*60 + 20*70 + 10*50 = 32+18+14+5 = 69
        score = compute_overall_score(
            skills=80, semantic=60, experience=70, education=50
        )
        expected = round(
            80 * WEIGHT_SKILLS +
            60 * WEIGHT_SEMANTIC +
            70 * WEIGHT_EXPERIENCE +
            50 * WEIGHT_EDUCATION,
            1,
        )
        assert score == expected

    def test_weights_sum_to_one(self):
        total = WEIGHT_SKILLS + WEIGHT_SEMANTIC + WEIGHT_EXPERIENCE + WEIGHT_EDUCATION
        assert abs(total - 1.0) < 1e-9

    def test_drop_experience_renormalizes(self):
        # experience=None — remaining weights: 0.40+0.30+0.10 = 0.80
        score = compute_overall_score(
            skills=100, semantic=100, experience=None, education=100
        )
        # With all remaining at 100, normalized result must still be 100
        assert score == 100.0

    def test_drop_education_renormalizes(self):
        score = compute_overall_score(
            skills=100, semantic=100, experience=100, education=None
        )
        assert score == 100.0

    def test_skills_none(self):
        # Only semantic, experience, education — weights 0.30+0.20+0.10=0.60
        score = compute_overall_score(
            skills=None, semantic=60, experience=80, education=40
        )
        remaining_w = WEIGHT_SEMANTIC + WEIGHT_EXPERIENCE + WEIGHT_EDUCATION
        expected = round(
            60 * (WEIGHT_SEMANTIC / remaining_w) +
            80 * (WEIGHT_EXPERIENCE / remaining_w) +
            40 * (WEIGHT_EDUCATION / remaining_w),
            1,
        )
        assert score == expected

    def test_only_semantic(self):
        # Only one component — must equal that score
        score = compute_overall_score(
            skills=None, semantic=73, experience=None, education=None
        )
        assert score == 73.0

    def test_all_none_raises(self):
        import pytest
        with pytest.raises(ValueError, match="all components are None"):
            compute_overall_score(None, None, None, None)

    def test_zero_scores(self):
        score = compute_overall_score(
            skills=0, semantic=0, experience=0, education=0
        )
        assert score == 0.0

    def test_hundred_scores(self):
        score = compute_overall_score(
            skills=100, semantic=100, experience=100, education=100
        )
        assert score == 100.0

    def test_returns_float(self):
        score = compute_overall_score(
            skills=50, semantic=50, experience=50, education=50
        )
        assert isinstance(score, float)
