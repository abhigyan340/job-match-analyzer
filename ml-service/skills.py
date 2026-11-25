"""
Pure-Python skill utilities: normalization, matching, score computation,
and rewrite-suggestion validation.
No network calls, no ML libraries — safe to unit-test in isolation.
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Skill taxonomy (fallback when Gemini is unavailable)
# ---------------------------------------------------------------------------

SKILLS_TAXONOMY: set[str] = {
    "python", "java", "javascript", "typescript", "c++", "sql", "r",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "react", "next.js", "node.js", "fastapi", "django", "flask",
    "docker", "kubernetes", "aws", "gcp", "azure", "git",
    "postgresql", "mongodb", "redis", "supabase", "firebase",
    "rest api", "graphql", "microservices", "ci/cd", "agile",
    "data analysis", "data engineering", "mlops", "llm", "rag",
}

# ---------------------------------------------------------------------------
# Alias map — maps common abbreviations/variants to their canonical form.
# Keys and values are both lowercase.
# ---------------------------------------------------------------------------

SKILL_ALIASES: dict[str, str] = {
    "js":          "javascript",
    "ts":          "typescript",
    "node":        "node.js",
    "nodejs":      "node.js",
    "react.js":    "react",
    "reactjs":     "react",
    "postgres":    "postgresql",
    "pg":          "postgresql",
    "ml":          "machine learning",
    "sklearn":     "scikit-learn",
    "k8s":         "kubernetes",
    "tf":          "tensorflow",
    "py":          "python",
    "mongo":       "mongodb",
    "vue":         "vue.js",
    "vuejs":       "vue.js",
    "vue.js":      "vue.js",
    "angular":     "angular",
    "angularjs":   "angular",
    "express":     "express.js",
    "expressjs":   "express.js",
    "express.js":  "express.js",
    "gpt":         "llm",
    "openai":      "llm",
}


def normalize_skill(skill: str) -> str:
    """
    Lowercase, strip whitespace, remove trailing/leading punctuation,
    then apply the alias map.

    Examples:
        "Next.js"  -> "next.js"   (kept as-is, no alias)
        "JS"       -> "javascript" (alias)
        "Postgres" -> "postgresql" (alias)
        ".react."  -> "react"      (punctuation stripped then alias)
    """
    # Lowercase and strip surrounding whitespace
    s = skill.lower().strip()
    # Strip leading/trailing punctuation (but keep internal dots like "next.js")
    s = s.strip(".,;:!?\"'()[]{}\\/ ")
    # Apply alias map
    return SKILL_ALIASES.get(s, s)


def _extract_skills_from_text_taxonomy(text: str) -> set[str]:
    """
    Scan text for any skill in SKILLS_TAXONOMY using whole-word matching.
    Returns the set of canonical skill names found.
    """
    text_lower = text.lower()
    found: set[str] = set()
    for skill in SKILLS_TAXONOMY:
        # Escape special regex chars (e.g. "c++", "next.js")
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def match_skills(
    required_skills: list[str],
    resume_skills: list[str],
    resume_text: str,
) -> tuple[list[str], list[str]]:
    """
    Determine which required skills are matched and which are missing.

    A required skill is considered matched when EITHER:
      1. Its normalized form equals the normalized form of any resume skill, OR
      2. Its normalized form appears as a whole-word/phrase in the resume text.

    Matching is done entirely in code — the LLM never makes this decision.

    Args:
        required_skills: skills the job description requires (from Gemini or taxonomy).
        resume_skills:   skills found in the resume (from Gemini or taxonomy).
        resume_text:     raw resume text, used for the whole-word fallback check.

    Returns:
        (matched, missing) — both lists contain the original (un-normalized) skill
        strings from required_skills, sorted alphabetically.
    """
    # Build a set of normalized resume skill strings for fast lookup
    normalized_resume = {normalize_skill(s) for s in resume_skills}
    resume_text_lower = resume_text.lower()

    matched: list[str] = []
    missing: list[str] = []

    for skill in required_skills:
        norm = normalize_skill(skill)

        # Check 1: direct normalized-set membership
        if norm in normalized_resume:
            matched.append(skill)
            continue

        # Check 2: whole-word/phrase match in resume text
        # Escape special regex chars so "c++" and "next.js" work correctly
        pattern = r"(?<![a-z0-9])" + re.escape(norm) + r"(?![a-z0-9])"
        if re.search(pattern, resume_text_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    return sorted(matched), sorted(missing)


# ---------------------------------------------------------------------------
# Overall score computation
# ---------------------------------------------------------------------------

# Named weight constants — documented so the formula is easy to explain.
WEIGHT_SKILLS:     float = 0.40  # Skill match rate (required skills)
WEIGHT_SEMANTIC:   float = 0.30  # Sentence-embedding cosine similarity
WEIGHT_EXPERIENCE: float = 0.20  # LLM-assessed experience relevance
WEIGHT_EDUCATION:  float = 0.10  # LLM-assessed education fit


def compute_overall_score(
    skills:     float | None,
    semantic:   float | None,
    experience: float | None,
    education:  float | None,
) -> float:
    """
    Weighted average of four sub-scores, all on a 0-100 scale.

    Default weights: skills 40%, semantic 30%, experience 20%, education 10%.
    If any component is None (e.g. Gemini unavailable), that component is
    dropped and the remaining weights are re-normalized so they still sum to 1.

    Example with experience=None:
        remaining weights: skills=0.40, semantic=0.30, education=0.10 → sum=0.80
        re-normalized:     skills=0.50, semantic=0.375, education=0.125

    Returns the weighted average rounded to one decimal place.
    Raises ValueError if ALL components are None.
    """
    components = [
        (skills,     WEIGHT_SKILLS),
        (semantic,   WEIGHT_SEMANTIC),
        (experience, WEIGHT_EXPERIENCE),
        (education,  WEIGHT_EDUCATION),
    ]

    # Keep only the components that have a value
    available = [(score, weight) for score, weight in components if score is not None]

    if not available:
        raise ValueError("compute_overall_score: all components are None")

    total_weight = sum(w for _, w in available)
    # Re-normalize weights so they sum to 1
    weighted_sum = sum(score * (weight / total_weight) for score, weight in available)

    return round(weighted_sum, 1)


# ---------------------------------------------------------------------------
# Rewrite-suggestion helpers
# ---------------------------------------------------------------------------

# Max characters for the suggested rewrite and its reason (keeps cards readable)
_MAX_SUGGESTION_CHARS = 300

# Collapse all whitespace runs and strip before comparing, so minor formatting
# differences between the LLM's copy and the original text don't cause misses.
def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).lower()


def verify_original(original: str, resume_text: str) -> bool:
    """
    Return True if `original` (a bullet or line the LLM claims to have copied
    from the resume) actually appears in `resume_text`.

    The check is whitespace- and case-insensitive: runs of whitespace are
    collapsed and both strings are lowercased before comparing.  This handles
    minor line-ending or spacing differences between the LLM's copy and the
    source text without being so loose that unrelated text matches.

    Returns False for empty strings.
    """
    if not original or not original.strip():
        return False
    return _normalize_whitespace(original) in _normalize_whitespace(resume_text)


def filter_rewrite_suggestions(
    raw_items: list[Any],
    missing_skills: list[str],
    resume_text: str,
) -> list[dict]:
    """
    Clean and validate a list of raw rewrite suggestion dicts from Gemini.

    Rules applied in order:
    1. Each item must be a dict with keys target_skill, original, suggested, reason.
       Malformed items are skipped.
    2. The target_skill (after normalize_skill) must be in the code-computed
       missing_skills list (also normalized).  Items for skills that are not
       actually missing — or that the LLM invented — are dropped.
    3. suggested and reason are truncated to _MAX_SUGGESTION_CHARS characters.
    4. verify_original is run on the original field.  If it fails (the text
       was not found in the resume), original is set to "" so the UI can show
       "Consider adding a bullet like:" instead of a struck-through fabrication.
    5. The list is capped at 5 items.

    Returns a list of clean dicts: {target_skill, original, suggested, reason}.
    """
    # Build a set of normalized missing skill strings for fast O(1) lookup
    normalized_missing = {normalize_skill(s) for s in missing_skills}

    cleaned: list[dict] = []

    for item in raw_items:
        # --- shape check ---
        if not isinstance(item, dict):
            continue
        try:
            target_skill = str(item["target_skill"]).strip()
            original     = str(item["original"]).strip()
            suggested    = str(item["suggested"]).strip()
            reason       = str(item["reason"]).strip()
        except KeyError:
            continue  # missing required key — skip

        # --- drop items for skills that aren't actually missing ---
        if normalize_skill(target_skill) not in normalized_missing:
            continue

        # --- trim overly long strings ---
        suggested = suggested[:_MAX_SUGGESTION_CHARS]
        reason    = reason[:_MAX_SUGGESTION_CHARS]

        # --- original verification: blank it out if not found in resume ---
        if not verify_original(original, resume_text):
            original = ""

        cleaned.append({
            "target_skill": target_skill,
            "original":     original,
            "suggested":    suggested,
            "reason":       reason,
        })

        if len(cleaned) == 5:
            break

    return cleaned
