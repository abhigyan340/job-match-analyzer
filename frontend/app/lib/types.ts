/**
 * Shared TypeScript types for the /analyze API response.
 * Keep this in sync with the Pydantic models in ml-service/main.py.
 */

export interface SubScore {
  score: number | null;
  explanation: string;
}

export interface SubScores {
  skills:     SubScore;
  semantic:   SubScore;
  experience: SubScore;
  education:  SubScore;
}

export interface AnalysisResult {
  overall_score:            number;
  sub_scores:               SubScores;
  tfidf_score:              number;
  matched_skills:           string[];
  missing_skills:           string[];
  missing_preferred_skills: string[];
  feedback:                 string;
  ai_used:                  boolean;
  job_description_text:     string;
}
