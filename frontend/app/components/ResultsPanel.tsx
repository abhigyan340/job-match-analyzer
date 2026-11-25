"use client";

import { useState } from "react";
import type { AnalysisResult, SubScore } from "../lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Tailwind text-color class based on score. */
function scoreColorClass(score: number): string {
  if (score >= 70) return "text-emerald-600";
  if (score >= 40) return "text-amber-500";
  return "text-red-500";
}

/** SVG stroke hex color based on score. */
function ringStrokeColor(score: number): string {
  if (score >= 70) return "#059669"; // emerald-600
  if (score >= 40) return "#f59e0b"; // amber-500
  return "#dc2626";                  // red-600
}

// ---------------------------------------------------------------------------
// Overall score ring (SVG circle, colored by score)
// ---------------------------------------------------------------------------

function ScoreRing({ score }: { score: number }) {
  const radius = 44;
  const circumference = 2 * Math.PI * radius;
  const dashFilled = (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width="120" height="120" className="-rotate-90" aria-hidden="true">
        {/* Track */}
        <circle
          cx="60" cy="60" r={radius}
          fill="none" stroke="#e2e8f0" strokeWidth="10"
        />
        {/* Filled arc */}
        <circle
          cx="60" cy="60" r={radius}
          fill="none"
          stroke={ringStrokeColor(score)}
          strokeWidth="10"
          strokeDasharray={`${dashFilled} ${circumference}`}
          strokeLinecap="round"
        />
      </svg>
      {/* Score number centered below the ring */}
      <div className="mt-2 text-center">
        <span className={`text-4xl font-bold ${scoreColorClass(score)}`}>
          {score.toFixed(0)}
        </span>
        <span className="text-slate-500 text-lg">/100</span>
      </div>
      <p className="text-sm text-slate-500 mt-1">Overall match</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-score bar (label + explanation + progress bar + value)
// ---------------------------------------------------------------------------

interface SubScoreBarProps {
  label:       string;
  sub:         SubScore;
  diagnostic?: boolean; // true for TF-IDF — renders smaller/muted
}

function SubScoreBar({ label, sub, diagnostic = false }: SubScoreBarProps) {
  const { score, explanation } = sub;

  if (score === null || score === undefined) {
    return (
      <div className={`mb-4 ${diagnostic ? "opacity-60" : ""}`}>
        <div className="flex justify-between text-sm mb-0.5">
          <span className={diagnostic ? "text-slate-500 text-xs" : "font-medium"}>
            {label}
          </span>
          <span className="text-slate-400 text-xs">Not available</span>
        </div>
        <p className="text-xs text-slate-400 mb-1">{explanation}</p>
        <div className="w-full h-1.5 bg-slate-100 rounded-full" />
      </div>
    );
  }

  const barColor = diagnostic
    ? "bg-slate-400"
    : score >= 70
    ? "bg-emerald-500"
    : score >= 40
    ? "bg-amber-400"
    : "bg-red-500";

  return (
    <div className={`mb-4 ${diagnostic ? "opacity-70" : ""}`}>
      <div className="flex justify-between text-sm mb-0.5">
        <span className={diagnostic ? "text-slate-500 text-xs" : "font-medium"}>
          {label}
        </span>
        <span className={diagnostic ? "text-slate-400 text-xs" : "text-sm font-medium"}>
          {score.toFixed(1)}
        </span>
      </div>
      <p className={`mb-1 ${diagnostic ? "text-xs text-slate-400" : "text-xs text-slate-500"}`}>
        {explanation}
      </p>
      <div className={`w-full rounded-full overflow-hidden ${diagnostic ? "h-1.5 bg-slate-100" : "h-2 bg-slate-200"}`}>
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-500`}
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skill pills
// ---------------------------------------------------------------------------

function SkillPills({
  skills,
  tone,
}: {
  skills: string[];
  tone: "match" | "missing" | "preferred";
}) {
  if (skills.length === 0) {
    return <p className="text-sm text-slate-400">None.</p>;
  }
  const cls =
    tone === "match"
      ? "bg-emerald-100 text-emerald-800"
      : tone === "preferred"
      ? "bg-sky-100 text-sky-800"
      : "bg-amber-100 text-amber-800";

  return (
    <div className="flex flex-wrap gap-2">
      {skills.map((skill) => (
        <span
          key={skill}
          className={`text-xs font-medium px-2.5 py-1 rounded-full ${cls}`}
        >
          {skill}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// "How is this calculated?" collapsed section
// ---------------------------------------------------------------------------

function ScoreFormula() {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-4 border border-slate-200 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex justify-between items-center px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-slate-400"
      >
        <span>How is this calculated?</span>
        <span aria-hidden="true">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="px-4 py-3 text-xs text-slate-600 bg-slate-50 space-y-1">
          <p className="font-medium text-slate-700 mb-2">Overall score weights:</p>
          <p>Skills match — <strong>40%</strong> (matched required skills / total required)</p>
          <p>Semantic similarity — <strong>30%</strong> (sentence-embedding cosine similarity)</p>
          <p>Experience fit — <strong>20%</strong> (LLM-assessed; dropped if AI unavailable)</p>
          <p>Education fit — <strong>10%</strong> (LLM-assessed; dropped if AI unavailable)</p>
          <p className="mt-2 text-slate-400">
            If a component is unavailable, the remaining weights are re-normalized
            so they still sum to 100%.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ResultsPanel({ result }: { result: AnalysisResult }) {
  const {
    overall_score,
    sub_scores,
    tfidf_score,
    matched_skills,
    missing_skills,
    missing_preferred_skills,
    feedback,
    ai_used,
  } = result;

  return (
    <section
      className="mt-10 border border-slate-200 rounded-xl p-6 bg-white space-y-6"
      aria-label="Analysis results"
    >
      <h2 className="text-xl font-semibold">Results</h2>

      {/* AI unavailable notice */}
      {!ai_used && (
        <div className="rounded-lg bg-amber-50 border border-amber-200 px-4 py-2.5 text-sm text-amber-800">
          AI analysis was unavailable — showing similarity scores only. Skill
          lists and sub-scores are based on keyword matching.
        </div>
      )}

      {/* Overall score ring */}
      <div className="flex justify-center py-2">
        <ScoreRing score={overall_score} />
      </div>

      {/* Sub-score bars */}
      <div>
        <h3 className="font-medium text-slate-700 mb-3">Score breakdown</h3>
        <SubScoreBar label="Skills match"       sub={sub_scores.skills}     />
        <SubScoreBar label="Semantic similarity" sub={sub_scores.semantic}   />
        <SubScoreBar label="Experience fit"      sub={sub_scores.experience} />
        <SubScoreBar label="Education fit"       sub={sub_scores.education}  />

        {/* TF-IDF as a diagnostic row */}
        <SubScoreBar
          label="Keyword overlap (diagnostic)"
          sub={{ score: tfidf_score, explanation: "TF-IDF cosine similarity — not part of the overall score" }}
          diagnostic
        />

        <ScoreFormula />
      </div>

      {/* Skills */}
      <div className="grid sm:grid-cols-2 gap-6">
        <div>
          <h3 className="font-medium mb-2">Matched skills</h3>
          <SkillPills skills={matched_skills} tone="match" />
        </div>
        <div>
          <h3 className="font-medium mb-2">Missing required skills</h3>
          <SkillPills skills={missing_skills} tone="missing" />
        </div>
      </div>

      {missing_preferred_skills.length > 0 && (
        <div>
          <h3 className="font-medium mb-2">Missing preferred skills</h3>
          <SkillPills skills={missing_preferred_skills} tone="preferred" />
        </div>
      )}

      {/* Feedback */}
      <div>
        <h3 className="font-medium mb-2">Feedback</h3>
        <p className="text-sm text-slate-700 leading-relaxed">{feedback}</p>
      </div>
    </section>
  );
}
