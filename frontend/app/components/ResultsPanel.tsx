import type { AnalysisResult } from "../page";

function ScoreBar({ label, score }: { label: string; score: number }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium">{label}</span>
        <span>{score.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-slate-900 rounded-full"
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
    </div>
  );
}

function SkillPills({ skills, tone }: { skills: string[]; tone: "match" | "missing" }) {
  if (skills.length === 0) {
    return <p className="text-sm text-slate-500">None detected.</p>;
  }
  const toneClasses =
    tone === "match"
      ? "bg-emerald-100 text-emerald-800"
      : "bg-amber-100 text-amber-800";

  return (
    <div className="flex flex-wrap gap-2">
      {skills.map((skill) => (
        <span key={skill} className={`text-xs font-medium px-2.5 py-1 rounded-full ${toneClasses}`}>
          {skill}
        </span>
      ))}
    </div>
  );
}

export default function ResultsPanel({ result }: { result: AnalysisResult }) {
  return (
    <section className="mt-10 border border-slate-200 rounded-xl p-6 bg-white">
      <h2 className="text-xl font-semibold mb-4">Results</h2>

      <ScoreBar label="Overall Match (semantic)" score={result.embedding_score} />
      <ScoreBar label="Keyword Overlap (TF-IDF)" score={result.tfidf_score} />

      <div className="grid sm:grid-cols-2 gap-6 mt-6">
        <div>
          <h3 className="font-medium mb-2">Matched Skills</h3>
          <SkillPills skills={result.matched_skills} tone="match" />
        </div>
        <div>
          <h3 className="font-medium mb-2">Missing Skills</h3>
          <SkillPills skills={result.missing_skills} tone="missing" />
        </div>
      </div>

      <div className="mt-6">
        <h3 className="font-medium mb-2">Feedback</h3>
        <p className="text-sm text-slate-700 leading-relaxed">{result.feedback}</p>
      </div>
    </section>
  );
}
