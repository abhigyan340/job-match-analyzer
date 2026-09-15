"use client";

import { useState } from "react";
import ResultsPanel from "./components/ResultsPanel";

export interface AnalysisResult {
  match_score: number;
  tfidf_score: number;
  embedding_score: number;
  matched_skills: string[];
  missing_skills: string[];
  feedback: string;
}

export default function Home() {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!resumeFile || !jobDescription.trim()) {
      setError("Please upload a resume and paste a job description.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("resume_file", resumeFile);
      formData.append("job_description", jobDescription);

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Analysis failed. Please try again.");
      }

      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-3xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-2">AI Job Match Analyzer</h1>
      <p className="text-slate-600 mb-8">
        Upload your resume and paste a job description to get a match score,
        skill gaps, and tailored feedback.
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block font-medium mb-2" htmlFor="resume">
            Resume (PDF or DOCX)
          </label>
          <input
            id="resume"
            type="file"
            accept=".pdf,.docx"
            onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm border border-slate-300 rounded-lg p-2 bg-white"
          />
        </div>

        <div>
          <label className="block font-medium mb-2" htmlFor="jd">
            Job Description
          </label>
          <textarea
            id="jd"
            rows={8}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the full job description here..."
            className="block w-full text-sm border border-slate-300 rounded-lg p-3 bg-white"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="bg-slate-900 text-white font-medium px-5 py-2.5 rounded-lg hover:bg-slate-700 disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze Match"}
        </button>

        {error && <p className="text-red-600 text-sm">{error}</p>}
      </form>

      {result && <ResultsPanel result={result} />}
    </main>
  );
}
