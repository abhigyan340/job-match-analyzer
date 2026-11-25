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

// Which mode the JD input is in
type JdMode = "paste" | "upload";

export default function Home() {
  const [resumeFile, setResumeFile] = useState<File | null>(null);

  // Job description — either pasted text or an uploaded file
  const [jdMode, setJdMode] = useState<JdMode>("paste");
  const [jobDescription, setJobDescription] = useState("");
  const [jdFile, setJdFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  function handleJdModeChange(mode: JdMode) {
    setJdMode(mode);
    // Clear the other source so we never send stale data
    if (mode === "paste") setJdFile(null);
    else setJobDescription("");
    setError(null);
  }

  function handleJdFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setJdFile(e.target.files?.[0] ?? null);
  }

  function removeJdFile() {
    setJdFile(null);
    // Reset the hidden file input so the user can re-select the same file
    const input = document.getElementById("jd-file") as HTMLInputElement | null;
    if (input) input.value = "";
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    // --- Client-side validation ---
    if (!resumeFile) {
      setError("Please upload your resume (PDF or DOCX).");
      return;
    }
    if (jdMode === "paste" && !jobDescription.trim()) {
      setError("Please paste the job description.");
      return;
    }
    if (jdMode === "upload" && !jdFile) {
      setError("Please upload a job description file (PDF or DOCX).");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("resume_file", resumeFile);

      if (jdMode === "paste") {
        formData.append("job_description", jobDescription);
      } else if (jdFile) {
        formData.append("jd_file", jdFile);
      }

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const payload = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          (payload as { detail?: string }).detail ||
            "Analysis failed. Please try again."
        );
      }

      setResult(payload as AnalysisResult);
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
        Upload your resume and provide a job description to get a match score,
        skill gaps, and tailored feedback.
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Resume upload */}
        <div>
          <label className="block font-medium mb-2" htmlFor="resume">
            Resume (PDF or DOCX) <span className="text-red-500">*</span>
          </label>
          <input
            id="resume"
            type="file"
            accept=".pdf,.docx"
            onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm border border-slate-300 rounded-lg p-2 bg-white focus:outline-none focus:ring-2 focus:ring-slate-500"
          />
          {resumeFile && (
            <p className="mt-1 text-xs text-slate-500">
              Selected: {resumeFile.name}
            </p>
          )}
        </div>

        {/* Job description — Paste / Upload toggle */}
        <div>
          <span className="block font-medium mb-2">
            Job Description <span className="text-red-500">*</span>
          </span>

          {/* Tab-style toggle */}
          <div className="flex border border-slate-300 rounded-lg overflow-hidden w-fit mb-3" role="group" aria-label="Job description input method">
            <button
              type="button"
              onClick={() => handleJdModeChange("paste")}
              aria-pressed={jdMode === "paste"}
              className={`px-4 py-1.5 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-inset focus:ring-slate-500 ${
                jdMode === "paste"
                  ? "bg-slate-900 text-white"
                  : "bg-white text-slate-700 hover:bg-slate-100"
              }`}
            >
              Paste text
            </button>
            <button
              type="button"
              onClick={() => handleJdModeChange("upload")}
              aria-pressed={jdMode === "upload"}
              className={`px-4 py-1.5 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-inset focus:ring-slate-500 ${
                jdMode === "upload"
                  ? "bg-slate-900 text-white"
                  : "bg-white text-slate-700 hover:bg-slate-100"
              }`}
            >
              Upload file
            </button>
          </div>

          {jdMode === "paste" ? (
            <textarea
              id="jd"
              rows={8}
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the full job description here..."
              aria-label="Job description text"
              className="block w-full text-sm border border-slate-300 rounded-lg p-3 bg-white focus:outline-none focus:ring-2 focus:ring-slate-500"
            />
          ) : (
            <div>
              <label
                htmlFor="jd-file"
                className="block text-sm text-slate-600 mb-1"
              >
                Upload a PDF or DOCX copy of the job description
              </label>
              <input
                id="jd-file"
                type="file"
                accept=".pdf,.docx"
                onChange={handleJdFileChange}
                className="block w-full text-sm border border-slate-300 rounded-lg p-2 bg-white focus:outline-none focus:ring-2 focus:ring-slate-500"
              />
              {jdFile && (
                <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                  <span>Selected: {jdFile.name}</span>
                  <button
                    type="button"
                    onClick={removeJdFile}
                    className="text-red-500 hover:text-red-700 underline focus:outline-none focus:ring-1 focus:ring-red-500 rounded"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="bg-slate-900 text-white font-medium px-5 py-2.5 rounded-lg hover:bg-slate-700 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2"
        >
          {loading ? "Analyzing…" : "Analyze Match"}
        </button>

        {error && (
          <p role="alert" className="text-red-600 text-sm">
            {error}
          </p>
        )}
      </form>

      {result && <ResultsPanel result={result} />}
    </main>
  );
}
