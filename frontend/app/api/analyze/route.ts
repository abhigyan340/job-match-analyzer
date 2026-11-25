import { NextRequest, NextResponse } from "next/server";

// URL of the Python FastAPI ML microservice.
// Set ML_SERVICE_URL in .env.local for dev (defaults to http://localhost:8000).
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  let mlResponse: Response;

  try {
    const formData = await req.formData();

    // Forward the entire multipart form (resume_file, job_description, and
    // the new optional jd_file) to the Python service unchanged.
    mlResponse = await fetch(`${ML_SERVICE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    // fetch() itself threw — the Python service is unreachable.
    console.error("Could not reach ML service:", err);
    return NextResponse.json(
      { detail: "Could not reach the analysis service. Is it running?" },
      { status: 502 }
    );
  }

  // Read the body as text first so we can handle non-JSON error responses
  // (e.g. a plain-text 500 from uvicorn) without crashing the proxy.
  const rawText = await mlResponse.text();

  let data: unknown;
  try {
    data = JSON.parse(rawText);
  } catch {
    // The upstream response was not valid JSON (e.g. a plain-text crash trace).
    // Surface a short, human-readable message instead of leaking the raw body.
    console.error("ML service returned non-JSON response:", rawText.slice(0, 500));
    return NextResponse.json(
      { detail: "The analysis service returned an unexpected response. Check the service logs." },
      { status: mlResponse.status }
    );
  }

  return NextResponse.json(data, { status: mlResponse.status });
}
