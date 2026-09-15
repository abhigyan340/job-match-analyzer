import { NextRequest, NextResponse } from "next/server";

// URL of the Python FastAPI ML microservice.
// Set ML_SERVICE_URL in your environment (e.g. http://localhost:8000 in dev,
// or your deployed microservice URL in production).
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    // Forward the multipart form data as-is to the Python service
    const mlResponse = await fetch(`${ML_SERVICE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await mlResponse.json();

    if (!mlResponse.ok) {
      return NextResponse.json(data, { status: mlResponse.status });
    }

    return NextResponse.json(data);
  } catch (err) {
    console.error("Error proxying to ML service:", err);
    return NextResponse.json(
      { detail: "Could not reach the analysis service. Is it running?" },
      { status: 502 }
    );
  }
}
