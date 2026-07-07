import { NextResponse } from "next/server";

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const response = await fetch(`${PYTHON_API_URL}/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail ?? data.error ?? "Python forecasting service failed" },
        { status: response.status },
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        error:
          `Unable to reach Python forecasting service at ${PYTHON_API_URL}. ` +
          `Start it with: cd python && uvicorn energy_forecasting.api:app --host 0.0.0.0 --port 8000. ` +
          message,
      },
      { status: 502 },
    );
  }
}
