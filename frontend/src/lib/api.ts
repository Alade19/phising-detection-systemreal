export type PredictionResult = "phishing" | "legitimate";

export interface PredictionEntry {
  url: string;
  result: PredictionResult;
  confidence: number;
  timestamp: string;
}

export interface EmailPredictionEntry {
  subject: string;
  preview: string;
  result: PredictionResult;
  confidence: number;
  timestamp: string;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:5000";

async function handleResponse<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || "Something went wrong. Please try again.");
  }
  return data as T;
}

export async function checkUrl(url: string): Promise<{
  prediction: PredictionEntry;
  history: PredictionEntry[];
}> {
  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return handleResponse(res);
}

export async function fetchHistory(): Promise<{ history: PredictionEntry[] }> {
  const res = await fetch(`${API_BASE_URL}/api/history`, {
    cache: "no-store",
  });
  return handleResponse(res);
}

export async function clearHistory(): Promise<{ history: PredictionEntry[] }> {
  const res = await fetch(`${API_BASE_URL}/api/history`, {
    method: "DELETE",
  });
  return handleResponse(res);
}

export async function checkEmail(
  subject: string,
  body: string
): Promise<{
  prediction: EmailPredictionEntry;
  history: EmailPredictionEntry[];
}> {
  const res = await fetch(`${API_BASE_URL}/api/predict-email`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject, body }),
  });
  return handleResponse(res);
}

export async function fetchEmailHistory(): Promise<{
  history: EmailPredictionEntry[];
}> {
  const res = await fetch(`${API_BASE_URL}/api/email-history`, {
    cache: "no-store",
  });
  return handleResponse(res);
}

export async function clearEmailHistory(): Promise<{
  history: EmailPredictionEntry[];
}> {
  const res = await fetch(`${API_BASE_URL}/api/email-history`, {
    method: "DELETE",
  });
  return handleResponse(res);
}
