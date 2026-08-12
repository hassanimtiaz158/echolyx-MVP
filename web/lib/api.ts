export type Prediction = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
};

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/+$/, "");

export class ApiError extends Error {}

export async function classifyAudio(file: File | Blob, filename: string): Promise<Prediction> {
  if (!API_BASE) {
    throw new ApiError(
      "NEXT_PUBLIC_API_URL is not set — point it at your inference backend (e.g. a Hugging Face Space URL)."
    );
  }

  const form = new FormData();
  form.append("file", file, filename);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  } catch {
    throw new ApiError("Could not reach the inference backend. It may be asleep or offline — try again in a moment.");
  }

  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(detail?.detail ?? `Prediction failed (HTTP ${res.status}).`);
  }

  return res.json();
}
