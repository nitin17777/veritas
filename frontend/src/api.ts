import type { ExampleClaim, HistoryItem, VerificationResult } from "./types";

const API_BASE = ""; // Uses relative URL so Vite proxy or direct server handles /api

export async function verifyClaim(params: {
  claim: string;
  claim_date?: string;
  mode?: "benchmark" | "live_search";
  backend?: "mock" | "api";
}): Promise<VerificationResult> {
  const res = await fetch(`${API_BASE}/api/verify`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `Verification failed with HTTP ${res.status}`);
  }

  return res.json();
}

export async function extractClaims(text: string, backend = "mock"): Promise<{ claim: string; confidence: number }[]> {
  const res = await fetch(`${API_BASE}/api/extract`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text, backend }),
  });

  if (!res.ok) {
    return [];
  }

  const data = await res.json();
  return data.claims || [];
}

export async function getHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${API_BASE}/api/history`);
  if (!res.ok) {
    throw new Error(`Failed to load history: ${res.status}`);
  }
  return res.json();
}

export async function clearHistory(): Promise<void> {
  await fetch(`${API_BASE}/api/history`, {
    method: "DELETE",
  });
}

export async function getExamples(): Promise<ExampleClaim[]> {
  const res = await fetch(`${API_BASE}/api/examples`);
  if (!res.ok) {
    return [
      {
        id: 1,
        claim: "COVID-19 mRNA vaccines significantly reduce the risk of severe hospitalizations.",
        topic: "Public Health",
        expected_verdict: "TRUE",
      },
      {
        id: 2,
        claim: "City X banned all private vehicles from its centre starting January 2026.",
        topic: "Urban Transport",
        expected_verdict: "MISLEADING",
      },
      {
        id: 3,
        claim: "The national rail network will be free for everyone under 25.",
        topic: "Public Policy",
        expected_verdict: "PARTIALLY TRUE",
      },
      {
        id: 4,
        claim: "NASA confirmed the Moon is entirely composed of aged dairy cheddar cheese.",
        topic: "Science & Astronomy",
        expected_verdict: "FALSE",
      },
      {
        id: 5,
        claim: "A newly discovered comet will be visible during daylight this year.",
        topic: "Astronomy",
        expected_verdict: "UNVERIFIED",
      },
    ];
  }
  return res.json();
}

export async function getHealth(): Promise<{ status: string; backends: string[]; modes: string[] }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    return { status: "unknown", backends: ["mock"], modes: ["benchmark"] };
  }
  return res.json();
}
