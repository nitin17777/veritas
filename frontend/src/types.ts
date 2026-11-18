import type { ReactNode } from "react";

export type Page = "landing" | "verify" | "processing" | "results" | "history" | "research" | "settings";

export type Verdict = "TRUE" | "PARTIALLY TRUE" | "MISLEADING" | "FALSE" | "UNVERIFIED";

export interface EvidenceCard {
  id: string; // e.g. "E1"
  passage_id: string;
  source: string;
  domain: string;
  date: string;
  tier: string;
  stance: string; // "Supports claim" | "Contradicts claim" | "Provides context"
  quote: string;
  relevance: number; // 0 - 100
  credibility: number; // 0 - 100
  temporal: number; // 0 - 100
  combined: number; // 0 - 100
  is_cited: boolean;
  url: string;
}

export interface PipelineStage {
  name: string;
  time_ms: number;
}

export interface EvidenceScoringSummary {
  relevance: number;
  credibility: number;
  temporal: number;
  combined: number;
  formula: string;
}

export interface VerificationResult {
  id: string;
  claim: string;
  verdict: Verdict;
  confidence: number; // 0 - 100
  explanation: string;
  evidence_cards: EvidenceCard[];
  source_mode: string;
  abstained: boolean;
  abstention_reasons: string[];
  faithfulness_score: number; // 0 - 100
  is_supported: boolean;
  nli_status: string; // "Passed" | "Failed" | "Abstained"
  evidence_scoring: EvidenceScoringSummary;
  pipeline_stages: PipelineStage[];
  completed_at: string;
  disclaimer: string;
}

export interface HistoryItem {
  id: string;
  claim: string;
  verdict: Verdict;
  confidence: number;
  date: string;
  evidence: number;
  result: VerificationResult;
}

export interface ExampleClaim {
  id: number;
  claim: string;
  topic: string;
  expected_verdict: string;
}

export interface SettingsConfig {
  theme: "Light" | "System" | "Dark";
  sourceMode: "benchmark" | "live_search";
  backend: "mock" | "api";
  includeCredibility: boolean;
  prioritizeRecent: boolean;
  detailedView: boolean;
  userName: string;
  userEmail: string;
  workspaceName: string;
  notifyComplete: boolean;
  notifyUpdates: boolean;
}
