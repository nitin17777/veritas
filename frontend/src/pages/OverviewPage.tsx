import { useState } from "react";
import type { HistoryItem, SettingsConfig, VerificationResult } from "../types";
import { Badge } from "../components/Badge";
import { ClaimComposer } from "../components/ClaimComposer";
import { Icon } from "../components/Icons";

interface OverviewProps {
  onStart: (params: { claim: string; claim_date?: string; mode: "benchmark" | "live_search"; backend: "mock" | "local" | "api" }) => void;
  onResult: (result: VerificationResult) => void;
  onHistory: () => void;
  onResearch: () => void;
  history: HistoryItem[];
  settings?: SettingsConfig;
}

export function OverviewPage({
  onStart,
  onResult,
  onHistory,
  onResearch,
  history,
  settings,
}: OverviewProps) {
  const [claim, setClaim] = useState("");

  const stages = [
    ["search", "Retrieve", "Find relevant evidence using keyword and semantic retrieval."],
    ["filter", "Evaluate", "Rank sources by relevance, credibility, and recency."],
    ["shield", "Verify", "Analyze the claim using evidence-constrained AI."],
    ["check", "Validate", "Check explanation faithfulness and confidence."],
  ];

  const now = new Date();
  const dayStr = now.toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" }).toUpperCase();
  const firstName = (settings?.userName || "Alex Morgan").split(" ")[0];

  return (
    <div className="page overview-page">
      <div className="greeting">
        <div>
          <span className="eyebrow">{dayStr}</span>
          <h1>Good morning, {firstName}.</h1>
          <p>Discover what stands behind a claim.</p>
        </div>
        <button className="icon-button" aria-label="Notifications" type="button">
          <Icon name="bell" />
        </button>
      </div>

      <ClaimComposer
        value={claim}
        setValue={setClaim}
        onVerify={onStart}
        defaultMode={settings?.sourceMode || "benchmark"}
        defaultBackend={settings?.backend || "mock"}
      />

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>How VERITAS works</h2>
            <p>From claim to inspectable verdict in four evidence-grounded stages.</p>
          </div>
          <button className="text-button" onClick={onResearch} type="button">
            Explore methodology <Icon name="arrow" size={15} />
          </button>
        </div>
        <div className="stage-grid">
          {stages.map((s, i) => (
            <div className="stage-card" key={s[1]}>
              <div className="stage-top">
                <span className="stage-icon">
                  <Icon name={s[0]} size={17} />
                </span>
                <span className="stage-num">0{i + 1}</span>
              </div>
              <h3>{s[1]}</h3>
              <p>{s[2]}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Recent verifications</h2>
            <p>Your latest evidence-based checks.</p>
          </div>
          <button className="text-button" onClick={onHistory} type="button">
            View all <Icon name="arrow" size={15} />
          </button>
        </div>
        <div className="recent-list card">
          {history.length > 0 ? (
            history.slice(0, 3).map((item) => (
              <div className="recent-row" key={item.id || item.claim}>
                <div className="claim-cell">
                  <span className="doc-icon">
                    <Icon name="file" size={17} />
                  </span>
                  <div>
                    <strong>{item.claim}</strong>
                    <span>Verified {item.date}</span>
                  </div>
                </div>
                <Badge verdict={item.verdict} />
                <div className="confidence-mini">
                  <strong>{item.confidence}%</strong>
                  <span>Confidence</span>
                </div>
                <button
                  className="row-action"
                  onClick={() => onResult(item.result)}
                  type="button"
                >
                  View details <Icon name="arrow" size={15} />
                </button>
              </div>
            ))
          ) : (
            <div style={{ padding: "1.5rem", textAlign: "center", color: "var(--muted)" }}>
              No previous verifications yet. Enter a claim above to verify!
            </div>
          )}
        </div>
      </section>

      <div className="notice">
        <Icon name="info" size={16} />
        <span>
          VERITAS assists human judgment. It should not be treated as the sole authority on factual
          matters.
        </span>
      </div>
    </div>
  );
}
