import { useEffect, useState } from "react";
import type { ExampleClaim, SettingsConfig } from "../types";
import { getExamples } from "../api";
import { ClaimComposer } from "../components/ClaimComposer";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";

interface VerifyPageProps {
  onStart: (params: { claim: string; claim_date?: string; mode: "benchmark" | "live_search"; backend: "mock" | "local" | "api" }) => void;
  settings?: SettingsConfig;
}

export function VerifyPage({ onStart, settings }: VerifyPageProps) {
  const [claim, setClaim] = useState("");
  const [examples, setExamples] = useState<ExampleClaim[]>([]);

  useEffect(() => {
    getExamples()
      .then((data) => setExamples(data))
      .catch(() => {});
  }, []);

  return (
    <div className="page narrow-page">
      <PageHeader
        title="Verify a claim"
        description="Submit a factual statement or news article and let VERITAS investigate the evidence."
      />

      <ClaimComposer
        expanded
        value={claim}
        setValue={setClaim}
        onVerify={onStart}
        defaultMode={settings?.sourceMode || "benchmark"}
        defaultBackend={settings?.backend || "mock"}
      />

      <div className="privacy-note">
        <Icon name="shield" size={18} />
        <div>
          <strong>Evidence-grounded by design</strong>
          <p>
            Your claim is analyzed through hybrid retrieval, source credibility scoring, temporal
            ranking, and evidence-constrained verification.
          </p>
        </div>
      </div>

      <div className="examples-section">
        <h2>Try an example</h2>
        <div className="example-grid">
          {examples.map((item, i) => (
            <button
              key={item.id || item.claim}
              onClick={() => setClaim(item.claim)}
              type="button"
            >
              <span>0{i + 1}</span>
              <div style={{ textAlign: "left", flex: 1 }}>
                <div>{item.claim}</div>
                <small style={{ color: "var(--muted)", fontSize: "0.76rem" }}>
                  {item.topic} · Expected: {item.expected_verdict}
                </small>
              </div>
              <Icon name="arrow" size={15} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
