import { useState } from "react";
import { Icon } from "./Icons";

interface ClaimComposerProps {
  value: string;
  setValue: (v: string) => void;
  onVerify: (params: { claim: string; claim_date?: string; mode: "benchmark" | "live_search"; backend: "mock" | "api" }) => void;
  expanded?: boolean;
  defaultMode?: "benchmark" | "live_search";
  defaultBackend?: "mock" | "api";
}

export function ClaimComposer({
  value,
  setValue,
  onVerify,
  expanded = false,
  defaultMode = "benchmark",
  defaultBackend = "mock",
}: ClaimComposerProps) {
  const [inputType, setInputType] = useState<"claim" | "article">("claim");
  const [claimDate, setClaimDate] = useState<string>("");
  const [mode, setMode] = useState<"benchmark" | "live_search">(defaultMode);
  const [backend, setBackend] = useState<"mock" | "api">(defaultBackend);

  const handleSubmit = () => {
    if (!value.trim()) return;
    onVerify({
      claim: value.trim(),
      claim_date: claimDate || undefined,
      mode,
      backend,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={`composer card ${expanded ? "composer-expanded" : ""}`}>
      {expanded && (
        <div className="segmented">
          <button
            className={inputType === "claim" ? "selected" : ""}
            onClick={() => setInputType("claim")}
            type="button"
          >
            Single claim
          </button>
          <button
            className={inputType === "article" ? "selected" : ""}
            onClick={() => setInputType("article")}
            type="button"
          >
            Long article
          </button>
        </div>
      )}

      <div className="textarea-wrap">
        <textarea
          aria-label="Claim text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            inputType === "article"
              ? "Paste the full article text here. VERITAS will extract check-worthy factual claims..."
              : "Paste a news headline, social media post, or factual claim..."
          }
          maxLength={inputType === "article" ? 12000 : 1200}
          rows={expanded ? 4 : 3}
        />
        <span className="char-count">
          {value.length.toLocaleString()} / {inputType === "article" ? "12,000" : "1,200"}
        </span>
      </div>

      {expanded && (
        <div className="date-row" style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-end" }}>
          <label style={{ flex: "1 1 180px" }}>
            <span>
              Claim date <small>Optional</small>
            </span>
            <div className="field with-icon">
              <Icon name="calendar" size={16} />
              <input
                type="date"
                value={claimDate}
                onChange={(e) => setClaimDate(e.target.value)}
              />
            </div>
          </label>

          <label style={{ flex: "1 1 180px" }}>
            <span>
              Evidence retrieval mode
            </span>
            <div className="select-wrap" style={{ width: "100%", marginTop: "0.25rem" }}>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as "benchmark" | "live_search")}
                style={{ width: "100%", padding: "0.45rem 2rem 0.45rem 0.75rem", borderRadius: "8px" }}
              >
                <option value="benchmark">Benchmark Corpus (RAG + Fixtures)</option>
                <option value="live_search">Live Web Search (DuckDuckGo)</option>
              </select>
              <Icon name="chevron" size={15} />
            </div>
          </label>

          <label style={{ flex: "1 1 160px" }}>
            <span>
              Verification backend
            </span>
            <div className="select-wrap" style={{ width: "100%", marginTop: "0.25rem" }}>
              <select
                value={backend}
                onChange={(e) => setBackend(e.target.value as "mock" | "api")}
                style={{ width: "100%", padding: "0.45rem 2rem 0.45rem 0.75rem", borderRadius: "8px" }}
              >
                <option value="mock">Mock Engine (Deterministic / Fast)</option>
                <option value="api">OpenAI API (GPT-4o-mini)</option>
              </select>
              <Icon name="chevron" size={15} />
            </div>
          </label>
        </div>
      )}

      <div className="composer-footer">
        <div className="helper">
          <Icon name="shield" size={16} />
          <span>
            {expanded
              ? "Hybrid retrieval, credibility scoring, and evidence-based verification."
              : "Every verdict is backed by inspectable evidence."}
          </span>
        </div>
        <div className="button-row">
          {value && (
            <button className="btn btn-ghost" type="button" onClick={() => setValue("")}>
              Clear
            </button>
          )}
          {!expanded && (
            <button
              className="btn btn-secondary"
              type="button"
              onClick={() => setInputType(inputType === "claim" ? "article" : "claim")}
            >
              <Icon name="file" size={16} />
              {inputType === "claim" ? "Paste article" : "Single claim"}
            </button>
          )}
          <button
            className="btn btn-primary"
            disabled={!value.trim()}
            onClick={handleSubmit}
            type="button"
          >
            Verify claim <Icon name="arrow" size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
