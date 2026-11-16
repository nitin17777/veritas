import { useMemo, useState } from "react";
import type { EvidenceCard, VerificationResult } from "../types";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";

interface ResultsPageProps {
  result: VerificationResult;
}

function ScoreBar({
  label,
  value,
  tone = "blue",
}: {
  label: string;
  value: number;
  tone?: string;
}) {
  return (
    <div className="score-row">
      <div>
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="bar">
        <span className={tone} style={{ width: `${Math.min(100, Math.max(0, value))}%` }} />
      </div>
    </div>
  );
}

export function ResultsPage({ result }: ResultsPageProps) {
  const [sort, setSort] = useState("Highest combined score");
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [methodOpen, setMethodOpen] = useState(false);
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  const sortedEvidence = useMemo(() => {
    const items = [...(result.evidence_cards || [])];
    if (sort === "Most relevant") return items.sort((a, b) => b.relevance - a.relevance);
    if (sort === "Most credible") return items.sort((a, b) => b.credibility - a.credibility);
    if (sort === "Most recent") {
      return items.sort((a, b) => Date.parse(b.date || "1970") - Date.parse(a.date || "1970"));
    }
    return items.sort((a, b) => b.combined - a.combined);
  }, [result.evidence_cards, sort]);

  const jumpTo = (id: string) => {
    setHighlightedId(id);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    window.setTimeout(() => setHighlightedId(null), 2500);
  };

  const handleCopyReport = () => {
    const report = `# VERITAS Verification Report
Claim: "${result.claim}"
Verdict: ${result.verdict} (Confidence: ${result.confidence}%)
Completed: ${result.completed_at}

## Explanation
${result.explanation}

## Evidence Summary
- Relevance: ${result.evidence_scoring?.relevance || 0}%
- Source Credibility: ${result.evidence_scoring?.credibility || 0}%
- Temporal Validity: ${result.evidence_scoring?.temporal || 0}%
- Combined Score: ${result.evidence_scoring?.combined || 0}%

## Cited Sources
${result.evidence_cards.map((ev) => `- [${ev.id}] ${ev.source} (${ev.domain}): "${ev.quote}" (Combined: ${ev.combined}%)`).join("\n")}

${result.disclaimer}
`;
    navigator.clipboard.writeText(report).then(() => {
      setCopyStatus("Copied report!");
      window.setTimeout(() => setCopyStatus(null), 2000);
    });
  };

  const handleDownloadReport = () => {
    const report = `# VERITAS Verification Report
Claim: "${result.claim}"
Verdict: ${result.verdict} (Confidence: ${result.confidence}%)
Completed: ${result.completed_at || "Recent"}

## Explanation
${result.explanation}

## Evidence Summary
- Relevance: ${result.evidence_scoring?.relevance || 0}%
- Source Credibility: ${result.evidence_scoring?.credibility || 0}%
- Temporal Validity: ${result.evidence_scoring?.temporal || 0}%
- Combined Score: ${result.evidence_scoring?.combined || 0}%

## Cited Sources
${result.evidence_cards.length > 0
  ? result.evidence_cards.map((ev) => `- [${ev.id}] ${ev.source} (${ev.domain}): "${ev.quote}" (Combined: ${ev.combined}%)`).join("\n")
  : "No cited sources."}

${result.disclaimer}
`;
    const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `veritas_report_${Date.now()}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: `VERITAS: ${result.verdict} - ${result.claim}`,
        text: `VERITAS verified: "${result.claim}" -> Verdict: ${result.verdict} (${result.confidence}% confidence).`,
        url: window.location.href,
      }).catch(() => {});
    } else {
      handleCopyReport();
    }
  };

  const verdictClass = result.verdict.toLowerCase().replace(" ", "-");
  const corroboratingCount = result.evidence_cards.length;

  // Clear, non-technical verdict summaries for everyday users
  const getPlainEnglishSummary = (verdict: string) => {
    switch (verdict.toUpperCase()) {
      case "TRUE":
        return {
          title: "Verified: This statement is accurate",
          desc: "Reliable documentation and credible evidence confirm that this claim is factually correct.",
          badgeText: "Confirmed True",
        };
      case "FALSE":
        return {
          title: "Disproven: This statement is false",
          desc: "Direct evidence and official findings contradict this claim.",
          badgeText: "Factually Incorrect",
        };
      case "PARTIALLY TRUE":
        return {
          title: "Needs Context: Partially accurate",
          desc: "Some elements are real, but key context or nuances are missing or overstated.",
          badgeText: "Partially True",
        };
      case "MISLEADING":
        return {
          title: "Misleading: Out-of-context or distorted",
          desc: "The statement presents facts in a selective way that leads to a false impression.",
          badgeText: "Misleading",
        };
      case "UNVERIFIED":
      default:
        return {
          title: "Unverified: Insufficient evidence to confirm",
          desc: "No verified, credible records exist in the corpus to confirm or deny this claim. VERITAS abstains from guessing.",
          badgeText: "Unverified / Needs Evidence",
        };
    }
  };

  const plainSummary = getPlainEnglishSummary(result.verdict);

  // Render explanation text with clickable citation tokens like [E1], [E2]
  const renderFormattedExplanation = (text: string) => {
    const parts = text.split(/(\[E\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/\[(E\d+)\]/);
      if (match) {
        const eid = match[1];
        return (
          <button
            key={idx}
            className="citation"
            onClick={() => jumpTo(eid)}
            type="button"
            title={`Jump to evidence ${eid}`}
          >
            {eid}
          </button>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  return (
    <div className="page results-page">
      <PageHeader
        title="Verification result"
        description={`Completed ${result.completed_at || "recently"}`}
      >
        <button className="btn btn-secondary" onClick={handleDownloadReport} type="button" title="Download markdown report">
          <Icon name="file" size={16} />
          Download .md
        </button>
        <button className="btn btn-secondary" onClick={handleShare} type="button">
          <Icon name="share" size={16} />
          Share
        </button>
        <button className="btn btn-secondary" onClick={handleCopyReport} type="button">
          <Icon name="copy" size={16} />
          {copyStatus || "Copy report"}
        </button>
      </PageHeader>

      <div className="result-claim">
        <span className="eyebrow">
          SUBMITTED CLAIM · {result.source_mode?.toUpperCase() || "BENCHMARK CORPUS"}
        </span>
        <h2>“{result.claim}”</h2>
      </div>

      <section className={`verdict-card verdict-${verdictClass}`}>
        <div className="verdict-main">
          <div className="verdict-icon">
            <Icon name={result.verdict === "TRUE" ? "check" : result.verdict === "FALSE" ? "x" : "shield"} size={26} />
          </div>
          <div>
            <span className="label">FINAL VERDICT · {plainSummary.badgeText.toUpperCase()}</span>
            <h2>{result.verdict}</h2>
            <p style={{ fontSize: "1.05rem", fontWeight: 500, color: "var(--foreground)", margin: "0.25rem 0 0.5rem 0" }}>
              {plainSummary.title}
            </p>
            <p>
              {renderFormattedExplanation(result.explanation)}
            </p>
          </div>
        </div>
        <div className="confidence-block">
          <div className="confidence-value">
            <div>
              <span>Certainty level</span>
              <strong>{result.confidence}%</strong>
            </div>
            <span className="confidence-caption">
              {result.confidence >= 80 ? "Strong factual support" : result.confidence >= 60 ? "Moderate evidence backing" : "Low / Inconclusive"}
            </span>
          </div>
          <div className="confidence-bar">
            <span style={{ width: `${result.confidence}%` }} />
          </div>
          <p>
            Based on {corroboratingCount} corroborated source{corroboratingCount === 1 ? "" : "s"}.
          </p>
        </div>
      </section>

      <div className="results-grid">
        <main>
          <section className="result-section card">
            <div className="result-section-heading">
              <span className="section-icon">
                <Icon name="file" size={18} />
              </span>
              <div>
                <h2>Why this verdict?</h2>
                <p>An evidence-constrained explanation</p>
              </div>
            </div>
            <div className="explanation">
              <p>{renderFormattedExplanation(result.explanation)}</p>
              {result.abstained && result.abstention_reasons?.length > 0 && (
                <div style={{ marginTop: "1rem", padding: "0.75rem", background: "rgba(165,180,252,.12)", borderRadius: "8px", fontSize: "0.88rem" }}>
                  <strong>Abstention reason:</strong> {result.abstention_reasons.join("; ")}
                </div>
              )}
            </div>
          </section>

          <section className="evidence-section">
            <div className="section-heading">
              <div>
                <h2>Evidence behind this verdict</h2>
                <p>
                  {result.evidence_cards.length} source{result.evidence_cards.length === 1 ? "" : "s"} retrieved and scored
                </p>
              </div>
              {result.evidence_cards.length > 1 && (
                <label className="select-wrap">
                  <span className="sr-only">Sort evidence</span>
                  <select value={sort} onChange={(e) => setSort(e.target.value)}>
                    <option>Highest combined score</option>
                    <option>Most relevant</option>
                    <option>Most credible</option>
                    <option>Most recent</option>
                  </select>
                  <Icon name="chevron" size={15} />
                </label>
              )}
            </div>

            <div className="evidence-list">
              {sortedEvidence.length > 0 ? (
                sortedEvidence.map((ev) => {
                  const isHighlighted = highlightedId === ev.id;
                  const stanceClass = ev.stance.toLowerCase().includes("contradicts")
                    ? "contradicts"
                    : ev.stance.toLowerCase().includes("supports")
                    ? "supports"
                    : "context";

                  return (
                    <article
                      className={`evidence-card card ${isHighlighted ? "highlighted" : ""}`}
                      id={ev.id}
                      key={ev.id || ev.passage_id}
                      style={
                        isHighlighted
                          ? { outline: "2px solid var(--blue)", boxShadow: "0 0 16px rgba(66,109,169,0.3)" }
                          : undefined
                      }
                    >
                      <div className="evidence-top">
                        <div className="evidence-source">
                          <span className="evidence-id">{ev.id}</span>
                          <div>
                            <strong>{ev.source}</strong>
                            <span>
                              {ev.domain} · {ev.date || "Date unspecified"}
                            </span>
                          </div>
                        </div>
                        <span className={`stance ${stanceClass}`}>{ev.stance}</span>
                      </div>
                      <blockquote>“{ev.quote}”</blockquote>
                      <div className="evidence-meta">
                        <span className="tier">
                          <Icon name="shield" size={14} />
                          {ev.tier}
                        </span>
                        <span>
                          Relevance <strong>{ev.relevance}</strong>
                        </span>
                        <span>
                          Credibility <strong>{ev.credibility}</strong>
                        </span>
                        <span>
                          Temporal <strong>{ev.temporal}</strong>
                        </span>
                        <span className="combined">
                          Combined <strong>{ev.combined}</strong>
                        </span>
                        {ev.url && (
                          <a
                            href={ev.url.startsWith("http") ? ev.url : `https://${ev.url}`}
                            target="_blank"
                            rel="noreferrer"
                            aria-label={`Open ${ev.source}`}
                          >
                            Source <Icon name="external" size={13} />
                          </a>
                        )}
                      </div>
                    </article>
                  );
                })
              ) : (
                <div className="card" style={{ padding: "2.5rem 1.5rem", textAlign: "center" }}>
                  <div style={{ display: "inline-flex", padding: "0.75rem", background: "var(--surface)", borderRadius: "50%", marginBottom: "0.75rem", color: "var(--muted)" }}>
                    <Icon name="search" size={24} />
                  </div>
                  <h3 style={{ fontSize: "1.05rem", fontWeight: 600, marginBottom: "0.35rem" }}>
                    No correlated evidence found in benchmark index
                  </h3>
                  <p style={{ color: "var(--muted)", fontSize: "0.9rem", maxWidth: "440px", margin: "0 auto 1.25rem auto", lineHeight: 1.5 }}>
                    Under strict VERITAS closed-world constraints, the system abstains from guessing when facts are absent from the local corpus.
                  </p>
                </div>
              )}
            </div>
          </section>

          <section className="method card">
            <button
              className="method-toggle"
              onClick={() => setMethodOpen(!methodOpen)}
              aria-expanded={methodOpen}
              type="button"
            >
              <div>
                <span className="section-icon">
                  <Icon name="research" size={18} />
                </span>
                <div>
                  <h2>How this result was generated</h2>
                  <p>Inspect the verification pipeline and stage latencies</p>
                </div>
              </div>
              <span className={methodOpen ? "rotate" : ""}>
                <Icon name="chevron" />
              </span>
            </button>
            {methodOpen && (
              <div className="pipeline" style={{ padding: "1.2rem 1rem", borderTop: "1px solid var(--border)" }}>
                {result.pipeline_stages && result.pipeline_stages.length > 0 ? (
                  result.pipeline_stages.map((stage, i) => (
                    <div className="pipeline-node" key={stage.name}>
                      <span>{i + 1}</span>
                      <div>
                        <strong>{stage.name}</strong>
                        <small style={{ display: "block", color: "var(--muted)", fontSize: "0.76rem" }}>
                          {stage.time_ms} ms
                        </small>
                      </div>
                      {i < result.pipeline_stages.length - 1 && <Icon name="arrow" size={14} />}
                    </div>
                  ))
                ) : (
                  [
                    "Claim input",
                    "Hybrid retrieval",
                    "RRF fusion",
                    "Evidence scoring",
                    "LLM verification",
                    "NLI faithfulness check",
                    "Final verdict",
                  ].map((x, i) => (
                    <div className="pipeline-node" key={x}>
                      <span>{i + 1}</span>
                      <strong>{x}</strong>
                      {i < 6 && <Icon name="arrow" size={14} />}
                    </div>
                  ))
                )}
              </div>
            )}
          </section>
        </main>

        <aside className="result-aside">
          <section className="card analytics">
            <div className="aside-heading">
              <div>
                <h3>Evidence scoring</h3>
                <p>How sources were evaluated</p>
              </div>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: "0.75rem", padding: "0.2rem 0.5rem" }}
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              >
                {showTechnicalDetails ? "Simple view" : "Tech details"}
              </button>
            </div>
            <ScoreBar label="Relevance to claim" value={result.evidence_scoring?.relevance ?? 92} />
            <ScoreBar
              label="Source reputation"
              value={result.evidence_scoring?.credibility ?? 90}
              tone="green"
            />
            <ScoreBar
              label="Recency alignment"
              value={result.evidence_scoring?.temporal ?? 94}
              tone="amber"
            />
            {showTechnicalDetails && (
              <div className="formula" style={{ marginTop: "1rem" }}>
                <span>COMBINED SCORE FORMULA</span>
                <strong>{result.evidence_scoring?.combined ?? 92} / 100</strong>
                <code>{result.evidence_scoring?.formula || "0.60 × Rel. + 0.25 × Cred. + 0.15 × Temp."}</code>
              </div>
            )}
            <p className="fineprint">
              Scores measure source credibility and topical match, ensuring conclusions are based on trustworthy publications.
            </p>
          </section>

          <section className="card reliability">
            <div className="aside-heading">
              <div>
                <h3>Hallucination check</h3>
                <p>NLI truth-grounding guard</p>
              </div>
            </div>
            <div className="reliability-status">
              <Icon name="shield" size={20} />
              <div>
                <strong>
                  {result.is_supported ? "Verified Grounded" : "Needs More Proof"}
                </strong>
                <span>
                  {result.is_supported
                    ? "Explanation statements are strictly backed by the cited quotes"
                    : "System abstained to prevent guessing without evidence"}
                </span>
              </div>
            </div>
            <div className="metric-list">
              <div>
                <span>Source match rate</span>
                <strong>{result.faithfulness_score}%</strong>
              </div>
              <div>
                <span>Certainty score</span>
                <strong>{result.confidence}%</strong>
              </div>
              <div>
                <span>Fact-check guard</span>
                <strong className={result.nli_status === "Passed" ? "pass" : "fail"}>
                  {result.nli_status === "Passed" && <Icon name="check" size={13} />}
                  {" "}{result.nli_status === "Passed" ? "Passed (No Hallucination)" : "Flagged"}
                </strong>
              </div>
            </div>
            <p className="fineprint">
              VERITAS checks every sentence against the actual documents to prevent AI fabricated claims.
            </p>
          </section>

          <div className="demo-disclaimer">
            <Icon name="info" size={16} />
            <p>
              <strong>Scientific Notice:</strong> {result.disclaimer || "VERITAS assists human judgment."}
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
