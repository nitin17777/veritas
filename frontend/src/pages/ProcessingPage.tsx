import { useEffect, useState } from "react";
import type { VerificationResult } from "../types";
import { verifyClaim } from "../api";
import { Icon } from "../components/Icons";

interface ProcessingPageProps {
  claim: string;
  claimDate?: string;
  mode: "benchmark" | "live_search";
  backend: "mock" | "local" | "api";
  onDone: (result: VerificationResult) => void;
  onError: (errMsg: string) => void;
}

export function ProcessingPage({
  claim,
  claimDate,
  mode,
  backend,
  onDone,
  onError,
}: ProcessingPageProps) {
  const [step, setStep] = useState(1);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const steps = [
    "Extracting factual claims",
    "Retrieving relevant evidence",
    "Ranking sources & evaluating credibility",
    "Evidence-constrained AI verification",
    "Validating explanation faithfulness & NLI gate",
  ];

  // Start real verification API call
  useEffect(() => {
    let active = true;

    verifyClaim({
      claim,
      claim_date: claimDate,
      mode,
      backend,
    })
      .then((data) => {
        if (active) {
          setResult(data);
        }
      })
      .catch((err) => {
        if (active) {
          setError(err.message || "Failed to verify claim");
        }
      });

    return () => {
      active = false;
    };
  }, [claim, claimDate, mode, backend]);

  // Step advancement timer
  useEffect(() => {
    if (error) return;

    const timer = window.setInterval(() => {
      setStep((currentStep) => {
        if (currentStep < steps.length) {
          return currentStep + 1;
        } else if (result) {
          // If we reached the end and result is ready, finish
          window.clearInterval(timer);
          window.setTimeout(() => onDone(result), 400);
          return currentStep;
        }
        return currentStep;
      });
    }, 700);

    return () => window.clearInterval(timer);
  }, [result, error, onDone, steps.length]);

  // If result arrived and we're at or near end of steps, transition smoothly
  useEffect(() => {
    if (result && step >= steps.length) {
      const finishTimer = window.setTimeout(() => {
        onDone(result);
      }, 450);
      return () => window.clearTimeout(finishTimer);
    }
  }, [result, step, steps.length, onDone]);

  if (error) {
    return (
      <div className="page processing-page">
        <div className="processing-card card" style={{ borderColor: "var(--red)" }}>
          <div className="scan-mark" style={{ background: "var(--red-soft)", color: "var(--red)" }}>
            <Icon name="info" size={28} />
          </div>
          <span className="eyebrow" style={{ color: "var(--red)" }}>VERIFICATION ERROR</span>
          <h1>Unable to complete verification</h1>
          <p className="processing-claim">“{claim}”</p>
          <div style={{ margin: "1.5rem 0", padding: "1rem", background: "var(--red-soft)", color: "var(--red)", borderRadius: "8px", fontSize: "0.92rem", lineHeight: 1.5 }}>
            {error}
          </div>
          <button className="btn btn-primary" onClick={() => onError(error)} type="button">
            Back to verification
          </button>
        </div>
      </div>
    );
  }

  const progressPct = Math.min(100, Math.round((step / steps.length) * 100));

  return (
    <div className="page processing-page">
      <div className="processing-card card">
        <div className="scan-mark">
          <span />
          <Icon name="shield" size={28} />
        </div>
        <span className="eyebrow">VERIFICATION IN PROGRESS · {mode === "live_search" ? "LIVE SEARCH" : "BENCHMARK CORPUS"}</span>
        <h1>Investigating your claim</h1>
        <p className="processing-claim">“{claim}”</p>

        <div className="overall-progress">
          <span style={{ width: `${progressPct}%` }} />
        </div>

        <div className="step-list">
          {steps.map((label, i) => {
            const n = i + 1;
            const isComplete = n < step || (n === step && result !== null);
            const isActive = n === step && result === null;

            return (
              <div
                className={`process-step ${isComplete ? "complete" : isActive ? "active" : ""}`}
                key={label}
              >
                <span className="step-state">
                  {isComplete ? (
                    <Icon name="check" size={15} />
                  ) : isActive ? (
                    <span className="spinner" />
                  ) : (
                    n
                  )}
                </span>
                <div>
                  <strong>{label}</strong>
                  <small>
                    {isComplete ? "Complete" : isActive ? "In progress" : "Waiting"}
                  </small>
                </div>
              </div>
            );
          })}
        </div>

        <div className="processing-note">
          <Icon name="info" size={16} />
          <span>
            {mode === "live_search"
              ? "Searching live web sources, extracting passages, and scoring domain credibility."
              : "Scanning offline RAG index, computing credibility weights, and checking faithfulness."}
          </span>
        </div>
      </div>
    </div>
  );
}
