import { useEffect, useState } from "react";
import type { HistoryItem, Page, SettingsConfig, VerificationResult } from "./types";
import { clearHistory, getHistory } from "./api";
import { Sidebar } from "./components/Sidebar";
import { HistoryPage } from "./pages/HistoryPage";
import { LandingPage } from "./pages/LandingPage";
import { ProcessingPage } from "./pages/ProcessingPage";
import { ResearchPage } from "./pages/ResearchPage";
import { ResultsPage } from "./pages/ResultsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { VerifyPage } from "./pages/VerifyPage";

const DEFAULT_SETTINGS: SettingsConfig = {
  theme: "Light",
  sourceMode: "benchmark",
  backend: "mock",
  includeCredibility: true,
  prioritizeRecent: true,
  detailedView: false,
  userName: "Alex Morgan",
  userEmail: "alex@research.example",
  workspaceName: "Research workspace",
  notifyComplete: true,
  notifyUpdates: false,
};

function getSavedSettings(): SettingsConfig {
  try {
    const raw = localStorage.getItem("veritas_settings");
    if (raw) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
    }
  } catch {}
  return DEFAULT_SETTINGS;
}

export default function App() {
  const [page, setPage] = useState<Page>("landing");
  const [claimToVerify, setClaimToVerify] = useState<string>("City X banned all private vehicles from its centre starting January 2026.");
  const [claimDate, setClaimDate] = useState<string | undefined>();
  const [verifyMode, setVerifyMode] = useState<"benchmark" | "live_search">("benchmark");
  const [verifyBackend, setVerifyBackend] = useState<"mock" | "local" | "api">("mock");
  const [currentResult, setCurrentResult] = useState<VerificationResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [settings, setSettings] = useState<SettingsConfig>(getSavedSettings);

  // Apply theme on initial load
  useEffect(() => {
    const resolved =
      settings.theme === "System"
        ? window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
        : settings.theme.toLowerCase();
    document.documentElement.dataset.theme = resolved;
  }, [settings.theme]);

  // Load history from API on mount
  useEffect(() => {
    getHistory()
      .then((items) => {
        setHistory(items);
        if (items.length > 0 && !currentResult) {
          setCurrentResult(items[0].result);
        }
      })
      .catch((err) => {
        console.warn("Could not load history from API:", err);
      });
  }, []);

  const updateSettings = (partial: Partial<SettingsConfig>) => {
    const updated = { ...settings, ...partial };
    setSettings(updated);
    try {
      localStorage.setItem("veritas_settings", JSON.stringify(updated));
    } catch {}
  };

  const navigate = (p: Page) => {
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleStartVerification = (params: {
    claim: string;
    claim_date?: string;
    mode: "benchmark" | "live_search";
    backend: "mock" | "local" | "api";
  }) => {
    setClaimToVerify(params.claim);
    setClaimDate(params.claim_date);
    setVerifyMode(params.mode);
    setVerifyBackend(params.backend);
    navigate("processing");
  };

  const handleVerificationDone = (result: VerificationResult) => {
    setCurrentResult(result);
    // Refresh history
    getHistory()
      .then((items) => setHistory(items))
      .catch(() => {});
    navigate("results");
  };

  const handleVerificationError = () => {
    navigate("verify");
  };

  const handleViewResult = (result: VerificationResult) => {
    setCurrentResult(result);
    navigate("results");
  };

  const handleClearHistory = async () => {
    try {
      await clearHistory();
      setHistory([]);
    } catch (err) {
      console.warn("Failed to clear history:", err);
    }
  };

  return (
    <div className="app-shell">
      {/* Landing page — full screen, no sidebar */}
      {page === "landing" && (
        <LandingPage onEnter={() => navigate("verify")} />
      )}

      {/* App shell with sidebar — all non-landing pages */}
      {page !== "landing" && (
        <>
          <Sidebar page={page} setPage={navigate} settings={settings} />

          <div className="main-shell">
            {page === "verify" && (
              <VerifyPage
                onStart={handleStartVerification}
                onResult={handleViewResult}
                onHistory={() => navigate("history")}
                history={history}
                settings={settings}
              />
            )}

            {page === "processing" && (
              <ProcessingPage
                claim={claimToVerify}
                claimDate={claimDate}
                mode={verifyMode}
                backend={verifyBackend}
                onDone={handleVerificationDone}
                onError={handleVerificationError}
              />
            )}

            {page === "results" && currentResult && (
              <ResultsPage result={currentResult} />
            )}

            {page === "results" && !currentResult && (
              <div className="page" style={{ padding: "3rem", textAlign: "center" }}>
                <h2>No verification selected</h2>
                <p style={{ color: "var(--muted)", margin: "1rem 0" }}>
                  Please verify a new claim or choose one from your history.
                </p>
                <button className="btn btn-primary" onClick={() => navigate("verify")} type="button">
                  Verify a claim
                </button>
              </div>
            )}

            {page === "history" && (
              <HistoryPage
                history={history}
                onResult={handleViewResult}
                onNew={() => navigate("verify")}
                onClear={handleClearHistory}
              />
            )}

            {page === "research" && <ResearchPage />}

            {page === "settings" && (
              <SettingsPage
                settings={settings}
                onUpdateSettings={updateSettings}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
