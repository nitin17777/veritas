import { useEffect, useState } from "react";
import type { SettingsConfig } from "../types";
import { Icon } from "../components/Icons";
import { Logo } from "../components/Logo";
import { PageHeader } from "../components/PageHeader";

interface SettingsPageProps {
  settings: SettingsConfig;
  onUpdateSettings: (newSettings: Partial<SettingsConfig>) => void;
}

function Toggle({
  on = false,
  onChange,
}: {
  on?: boolean;
  onChange?: (val: boolean) => void;
}) {
  const [active, setActive] = useState(on);

  const handleClick = () => {
    const next = !active;
    setActive(next);
    onChange?.(next);
  };

  return (
    <button
      className={`toggle ${active ? "on" : ""}`}
      onClick={handleClick}
      aria-pressed={active}
      type="button"
    >
      <span />
    </button>
  );
}

export function SettingsPage({ settings, onUpdateSettings }: SettingsPageProps) {
  const [activeTab, setActiveTab] = useState<"profile" | "appearance" | "verification" | "notifications" | "about">("appearance");
  const [theme, setTheme] = useState(settings.theme || "Light");
  const [userName, setUserName] = useState(settings.userName || "Alex Morgan");
  const [userEmail, setUserEmail] = useState(settings.userEmail || "alex@research.example");
  const [workspaceName, setWorkspaceName] = useState(settings.workspaceName || "Research workspace");
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    const resolved =
      theme === "System"
        ? window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
        : theme.toLowerCase();
    document.documentElement.dataset.theme = resolved;
    onUpdateSettings({ theme });
  }, [theme]);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    onUpdateSettings({
      userName,
      userEmail,
      workspaceName,
    });
    setSaveSuccess(true);
    window.setTimeout(() => setSaveSuccess(false), 2500);
  };

  const initials = userName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase() || "AM";

  return (
    <div className="page settings-page">
      <PageHeader
        title="Settings"
        description="Manage your profile, preferences, and verification defaults."
      />

      <div className="settings-layout">
        <nav className="settings-nav">
          <button
            className={activeTab === "profile" ? "active" : ""}
            onClick={() => setActiveTab("profile")}
            type="button"
          >
            <Icon name="user" size={17} />
            Profile
          </button>
          <button
            className={activeTab === "appearance" ? "active" : ""}
            onClick={() => setActiveTab("appearance")}
            type="button"
          >
            <Icon name="moon" size={17} />
            Appearance
          </button>
          <button
            className={activeTab === "verification" ? "active" : ""}
            onClick={() => setActiveTab("verification")}
            type="button"
          >
            <Icon name="shield" size={17} />
            Verification
          </button>
          <button
            className={activeTab === "notifications" ? "active" : ""}
            onClick={() => setActiveTab("notifications")}
            type="button"
          >
            <Icon name="bell" size={17} />
            Notifications
          </button>
          <button
            className={activeTab === "about" ? "active" : ""}
            onClick={() => setActiveTab("about")}
            type="button"
          >
            <Icon name="info" size={17} />
            About
          </button>
        </nav>

        <main className="settings-content">
          {/* Appearance Section */}
          {(activeTab === "appearance" || activeTab === "profile") && (
            <section className="settings-section card">
              <div className="settings-heading">
                <div>
                  <h2>Theme preferences</h2>
                  <p>Choose how VERITAS appears in this workspace.</p>
                </div>
              </div>
              <div className="theme-options">
                {(["Light", "System", "Dark"] as const).map((option) => (
                  <button
                    className={theme === option ? "selected" : ""}
                    key={option}
                    onClick={() => setTheme(option)}
                    type="button"
                  >
                    <span className={`theme-preview ${option.toLowerCase()}`}>
                      <i />
                      <b />
                    </span>
                    <strong>{option}</strong>
                    <small>{option === "System" ? "Match device" : `${option} interface`}</small>
                  </button>
                ))}
              </div>
            </section>
          )}

          {/* Profile Section */}
          {(activeTab === "profile") && (
            <section className="settings-section card">
              <div className="settings-heading">
                <div>
                  <h2>Profile preferences</h2>
                  <p>Update your personal information and workspace details.</p>
                </div>
              </div>

              <div className="profile-edit">
                <div className="avatar large">{initials}</div>
                <div>
                  <button className="btn btn-secondary" type="button">Change photo</button>
                  <p>JPG or PNG. Maximum 2 MB.</p>
                </div>
              </div>

              <form onSubmit={handleSaveProfile}>
                <div className="form-grid">
                  <label>
                    Full name
                    <input
                      value={userName}
                      onChange={(e) => setUserName(e.target.value)}
                    />
                  </label>
                  <label>
                    Email address
                    <input
                      type="email"
                      value={userEmail}
                      onChange={(e) => setUserEmail(e.target.value)}
                    />
                  </label>
                  <label className="full">
                    Workspace name
                    <input
                      value={workspaceName}
                      onChange={(e) => setWorkspaceName(e.target.value)}
                    />
                  </label>
                </div>

                <div className="save-row">
                  {saveSuccess && (
                    <span style={{ color: "var(--green)", fontSize: "0.88rem", fontWeight: 500, marginRight: "auto" }}>
                      ✓ Changes saved successfully
                    </span>
                  )}
                  <button className="btn btn-primary" type="submit">
                    Save changes
                  </button>
                </div>
              </form>
            </section>
          )}

          {/* Verification Defaults */}
          {(activeTab === "verification" || activeTab === "profile") && (
            <section className="settings-section card">
              <div className="settings-heading">
                <div>
                  <h2>Default verification options</h2>
                  <p>Choose how new claim verifications are prepared.</p>
                </div>
              </div>

              <div className="setting-row">
                <div>
                  <strong>Default evidence retrieval mode</strong>
                  <span>Select default source for evidence gathering.</span>
                </div>
                <div className="select-wrap" style={{ minWidth: "180px" }}>
                  <select
                    value={settings.sourceMode}
                    onChange={(e) => onUpdateSettings({ sourceMode: e.target.value as "benchmark" | "live_search" })}
                  >
                    <option value="benchmark">Benchmark Corpus</option>
                    <option value="live_search">Live Web Search</option>
                  </select>
                  <Icon name="chevron" size={15} />
                </div>
              </div>

              <div className="setting-row">
                <div>
                  <strong>Default verification backend</strong>
                  <span>Engine powering LLM claim interpretation.</span>
                </div>
                <div className="select-wrap" style={{ minWidth: "180px" }}>
                  <select
                    value={settings.backend}
                    onChange={(e) => onUpdateSettings({ backend: e.target.value as "mock" | "api" })}
                  >
                    <option value="mock">Mock Engine (Fast)</option>
                    <option value="api">OpenAI (GPT-4o-mini)</option>
                  </select>
                  <Icon name="chevron" size={15} />
                </div>
              </div>

              <div className="setting-row">
                <div>
                  <strong>Include source credibility analysis</strong>
                  <span>Assess provenance and editorial transparency.</span>
                </div>
                <Toggle
                  on={settings.includeCredibility}
                  onChange={(val) => onUpdateSettings({ includeCredibility: val })}
                />
              </div>

              <div className="setting-row">
                <div>
                  <strong>Prioritize recent evidence</strong>
                  <span>Increase the influence of temporally relevant sources.</span>
                </div>
                <Toggle
                  on={settings.prioritizeRecent}
                  onChange={(val) => onUpdateSettings({ prioritizeRecent: val })}
                />
              </div>

              <div className="setting-row">
                <div>
                  <strong>Detailed result view</strong>
                  <span>Expand pipeline inspection by default.</span>
                </div>
                <Toggle
                  on={settings.detailedView}
                  onChange={(val) => onUpdateSettings({ detailedView: val })}
                />
              </div>
            </section>
          )}

          {/* Notifications */}
          {(activeTab === "notifications" || activeTab === "profile") && (
            <section className="settings-section card">
              <div className="settings-heading">
                <div>
                  <h2>Notifications</h2>
                  <p>Control system alerts and verification status notifications.</p>
                </div>
              </div>

              <div className="setting-row">
                <div>
                  <strong>Verification complete</strong>
                  <span>Alert when long-form analysis is ready.</span>
                </div>
                <Toggle
                  on={settings.notifyComplete}
                  onChange={(val) => onUpdateSettings({ notifyComplete: val })}
                />
              </div>

              <div className="setting-row">
                <div>
                  <strong>Product & benchmark updates</strong>
                  <span>Occasional methodology and model updates.</span>
                </div>
                <Toggle
                  on={settings.notifyUpdates}
                  onChange={(val) => onUpdateSettings({ notifyUpdates: val })}
                />
              </div>
            </section>
          )}

          {/* About */}
          {(activeTab === "about") && (
            <section className="settings-section card">
              <div className="settings-heading">
                <div>
                  <h2>About VERITAS</h2>
                  <p>Evidence-grounded RAG fact verification system.</p>
                </div>
              </div>
              <div style={{ padding: "0.5rem 0", lineHeight: 1.6, fontSize: "0.93rem" }}>
                <p>
                  <strong>VERITAS</strong> (Verification through Evidence Retrieval, Interpretation, Temporal Analysis, and Source-credibility scoring) is a scientific research platform for automated fake news and claim verification.
                </p>
                <div style={{ marginTop: "1rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                  <div><strong>Pipeline Version:</strong> v1.4.0</div>
                  <div><strong>UI Architecture:</strong> React 19 + Vite + Tailwind</div>
                  <div><strong>Retrieval:</strong> BM25 + FAISS Dense + RRF</div>
                  <div><strong>NLI Model:</strong> RoBERTa / DeBERTa NLI Gate</div>
                </div>
              </div>
            </section>
          )}

          <div className="version-info">
            <Logo />
            <span>Version 1.4.0 · Research preview</span>
            <a href="https://github.com/nitin17777/veritas" target="_blank" rel="noreferrer">
              Documentation
            </a>
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
          </div>
        </main>
      </div>
    </div>
  );
}
