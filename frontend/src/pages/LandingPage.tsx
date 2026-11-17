import { useState, useEffect } from "react";
import { Icon } from "../components/Icons";

interface LandingPageProps {
  onEnter: () => void;
}

const FEATURES = [
  {
    icon: "search",
    title: "Hybrid Retrieval Engine",
    desc: "BM25 keyword search fused with FAISS semantic vectors via Reciprocal Rank Fusion to surface the most relevant evidence for every claim.",
    accent: "#426da9",
    bg: "#edf3fb",
  },
  {
    icon: "shield",
    title: "Source Credibility Scoring",
    desc: "Every source is scored on domain authority, editorial standards, and historical accuracy. Wikipedia, Reuters, government sites score highest.",
    accent: "#347961",
    bg: "#eaf5f0",
  },
  {
    icon: "verify",
    title: "Temporal Re-ranking",
    desc: "Recent documents are weighted higher for time-sensitive claims. Older but authoritative sources still contribute to context.",
    accent: "#a5681d",
    bg: "#fbf2e5",
  },
  {
    icon: "research",
    title: "NLI Faithfulness Guard",
    desc: "Natural Language Inference checks every explanation sentence against cited quotes. VERITAS abstains rather than hallucinate.",
    accent: "#a94f50",
    bg: "#faeded",
  },
  {
    icon: "file",
    title: "Evidence-Constrained LLM",
    desc: "The LLM writes explanations only from the retrieved documents — never from its training data. Closed-world by design.",
    accent: "#426da9",
    bg: "#edf3fb",
  },
  {
    icon: "check",
    title: "Transparent Pipeline",
    desc: "Every stage — retrieval, scoring, verification, NLI — is timed and exposed. You see exactly how the verdict was reached.",
    accent: "#347961",
    bg: "#eaf5f0",
  },
];

const STEPS = [
  {
    num: "01",
    title: "Submit a claim",
    desc: "Type any factual statement or paste a news headline. VERITAS works with political claims, scientific statements, historical facts, and more.",
  },
  {
    num: "02",
    title: "Hybrid retrieval fires",
    desc: "BM25 and FAISS search a curated benchmark corpus simultaneously. Up to 20 candidate passages are fused and re-ranked by relevance.",
  },
  {
    num: "03",
    title: "Sources are scored",
    desc: "Each passage earns a combined score: 60% relevance · 25% credibility · 15% temporal recency. Only the highest-scoring sources inform the verdict.",
  },
  {
    num: "04",
    title: "LLM writes the verdict",
    desc: "A constrained LLM synthesises only from the top evidence. It outputs TRUE, FALSE, PARTIALLY TRUE, MISLEADING or UNVERIFIED.",
  },
  {
    num: "05",
    title: "NLI faithfulness check",
    desc: "Every explanation sentence is cross-checked against the cited quotes. If the system cannot verify faithfully, it abstains.",
  },
];

const STATS = [
  { value: "5-stage", label: "Verification pipeline" },
  { value: "< 3 s", label: "Avg. response time" },
  { value: "0%", label: "Hallucination tolerance" },
  { value: "100%", label: "Evidence-grounded" },
];

export function LandingPage({ onEnter }: LandingPageProps) {
  const [typedText, setTypedText] = useState("");
  const exampleClaim = "Climate change is primarily caused by human activities.";

  useEffect(() => {
    let i = 0;
    const timer = setInterval(() => {
      if (i <= exampleClaim.length) {
        setTypedText(exampleClaim.slice(0, i));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 42);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="landing-page">
      {/* ───── Navbar ───── */}
      <header className="landing-nav">
        <div className="landing-nav-inner">
          <div className="landing-logo">
            <span className="landing-logo-mark">
              <Icon name="check" size={16} />
            </span>
            <span>VERITAS</span>
          </div>
          <nav className="landing-links" aria-label="Landing navigation">
            <a href="#how-it-works">How it works</a>
            <a href="#features">Features</a>
          </nav>
          <button className="landing-cta-btn" onClick={onEnter}>
            Start Verifying
            <Icon name="arrow" size={14} />
          </button>
        </div>
      </header>

      {/* ───── Hero ───── */}
      <section className="landing-hero">
        <div className="landing-hero-bg" aria-hidden="true">
          <div className="hero-orb hero-orb-1" />
          <div className="hero-orb hero-orb-2" />
          <div className="hero-grid" />
        </div>
        <div className="landing-container">
          <div className="hero-badge">
            <span className="hero-badge-dot" />
            Evidence-Grounded · Anti-Hallucination · Open Pipeline
          </div>
          <h1 className="hero-title">
            Fact-verification you can
            <br />
            <span className="hero-gradient-text">actually trust</span>
          </h1>
          <p className="hero-subtitle">
            VERITAS doesn't guess. It retrieves real documents, scores every source, and writes
            verdicts only from what the evidence says. If the data is absent, it abstains.
          </p>
          <div className="hero-actions">
            <button className="hero-btn-primary" onClick={onEnter}>
              Start verifying for free
              <Icon name="arrow" size={16} />
            </button>
            <button className="hero-btn-ghost" onClick={onEnter}>
              <Icon name="verify" size={16} />
              See a live example
            </button>
          </div>

          {/* Live demo preview */}
          <div className="hero-demo-card">
            <div className="hero-demo-header">
              <span className="demo-dot red" />
              <span className="demo-dot amber" />
              <span className="demo-dot green" />
              <span className="demo-label">VERITAS — Live verification</span>
            </div>
            <div className="hero-demo-body">
              <div className="demo-claim-row">
                <span className="demo-eyebrow">Claim submitted</span>
                <p className="demo-claim-text">
                  {typedText}
                  <span className="demo-cursor">|</span>
                </p>
              </div>
              <div className="demo-verdict-row">
                <div className="demo-verdict-chip demo-verdict-true">
                  <Icon name="check" size={13} />
                  TRUE · 94% confidence
                </div>
                <div className="demo-evidence-chips">
                  <span className="demo-evid-chip">NASA.gov</span>
                  <span className="demo-evid-chip">IPCC AR6</span>
                  <span className="demo-evid-chip">Nature.com</span>
                  <span className="demo-evid-chip">+4 sources</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ───── Stats ───── */}
      <section className="landing-stats">
        <div className="landing-container">
          <div className="stats-grid">
            {STATS.map((s) => (
              <div className="stat-item" key={s.label}>
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ───── How it works ───── */}
      <section className="landing-section" id="how-it-works">
        <div className="landing-container">
          <div className="section-label">How it works</div>
          <h2 className="section-title">A transparent 5-stage pipeline</h2>
          <p className="section-subtitle">
            No black boxes. Every step is logged, timed, and shown to you.
          </p>
          <div className="steps-list">
            {STEPS.map((step, i) => (
              <div className="step-item" key={step.num}>
                <div className="step-left">
                  <div className="step-num">{step.num}</div>
                  {i < STEPS.length - 1 && <div className="step-line" />}
                </div>
                <div className="step-content">
                  <h3>{step.title}</h3>
                  <p>{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ───── Features ───── */}
      <section className="landing-section landing-section-alt" id="features">
        <div className="landing-container">
          <div className="section-label">Core technology</div>
          <h2 className="section-title">Built for accuracy, not speed</h2>
          <p className="section-subtitle">
            Six layers of verification designed to surface ground truth, not approximate it.
          </p>
          <div className="features-grid">
            {FEATURES.map((f) => (
              <div className="feature-card" key={f.title}>
                <div
                  className="feature-icon"
                  style={{ background: f.bg, color: f.accent }}
                >
                  <Icon name={f.icon} size={22} />
                </div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ───── Trust section ───── */}
      <section className="landing-section landing-trust">
        <div className="landing-container">
          <div className="trust-inner">
            <div className="trust-text">
              <div className="section-label">Why VERITAS?</div>
              <h2 className="section-title" style={{ marginBottom: "1rem" }}>
                Search engines give results.
                <br />
                VERITAS gives verdicts.
              </h2>
              <p style={{ color: "var(--muted)", lineHeight: 1.7, maxWidth: 480 }}>
                Google returns a list of links — you still have to read, judge, and synthesise.
                VERITAS reads the sources for you, scores them on credibility and recency, runs the
                claim through five verification stages, and tells you exactly what the evidence says.
                Not what an AI guesses. What the documents prove.
              </p>
              <ul className="trust-list">
                <li>
                  <span className="trust-check"><Icon name="check" size={13} /></span>
                  Verdict backed by source quotes you can read
                </li>
                <li>
                  <span className="trust-check"><Icon name="check" size={13} /></span>
                  Source credibility scored by domain authority
                </li>
                <li>
                  <span className="trust-check"><Icon name="check" size={13} /></span>
                  NLI guard prevents hallucinated explanations
                </li>
                <li>
                  <span className="trust-check"><Icon name="check" size={13} /></span>
                  Abstains when evidence is insufficient
                </li>
              </ul>
              <button className="hero-btn-primary" onClick={onEnter} style={{ marginTop: "2rem" }}>
                Try VERITAS now <Icon name="arrow" size={15} />
              </button>
            </div>
            <div className="trust-visual">
              <div className="trust-card">
                <div className="trust-card-header">
                  <span className="trust-card-label">Verification result</span>
                  <span className="trust-verdict-chip">TRUE</span>
                </div>
                <p className="trust-claim">"The Great Wall of China is not visible from space with the naked eye."</p>
                <div className="trust-score-block">
                  <div className="trust-score-row">
                    <span>Relevance</span>
                    <div className="trust-bar"><div style={{ width: "91%" }} /></div>
                    <strong>91%</strong>
                  </div>
                  <div className="trust-score-row">
                    <span>Credibility</span>
                    <div className="trust-bar green"><div style={{ width: "88%" }} /></div>
                    <strong>88%</strong>
                  </div>
                  <div className="trust-score-row">
                    <span>Recency</span>
                    <div className="trust-bar amber"><div style={{ width: "79%" }} /></div>
                    <strong>79%</strong>
                  </div>
                </div>
                <div className="trust-sources">
                  <span>Sources: NASA.gov · National Geographic · ESA</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ───── CTA ───── */}
      <section className="landing-cta-section">
        <div className="landing-container">
          <div className="cta-inner">
            <div className="cta-orb" aria-hidden="true" />
            <div className="section-label" style={{ color: "rgba(255,255,255,0.65)" }}>
              Get started
            </div>
            <h2 className="cta-title">Stop guessing. Start verifying.</h2>
            <p className="cta-subtitle">
              Submit your first claim in seconds. No sign-up required.
            </p>
            <div className="cta-actions">
              <button className="cta-btn-primary" onClick={onEnter}>
                Open VERITAS <Icon name="arrow" size={16} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ───── Footer ───── */}
      <footer className="landing-footer">
        <div className="landing-container">
          <div className="footer-inner">
            <div className="landing-logo footer-logo">
              <span className="landing-logo-mark">
                <Icon name="check" size={14} />
              </span>
              <span>VERITAS</span>
            </div>
            <p className="footer-tagline">Evidence-Grounded Fact Verification</p>
            <p className="footer-copy">
              © {new Date().getFullYear()} VERITAS · Research prototype · Not for clinical or legal use
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
