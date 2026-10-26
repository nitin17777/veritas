"""VERITAS: Evidence-Grounded Fact Verification Interactive Demo (P11).

An interactive Streamlit interface powered by the VERITAS verification pipeline:
- Two operational modes: Benchmark Corpus (pre-built indexes) & Live Web Search.
- Strict closed-world evidence grounding and source credibility tiers.
- Visible Unverified abstention state.
- Evidence cards with domain, date, score, snippet, and canonical URL.
- Long text claim extraction (P5) for articles or social media posts.
- Transparent disclaimer: "VERITAS assists human judgment and does not guarantee factual correctness."
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from veritas.demo.engine import (
    DISCLAIMER_TEXT,
    DemoVerificationResponse,
    VeritasDemoEngine,
)

logger = logging.getLogger("veritas.streamlit_app")

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VERITAS Fact Verification",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium appearance
st.markdown(
    """
    <style>
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .verdict-true {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    .verdict-partially-true {
        background-color: #fefce8;
        border: 2px solid #eab308;
        color: #854d0e;
    }
    .verdict-misleading {
        background-color: #fff7ed;
        border: 2px solid #f97316;
        color: #9a3412;
    }
    .verdict-false {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .verdict-unverified {
        background-color: #f5f3ff;
        border: 2px solid #8b5cf6;
        color: #5b21b6;
    }
    .evidence-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .evidence-card-cited {
        border-left: 5px solid #2563eb;
        background-color: #f8fafc;
    }
    .badge-corpus {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-live {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .disclaimer-banner {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #334155;
        border-radius: 4px;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Engine Initializer & Query Cache
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_engine() -> VeritasDemoEngine:
    """Initialize and cache the VERITAS demo verification engine."""
    return VeritasDemoEngine(config_path="configs/demo.yaml")


@st.cache_data(show_spinner=False, ttl=3600)
def cached_verify_claim(
    claim: str,
    claim_date: str | None,
    mode: str,
) -> dict:
    """Cache verification results for repeated identical queries."""
    engine = get_engine()
    res = engine.verify_claim(claim=claim, claim_date=claim_date, mode=mode)
    return res.to_dict()


# ---------------------------------------------------------------------------
# Sidebar Controls & Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://raw.githubusercontent.com/google/material-design-icons/master/png/action/verified_user/materialicons/48dp/2x/baseline_verified_user_black_48dp.png", width=48)
    st.title("VERITAS Controls")

    st.markdown("### Operational Mode")
    mode_selection = st.radio(
        "Retrieval Source",
        options=["Benchmark Corpus", "Live Web Search"],
        index=0,
        help="Benchmark mode searches local pre-built index / curated corpus. Live Web Search queries the internet in real time.",
    )
    mode_key = "benchmark" if mode_selection == "Benchmark Corpus" else "live_search"

    st.markdown("---")
    st.markdown("### Verification Model")
    model_backend = st.selectbox(
        "LLM Backend",
        options=["mock", "api", "local"],
        index=0,
        help="Use 'mock' for fast offline testing. Switch to 'api' (OpenAI/Anthropic) or 'local' (Ollama) with your API key set in your environment.",
    )

    st.markdown("---")
    st.markdown("### Safety & Faithfulness Gate")
    use_abstention = st.toggle("Abstention Gate", value=True, help="Downgrade to 'Unverified' if explanation fails NLI faithfulness check.")
    min_confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.60, 0.05)
    min_faithfulness = st.slider("Faithfulness Threshold", 0.0, 1.0, 0.70, 0.05)

    st.markdown("---")
    st.markdown(
        f"<div class='disclaimer-banner'><strong>Disclaimer</strong>: {DISCLAIMER_TEXT}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Preset Sample Claims
# ---------------------------------------------------------------------------

SAMPLE_PRESETS = {
    "Select a preset sample...": "",
    "Efficacy of COVID-19 mRNA vaccines (Health)": "COVID-19 mRNA vaccines significantly reduce the risk of severe hospitalizations.",
    "US BLS Nonfarm Payrolls (Economy)": "The US Bureau of Labor Statistics reported nonfarm payrolls rose by 275,000 in February 2024.",
    "Moon Geological Composition (Science)": "NASA confirmed the Moon is entirely composed of aged dairy cheddar cheese.",
    "Mount Everest vs K2 Elevation (Geography)": "Mount Everest is significantly shorter than K2 in total elevation above sea level.",
    "Long News Article (Triggers P5 Claim Extraction)": (
        "In a joint briefing on Tuesday, the Federal Reserve indicated interest rates may remain elevated "
        "through the third quarter. Meanwhile, European Union regulators opened a formal investigation into "
        "artificial intelligence training practices under the new Digital Services Act. Labor statistics show "
        "unemployment fell to 3.8 percent in major tech hubs, though overall industrial manufacturing output "
        "contracted by 0.4 percent month-over-month according to national bank records."
    ),
}

# ---------------------------------------------------------------------------
# Main Page Header & Input Form
# ---------------------------------------------------------------------------

st.markdown("<h1 class='main-header'>⚖️ VERITAS Fact Verification Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Evidence-grounded automated verification with source credibility tiers, "
    "temporal alignment, and NLI explanation faithfulness checking.</p>",
    unsafe_allow_html=True,
)

# Preset Selector
selected_preset = st.selectbox("Sample Demonstrations", options=list(SAMPLE_PRESETS.keys()), index=0)
default_text = SAMPLE_PRESETS[selected_preset] if selected_preset != "Select a preset sample..." else ""

# Input Form
with st.form("verification_form"):
    claim_input = st.text_area(
        "Enter a factual statement, article excerpt, or social media post to verify:",
        value=default_text,
        height=130,
        placeholder="e.g. The Federal Reserve raised the benchmark interest rate by 25 basis points in July 2023.",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        claim_date_input = st.text_input(
            "Temporal Anchor Date (Optional, YYYY-MM-DD)",
            placeholder="2023-07-26",
            help="Grounds the statement temporally for recency and decay scoring.",
        )
    with col2:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("🔍 Verify Claim", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Verification Logic & Results Display
# ---------------------------------------------------------------------------

if submitted:
    if not claim_input.strip():
        st.warning("⚠️ Please provide a statement or article to verify.")
    else:
        engine = get_engine()

        # Check if long text triggers P5 Claim Extraction
        is_long = engine.is_long_text(claim_input)

        if is_long:
            st.info("ℹ️ Long text detected (>30 words). Extracting check-worthy claims via P5 extractor...")
            with st.spinner("Extracting verifiable factual claims..."):
                extracted_claims = engine.extract_claims(claim_input)

            if extracted_claims:
                st.markdown(f"### Extracted Claims ({len(extracted_claims)})")
                st.markdown("Select an extracted claim below to inspect its evidence and verification verdict:")

                for idx, ec in enumerate(extracted_claims, start=1):
                    with st.expander(f"Claim #{idx}: {ec.claim}", expanded=(idx == 1)):
                        st.markdown(f"**Targeted Search Queries**: `{', '.join(ec.queries)}`")
                        with st.spinner(f"Verifying Claim #{idx}..."):
                            res_dict = cached_verify_claim(
                                claim=ec.claim,
                                claim_date=claim_date_input.strip() or None,
                                mode=mode_key,
                            )
                            # Render verdict
                            lbl = res_dict["label"]
                            conf = res_dict["confidence"]
                            css_class = f"verdict-{lbl.lower().replace(' ', '-')}"
                            st.markdown(
                                f"<div class='verdict-box {css_class}'>"
                                f"<span>Verdict: <strong>{lbl}</strong></span>"
                                f"<span>Confidence: <strong>{conf:.1%}</strong></span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Explanation**: {res_dict['explanation']}")

            else:
                st.warning("Could not extract distinct verifiable factual claims from provided text.")

        else:
            # Single Claim Verification
            with st.spinner(f"Verifying claim via {mode_selection}..."):
                res_dict = cached_verify_claim(
                    claim=claim_input.strip(),
                    claim_date=claim_date_input.strip() or None,
                    mode=mode_key,
                )

            lbl = res_dict["label"]
            conf = res_dict["confidence"]
            exp = res_dict["explanation"]
            cards = res_dict.get("evidence_cards", [])
            origin_mode = res_dict.get("source_mode", mode_selection)

            # 1. Verdict Banner
            css_class = f"verdict-{lbl.lower().replace(' ', '-')}"
            icon = {
                "True": "✅",
                "Partially True": "⚠️",
                "Misleading": "⚡",
                "False": "❌",
                "Unverified": "🛡️",
            }.get(lbl, "⚖️")

            st.markdown(
                f"<div class='verdict-box {css_class}'>"
                f"<span>{icon} Final Verdict: <strong>{lbl}</strong></span>"
                f"<span>Model Confidence: <strong>{conf:.1%}</strong></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Confidence Progress Bar
            st.progress(float(conf))

            # Visible Unverified / Abstention Explanation
            if lbl == "Unverified":
                abst_reasons = res_dict.get("abstention_reasons", [])
                reasons_text = "; ".join(abst_reasons) if abst_reasons else "The supplied evidence is insufficient or conflicting."
                st.info(f"**Unverified Status**: {reasons_text}")

            # 2. Readable Explanation
            st.markdown("### Verifier Explanation")
            st.markdown(f"> {exp}")

            # 3. Evidence Cards
            st.markdown("### Retrieved Evidence Passages")
            badge_class = "badge-live" if "Live" in origin_mode else "badge-corpus"
            st.markdown(
                f"Showing **{len(cards)}** evidence candidates "
                f"<span class='{badge_class}'>Origin: {origin_mode}</span>",
                unsafe_allow_html=True,
            )

            if not cards:
                st.markdown("_No evidence passages were retrieved for this query._")
            else:
                for idx, c in enumerate(cards, start=1):
                    card_cited_class = "evidence-card-cited" if c.get("is_cited") else ""
                    cited_badge = " <span style='color:#2563eb; font-weight:bold;'>[CITED IN VERDICT]</span>" if c.get("is_cited") else ""

                    url_link = f"<a href='{c['url']}' target='_blank'>🔗 Source Link</a>" if c.get("url") else "<em>Local Corpus</em>"
                    date_str = f"📅 {c['date']}" if c.get("date") else "📅 Date Unspecified"

                    st.markdown(
                        f"""
                        <div class="evidence-card {card_cited_class}">
                            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                                <span><strong>#{idx} {c['domain']}</strong>{cited_badge}</span>
                                <span>{date_str} | Relevance Score: <strong>{c['score']:.4f}</strong></span>
                            </div>
                            <p style="font-size:0.95rem; color:#1e293b; margin-bottom:0.5rem;">"{c['snippet']}"</p>
                            <div style="font-size:0.8rem; color:#64748b;">
                                {url_link} | Passage ID: <code>{c['passage_id']}</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 4. Latency & Provenance Expander
            with st.expander("⏱️ System Latency & Provenance Tracing"):
                timings = res_dict.get("stage_timings", {})
                if timings:
                    t_cols = st.columns(len(timings))
                    for idx, (stage, sec) in enumerate(timings.items()):
                        with t_cols[idx]:
                            st.metric(label=stage.replace("_", " ").title(), value=f"{sec:.3f}s")
                else:
                    st.write("No granular timings recorded for this query.")

                st.write(f"**Parse Failure Flag**: `{res_dict.get('parse_failed', False)}`")
                if res_dict.get("faithfulness_score") is not None:
                    st.write(f"**NLI Explanation Faithfulness Score**: `{res_dict['faithfulness_score']:.2f}`")

# Footer Disclaimer
st.markdown("---")
st.markdown(
    f"<div class='disclaimer-banner'><strong>Note</strong>: {DISCLAIMER_TEXT}</div>",
    unsafe_allow_html=True,
)
