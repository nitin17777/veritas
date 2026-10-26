"""VERITAS: modern, interactive fact-verification interface.

Aurora hero, one-click example claims, animated confidence ring, tabbed
results (Explanation / Sources / Pipeline) and a cited-only source filter.
Backend pipelines, scoring and data contracts are unchanged.
Requires Streamlit >= 1.39 (uses keyed containers for styling).
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
import sys
import time

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

st.set_page_config(
    page_title="VERITAS · Is it true?",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    :root {
        --bg:#0a0c16; --bg2:#10132a;
        --glass:rgba(255,255,255,.045); --glass-hi:rgba(255,255,255,.08);
        --line:rgba(255,255,255,.09); --line-hi:rgba(255,255,255,.2);
        --ink:#f4f5fb; --ink2:#b4b9d0; --ink3:#7d83a3;
        --a1:#8b7bff; --a2:#4fd1c5; --a3:#ff7eb6;
        --grad:linear-gradient(120deg,#8b7bff 0%,#4fd1c5 100%);
        --c-true:#4ade80; --c-partial:#facc15; --c-mislead:#fb923c; --c-false:#f87171; --c-unver:#a5b4fc;
        --display:'Sora',sans-serif; --body:'DM Sans',-apple-system,'Segoe UI',sans-serif;
        --r:16px;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background:
            radial-gradient(900px 480px at 12% -8%, rgba(139,123,255,.22), transparent 60%),
            radial-gradient(800px 460px at 92% 0%, rgba(79,209,197,.16), transparent 60%),
            var(--bg) !important;
        background-attachment: fixed !important;
        color: var(--ink) !important; font-family: var(--body) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="block-container"] { padding: 2rem 2rem 4rem !important; max-width: 1040px !important; }
    p, li, label, span, div { font-family: var(--body); }
    *:focus-visible { outline: 2px solid var(--a2) !important; outline-offset: 2px; }

    /* Nav + hero */
    .nav { display:flex; justify-content:space-between; align-items:center; margin-bottom:2.6rem; gap:1rem; flex-wrap:wrap; }
    .logo { display:flex; align-items:center; gap:.65rem; font-family:var(--display); font-weight:700; font-size:1.15rem; letter-spacing:.02em; }
    .logo i { width:30px; height:30px; border-radius:9px; background:var(--grad); display:grid; place-items:center; font-style:normal; font-size:.95rem; box-shadow:0 0 12px rgba(79,209,197,.4); }
    .pills { display:flex; gap:.5rem; flex-wrap:wrap; }
    .pill { font-size:.8rem; color:var(--ink2); background:var(--glass); border:1px solid var(--line); padding:.3rem .8rem; border-radius:999px; backdrop-filter: blur(8px); }
    .pill b { color:var(--ink); font-weight:600; }
    .hero { text-align:center; margin:0 auto 2.2rem; max-width:720px; }
    .hero h1 { font-family:var(--display); font-weight:800; font-size:clamp(2.2rem,5.2vw,3.4rem); line-height:1.08; letter-spacing:-.03em; margin:0 0 .85rem; color:var(--ink); }
    .hero h1 em { font-style:normal; background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero p { color:var(--ink2); font-size:1.05rem; line-height:1.65; margin:0; max-width:56ch; margin-left:auto; margin-right:auto; }

    /* Example chips row */
    .try-label { text-align:center; color:var(--ink3); font-size:.84rem; margin:1.2rem 0 .55rem; letter-spacing:.01em; }
    .chips-row { display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin-bottom:1.4rem; }

    /* All secondary / chip buttons */
    .stButton > button,
    div[data-testid="column"] .stButton > button,
    button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
        border-radius: 999px !important;
        font-family: var(--body) !important;
        font-weight: 500 !important;
        font-size: .86rem !important;
        padding: .4rem .95rem !important;
        min-height: 0 !important;
        white-space: nowrap !important;
        transition: all .15s ease !important;
        box-shadow: none !important;
    }
    .stButton > button p,
    div[data-testid="column"] .stButton > button p,
    button[kind="secondary"] p,
    [data-testid="baseButton-secondary"] p {
        color: var(--ink) !important;
        font-size: .86rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .stButton > button:hover,
    div[data-testid="column"] .stButton > button:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: var(--a2) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(79, 209, 197, 0.22) !important;
    }
    .stButton > button:hover p,
    div[data-testid="column"] .stButton > button:hover p,
    button[kind="secondary"]:hover p,
    [data-testid="baseButton-secondary"]:hover p {
        color: #ffffff !important;
    }

    /* Composer */
    [data-testid="stForm"] {
        background: linear-gradient(var(--bg2),var(--bg2)) padding-box, var(--grad) border-box !important;
        border: 1.5px solid transparent !important; border-radius: 22px !important; padding: 1.3rem 1.4rem 1.1rem !important;
        box-shadow: 0 20px 60px rgba(0,0,0,.45), 0 0 80px rgba(139,123,255,.12) !important;
    }
    [data-baseweb="textarea"],
    [data-baseweb="textarea"] > div,
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    .stTextArea textarea,
    .stTextInput input {
        background: rgba(255, 255, 255, 0.04) !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        font-family: var(--body) !important;
        font-size: 1.02rem !important;
        line-height: 1.6 !important;
    }
    [data-baseweb="textarea"]:focus-within,
    [data-baseweb="input"]:focus-within,
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--a2) !important;
        box-shadow: 0 0 0 3px rgba(79, 209, 197, 0.22) !important;
    }
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: var(--ink3) !important;
        -webkit-text-fill-color: var(--ink3) !important;
    }
    .composer-top { display:flex; justify-content:space-between; font-size:.85rem; color:var(--ink3); margin-bottom:.5rem; flex-wrap:wrap; gap:.4rem; }
    .composer-top b { color:var(--a2); font-weight:600; }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        border-radius: 12px !important; font-weight: 600 !important; font-size: .96rem !important; padding: .65rem 1.2rem !important; transition: all .15s !important;
    }
    .stFormSubmitButton > button[kind="primaryFormSubmit"] {
        background: var(--grad) !important; border: 0 !important; border-radius: 12px !important; box-shadow: 0 8px 24px rgba(139,123,255,.35) !important;
    }
    .stFormSubmitButton > button[kind="primaryFormSubmit"] p {
        color: #0a0c16 !important; font-weight: 700 !important; font-size: .98rem !important;
    }
    .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-2px); box-shadow: 0 12px 30px rgba(79,209,197,.45) !important;
    }
    .stFormSubmitButton > button[kind="secondaryFormSubmit"], .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.04) !important; color: var(--ink2) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
    }
    .stFormSubmitButton > button[kind="secondaryFormSubmit"] p, .stDownloadButton > button p {
        color: var(--ink2) !important; font-weight: 600 !important;
    }
    .stFormSubmitButton > button[kind="secondaryFormSubmit"]:hover, .stDownloadButton > button:hover {
        border-color: var(--line-hi) !important; color: var(--ink) !important; background: rgba(255, 255, 255, 0.08) !important;
    }
    .stFormSubmitButton > button[kind="secondaryFormSubmit"]:hover p, .stDownloadButton > button:hover p {
        color: var(--ink) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background:var(--bg2) !important; border-right:1px solid var(--line) !important; }
    [data-testid="stSidebar"] h3 { font-family:var(--display) !important; font-size:.95rem !important; color:var(--ink) !important; margin:1.1rem 0 .3rem !important; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p { color:var(--ink2) !important; font-size:.88rem !important; }
    [data-testid="stSidebar"] hr { border-color:var(--line) !important; }
    [data-baseweb="select"], [data-baseweb="select"] > div { background:rgba(255,255,255,.05) !important; border-color:var(--line) !important; border-radius:10px !important; color:var(--ink) !important; }
    [data-baseweb="select"] * { color:var(--ink) !important; }
    [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] { background:var(--bg2) !important; border:1px solid var(--line) !important; border-radius:10px !important; }
    li[role="option"] { color:var(--ink2) !important; }
    li[role="option"]:hover, li[aria-selected="true"] { background:rgba(255,255,255,.08) !important; color:var(--ink) !important; }
    .note { font-size:.8rem; color:var(--ink3); line-height:1.5; }

    /* Verdict */
    .verdict { --c:var(--c-unver); display:flex; align-items:center; gap:2rem; flex-wrap:wrap; padding:1.8rem 2rem; margin:2rem 0 1rem;
        border-radius:24px; border:1px solid color-mix(in srgb, var(--c) 35%, transparent);
        background:radial-gradient(500px 200px at 0% 0%, color-mix(in srgb, var(--c) 16%, transparent), transparent 70%), var(--glass); }
    .verdict.small { padding:1rem 1.3rem; margin:.4rem 0 .8rem; gap:1.2rem; border-radius:18px; }
    .t-true{--c:var(--c-true)} .t-partial{--c:var(--c-partial)} .t-mislead{--c:var(--c-mislead)} .t-false{--c:var(--c-false)} .t-unver{--c:var(--c-unver)}
    .ring { position:relative; width:128px; height:128px; flex:none; }
    .small .ring { width:84px; height:84px; }
    .ring svg { width:100%; height:100%; transform:rotate(-90deg); }
    .ring .trk { stroke:rgba(255,255,255,.08); fill:none; stroke-width:9; }
    .ring .bar { stroke:var(--c); fill:none; stroke-width:9; stroke-linecap:round; stroke-dasharray:326.7; stroke-dashoffset:var(--off);
        animation:fill 1.1s cubic-bezier(.2,.8,.2,1) both; filter:drop-shadow(0 0 6px color-mix(in srgb, var(--c) 60%, transparent)); }
    @keyframes fill { from { stroke-dashoffset:326.7; } }
    .ring .num { position:absolute; inset:0; display:grid; place-items:center; text-align:center; font-family:var(--display); font-weight:700; font-size:1.7rem; }
    .small .ring .num { font-size:1.15rem; }
    .ring .num small { display:block; font-family:var(--body); font-weight:500; font-size:.68rem; color:var(--ink3); margin-top:-.2rem; }
    .v-kick { color:var(--ink3); font-size:.9rem; margin-bottom:.2rem; }
    .v-word { font-family:var(--display); font-weight:800; font-size:clamp(2rem,4.5vw,3rem); letter-spacing:-.025em; color:var(--c); line-height:1.05; }
    .small .v-word { font-size:1.6rem; }
    .v-claim { color:var(--ink2); font-size:.98rem; margin-top:.6rem; max-width:52ch; line-height:1.5; }

    /* Quick stats */
    .qs { display:flex; gap:.6rem; flex-wrap:wrap; margin-bottom:1.2rem; }
    .qs div { background:var(--glass); border:1px solid var(--line); border-radius:12px; padding:.6rem 1rem; font-size:.86rem; color:var(--ink2); display:flex; flex-direction:column; gap:.18rem; min-width:90px; }
    .qs b { color:var(--ink); font-family:var(--display); font-weight:700; font-size:1.25rem; line-height:1.1; display:block; }
    .qs small { font-size:.74rem; color:var(--ink3); }

    /* Tabs */
    [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line) !important; }
    [data-baseweb="tab"] { color:var(--ink3) !important; font-weight:600 !important; background:transparent !important; padding:.7rem 1rem !important; }
    [data-baseweb="tab"][aria-selected="true"] { color:var(--ink) !important; }
    [data-baseweb="tab-highlight"] { background:var(--grad) !important; height:3px !important; border-radius:3px; }
    [data-baseweb="tab-border"] { display:none; }

    .explain { background:var(--glass); border:1px solid var(--line); border-radius:var(--r); padding:1.3rem 1.5rem; font-size:1.05rem; line-height:1.78; color:var(--ink); max-width:68ch; margin-top:1rem; }
    .notice { background:rgba(165,180,252,.08); border:1px solid rgba(165,180,252,.3); color:#c7d0ff; border-radius:var(--r); padding:.9rem 1.2rem; font-size:.93rem; line-height:1.55; margin:1rem 0 0; }
    .pid { font-family:ui-monospace,'JetBrains Mono',Menlo,monospace; font-size:.72rem; color:var(--ink3); background:rgba(255,255,255,.04); border:1px solid var(--line); border-radius:5px; padding:.08rem .4rem; }

    /* Sources */
    .src { background:var(--glass); border:1px solid var(--line); border-radius:var(--r); padding:1.1rem 1.25rem; margin-top:.75rem; transition:all .15s; }
    .src:hover { border-color:var(--line-hi); background:var(--glass-hi); }
    .src.cited { border-left:3px solid var(--a2); }
    .src-top { display:flex; justify-content:space-between; gap:.6rem; flex-wrap:wrap; align-items:center; margin-bottom:.5rem; }
    .src-dom { display:flex; align-items:center; gap:.55rem; font-weight:600; }
    .src-dom i { width:24px; height:24px; border-radius:7px; background:var(--glass-hi); display:grid; place-items:center; font-size:.72rem; font-style:normal; color:var(--a2); font-family:var(--display); }
    .tag { font-size:.74rem; font-weight:600; padding:.15rem .6rem; border-radius:999px; background:rgba(79,209,197,.14); color:var(--a2); }
    .tag.dim { background:var(--glass-hi); color:var(--ink3); }
    .src-txt { color:var(--ink2); font-size:.96rem; line-height:1.6; max-width:80ch; }
    .src-bot { display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-top:.8rem; font-size:.82rem; color:var(--ink3); }
    .meter { flex:0 0 100px; height:4px; border-radius:4px; background:rgba(255,255,255,.07); overflow:hidden; }
    .meter span { display:block; height:100%; background:var(--grad); border-radius:4px; }
    .src-bot a { color:var(--a2); font-weight:600; text-decoration:none; margin-left:auto; padding:.2rem .6rem; border:1px solid rgba(79,209,197,.25); border-radius:6px; font-size:.8rem; transition:all .15s; }
    .src-bot a:hover { background:rgba(79,209,197,.1); text-decoration:none; }

    /* Pipeline */
    .stage { display:grid; grid-template-columns:32px 170px 1fr 72px; gap:.8rem; align-items:center; padding:.7rem 0; border-bottom:1px solid var(--line); font-size:.92rem; }
    .stage i { width:26px; height:26px; border-radius:50%; background:var(--grad); color:#0a0c16; display:grid; place-items:center; font-style:normal; font-weight:700; font-size:.78rem; }
    .stage .track { height:8px; background:rgba(255,255,255,.07); border-radius:8px; overflow:hidden; }
    .stage .track span { display:block; height:100%; background:var(--grad); border-radius:8px; }
    .stage em { font-style:normal; color:var(--ink3); text-align:right; font-variant-numeric:tabular-nums; }
    .kv { display:flex; justify-content:space-between; padding:.5rem 0; border-bottom:1px solid var(--line); font-size:.9rem; }
    .kv span:first-child { color:var(--ink3); }

    .h { font-family:var(--display); font-weight:700; font-size:1.2rem; margin:2rem 0 .3rem; }
    .sep { border:none; border-top:1px solid var(--line); margin:3.5rem 0 1.2rem; }
    .disc { text-align:center; font-size:.79rem; color:var(--ink3); line-height:1.6; max-width:760px; margin-left:auto; margin-right:auto; }
    /* Sidebar brand */
    .sb-brand { display:flex; align-items:center; gap:.6rem; padding:.2rem 0 1rem; }
    .sb-mark { width:28px; height:28px; border-radius:8px; background:var(--grad); display:grid; place-items:center; font-size:.9rem; }
    .sb-name { font-family:var(--display); font-weight:700; font-size:1rem; color:var(--ink); letter-spacing:.01em; }
    [data-testid="stExpander"] { background:var(--glass) !important; border:1px solid var(--line) !important; border-radius:var(--r) !important; }
    [data-testid="stExpander"] summary p { color:var(--ink) !important; }
    [data-testid="stMetricValue"] { color:var(--ink) !important; }

    @media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
    @media (max-width:640px) {
        [data-testid="block-container"] { padding:1.2rem 1rem 3rem !important; }
        .stage { grid-template-columns:28px 1fr 60px; } .stage .track { grid-column:2 / 4; }
        .verdict { padding:1.3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached engine & query cache (unchanged)
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


def esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


VERDICT_MAP: dict[str, tuple[str, str]] = {
    "True":           ("t-true", "✓"),
    "Partially True": ("t-partial", "◐"),
    "Misleading":     ("t-mislead", "!"),
    "False":          ("t-false", "✕"),
    "Unverified":     ("t-unver", "?"),
}


def verdict_html(label: str, conf: float, claim: str = "", small: bool = False) -> str:
    cls, icon = VERDICT_MAP.get(label, ("t-unver", "?"))
    off = 326.7 * (1 - max(0.0, min(conf, 1.0)))
    claim_html = f'<div class="v-claim">“{esc(claim)}”</div>' if claim and not small else ""
    return f"""
    <div class="verdict {cls} {'small' if small else ''}">
        <div class="ring">
            <svg viewBox="0 0 120 120"><circle class="trk" cx="60" cy="60" r="52"/>
            <circle class="bar" cx="60" cy="60" r="52" style="--off:{off:.1f}"/></svg>
            <div class="num"><div>{conf:.0%}<small>confident</small></div></div>
        </div>
        <div>
            <div class="v-kick">{'Verdict' if small else 'Here’s what we found'}</div>
            <div class="v-word">{icon}&nbsp;{esc(label)}</div>
            {claim_html}
        </div>
    </div>"""


# ---------------------------------------------------------------------------
# Sidebar (settings, tucked away)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-mark">⚖️</div><div class="sb-name">VERITAS</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Evidence source")
    mode_selection = st.radio(
        "Retrieval Source",
        options=["Benchmark Corpus", "Live Web Search"],
        label_visibility="collapsed",
        help=(
            "Benchmark Corpus searches local pre-built indexes (fast & offline). "
            "Live Web Search queries DuckDuckGo for live internet evidence."
        ),
    )
    mode_key = "benchmark" if mode_selection == "Benchmark Corpus" else "live_search"
    st.markdown('<div class="note">Corpus is fast and offline. Live web checks today’s internet.</div>', unsafe_allow_html=True)

    st.markdown("### Checking model")
    model_backend = st.selectbox(
        "Verification Model",
        options=["mock", "api", "local"],
        label_visibility="collapsed",
        help=(
            "mock: offline deterministic verifier (zero API cost). "
            "api: OpenAI / Anthropic cloud model. "
            "local: Ollama local weights."
        ),
    )

    st.markdown("### Safety checks")
    use_abstention = st.toggle(
        "Reasoning must match evidence",
        value=True,
        help="Downgrades verdict to Unverified if reasoning is not entailed by evidence.",
    )
    min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.60, 0.05)
    min_faithfulness = st.slider("Minimum evidence match", 0.0, 1.0, 0.70, 0.05)
    st.markdown('<div class="note">Below these levels VERITAS answers “Unverified” rather than guess.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, str] = {
    "💉 Vaccines": "COVID-19 mRNA vaccines significantly reduce the risk of severe hospitalizations.",
    "📈 US jobs": "The US Bureau of Labor Statistics reported nonfarm payrolls rose by 275,000 in February 2024.",
    "🌕 Moon cheese": "NASA confirmed the Moon is entirely composed of aged dairy cheddar cheese.",
    "🏔️ Everest vs K2": "Mount Everest is significantly shorter than K2 in total elevation above sea level.",
    "📰 News article": (
        "In a joint briefing on Tuesday, the Federal Reserve indicated interest rates may remain elevated "
        "through the third quarter. Meanwhile, European Union regulators opened a formal investigation into "
        "artificial intelligence training practices under the new Digital Services Act. Labor statistics show "
        "unemployment fell to 3.8 percent in major tech hubs, though overall industrial manufacturing output "
        "contracted by 0.4 percent month-over-month according to national bank records."
    ),
}

for key, default in (("claim_text", ""), ("claim_date", ""), ("last_result", None)):
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Nav + hero
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="nav">
        <div class="logo"><i>⚖️</i>VERITAS</div>
        <div class="pills">
            <span class="pill">Source <b>{'Corpus' if mode_key == 'benchmark' else 'Live web'}</b></span>
            <span class="pill">Model <b>{esc(model_backend)}</b></span>
            <span class="pill">Settings in the ‹ sidebar</span>
        </div>
    </div>
    <div class="hero">
        <h1>Is it true? <em>Let’s find out.</em></h1>
        <p>Paste a claim, a headline or a whole article. VERITAS searches for evidence, checks its own reasoning and shows you exactly where the answer came from.</p>
    </div>
    <div class="try-label">Not sure what to try? Pick one:</div>
    """,
    unsafe_allow_html=True,
)

chip_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(chip_cols, EXAMPLES.items()):
    if col.button(label, key=f"chip_{label}", use_container_width=True):
        st.session_state.claim_text = text
        st.session_state.last_result = None
        st.rerun()

# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

curr = st.session_state.claim_text
wc = len(curr.split()) if curr else 0
hint = "<b>Article mode</b> · claims will be split and checked one by one" if wc > 30 else "Single claim"
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

with st.form("saas_verification_form", clear_on_submit=False):
    st.markdown(
        f'<div class="composer-top"><span>What should we check?</span><span>{wc} words · {hint}</span></div>',
        unsafe_allow_html=True,
    )
    claim_input = st.text_area(
        "Claim Input",
        value=st.session_state.claim_text,
        height=130,
        placeholder="e.g. The Federal Reserve raised interest rates by 25 basis points in July 2023…",
        label_visibility="collapsed",
    )
    c_date, c_gap, c_clear, c_go = st.columns([2.4, 0.3, 1, 1.8])
    with c_date:
        claim_date_input = st.text_input(
            "Claim date",
            value=st.session_state.claim_date,
            placeholder="📅 Claim date YYYY-MM-DD (optional)",
            label_visibility="collapsed",
            help="Helps VERITAS prefer evidence from around that date. Leave empty to skip.",
        )
    with c_clear:
        clear_clicked = st.form_submit_button("Clear", type="secondary", use_container_width=True)
    with c_go:
        submit_clicked = st.form_submit_button("Check claim ⚖", type="primary", use_container_width=True)

if clear_clicked:
    st.session_state.claim_text = ""
    st.session_state.claim_date = ""
    st.session_state.last_result = None
    st.rerun()

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if submit_clicked:
    st.session_state.claim_text = claim_input.strip()
    st.session_state.claim_date = claim_date_input.strip()

    if not claim_input.strip():
        st.warning("Type or paste something to check, or pick an example above.")
    else:
        engine = get_engine()
        if engine.is_long_text(claim_input):
            with st.spinner("Reading the text and finding claims worth checking…"):
                extracted_claims = engine.extract_claims(claim_input)

            if extracted_claims:
                st.markdown(
                    f'<div class="h">We found {len(extracted_claims)} claims to check</div>',
                    unsafe_allow_html=True,
                )
                for idx, ec in enumerate(extracted_claims, start=1):
                    with st.expander(f"{idx}. {ec.claim}", expanded=(idx == 1)):
                        with st.spinner(f"Checking claim {idx}…"):
                            res_dict = cached_verify_claim(
                                claim=ec.claim,
                                claim_date=claim_date_input.strip() or None,
                                mode=mode_key,
                            )
                        st.markdown(verdict_html(res_dict["label"], res_dict["confidence"], small=True), unsafe_allow_html=True)
                        st.markdown(f'<div class="explain" style="margin-top:0">{esc(res_dict["explanation"])}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="note" style="margin-top:.6rem">Searched for: {esc(", ".join(ec.queries))}</div>', unsafe_allow_html=True)
            else:
                st.warning("No separate, checkable claims were found. Try a shorter or more specific passage.")
        else:
            with st.spinner(f"Searching evidence via {mode_selection}…"):
                res_dict = cached_verify_claim(
                    claim=claim_input.strip(),
                    claim_date=claim_date_input.strip() or None,
                    mode=mode_key,
                )
            st.session_state.last_result = res_dict

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.last_result:
    res_dict = st.session_state.last_result
    lbl         = res_dict["label"]
    conf        = float(res_dict["confidence"])
    exp         = res_dict["explanation"]
    cards       = res_dict.get("evidence_cards", [])
    origin_mode = res_dict.get("source_mode", mode_selection)
    faith_score = res_dict.get("faithfulness_score")
    timings     = res_dict.get("stage_timings", {})
    total_time  = sum(timings.values()) if timings else 0.0
    cited_count = sum(1 for c in cards if c.get("is_cited"))

    st.markdown(verdict_html(lbl, conf, st.session_state.claim_text), unsafe_allow_html=True)

    faith_txt = f"{faith_score:.0%}" if faith_score is not None else "n/a"
    conf_ok  = conf >= min_confidence
    faith_ok = faith_score is None or faith_score >= min_faithfulness
    st.markdown(
        f"""
        <div class="qs">
            <div><b>{conf:.0%}</b><small>confidence{'  ✓' if conf_ok else '  ✗ low'}</small></div>
            <div><b>{faith_txt}</b><small>evidence match{'  ✓' if faith_ok else '  ✗ low'}</small></div>
            <div><b>{len(cards)}</b><small>sources found</small></div>
            <div><b>{cited_count}</b><small>used in verdict</small></div>
            <div><b>{total_time:.1f}s</b><small>to check</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_why, tab_src, tab_pipe = st.tabs(["Why this verdict", f"Sources ({len(cards)})", "How it was checked"])

    with tab_why:
        if lbl == "Unverified":
            reasons = res_dict.get("abstention_reasons", [])
            reason_text = "; ".join(reasons) if reasons else (
                "The evidence was insufficient or conflicting, or the reasoning didn’t match it closely enough."
            )
            st.markdown(f'<div class="notice"><strong>Why “Unverified”?</strong> {esc(reason_text)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="explain">{esc(exp)}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "⤓ Download full report (JSON)",
            data=json.dumps(res_dict, indent=2, default=str),
            file_name=f"veritas_report_{int(time.time())}.json",
            mime="application/json",
            help="Save the verdict, scores and citations.",
        )

    with tab_src:
        only_cited = st.toggle("Only show sources used in the verdict", value=False, disabled=cited_count == 0)
        shown = [(i, c) for i, c in enumerate(cards, start=1) if c.get("is_cited") or not only_cited]
        if not shown:
            st.markdown('<div class="note" style="padding:1rem 0">No sources to show. Try rewording the claim or switching the evidence source in the sidebar.</div>', unsafe_allow_html=True)
        max_score = max([float(c.get("score") or 0.0) for c in cards] + [1e-9])
        for i, c in shown:
            domain  = c.get("domain") or "Unknown source"
            score   = float(c.get("score") or 0.0)
            pct     = int(100 * score / max_score)
            url     = c.get("url")
            date    = esc(c["date"]) if c.get("date") else "No date"
            tag     = '<span class="tag">Used in verdict</span>' if c.get("is_cited") else ""
            kind    = "Live web" if "Live" in origin_mode else "Corpus"
            link    = f'<a href="{esc(url)}" target="_blank" rel="noopener">Open source ↗</a>' if url else ""
            st.markdown(
                f"""
                <div class="src {'cited' if c.get('is_cited') else ''}">
                    <div class="src-top">
                        <div class="src-dom"><i>{i}</i>{esc(domain)} {tag}</div>
                        <span class="tag dim">{kind}</span>
                    </div>
                    <div class="src-txt">“{esc((c.get('snippet') or '').strip())}”</div>
                    <div class="src-bot">
                        <span>{date}</span>
                        <span>Relevance</span><div class="meter"><span style="width:{pct}%"></span></div>
                        {link}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_pipe:
        if timings:
            peak = max(timings.values()) or 1e-9
            rows = "".join(
                f'<div class="stage"><i>{n}</i><span>{esc(stage.replace("_", " ").title())}</span>'
                f'<div class="track"><span style="width:{max(4, int(100 * sec / peak))}%"></span></div>'
                f'<em>{sec:.3f}s</em></div>'
                for n, (stage, sec) in enumerate(timings.items(), start=1)
            )
            st.markdown(rows, unsafe_allow_html=True)
        details = [
            ("Evidence source", origin_mode),
            ("Model", model_backend),
            ("Passages retrieved", len(cards)),
            ("Passages cited", cited_count),
            ("Response parsed", "Yes" if not res_dict.get("parse_failed") else "No"),
            ("Confidence vs your minimum", f"{conf:.0%} vs {min_confidence:.0%}"),
        ]
        if faith_score is not None:
            details.append(("Evidence match vs your minimum", f"{faith_score:.0%} vs {min_faithfulness:.0%}"))
        if res_dict.get("abstained") and res_dict.get("original_label"):
            details.append(("Verdict before safety check", res_dict["original_label"]))
        st.markdown(
            '<div style="margin-top:1.2rem">' + "".join(f'<div class="kv"><span>{esc(k)}</span><span>{esc(v)}</span></div>' for k, v in details) + "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(f'<hr class="sep"><div class="disc">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)