"""
Streamlit UI for the KYC Triage Multi-Agent System.

This file intentionally contains almost NO business logic — its only
jobs are: collect input, kick off the orchestrator (in a background
thread, so we can show live progress), render the live log as it
streams in, and render the final dashboard. Every actual decision is
made in src/orchestrator.py and src/agents.py, which can be (and are,
see tests/) unit-tested completely independently of Streamlit.

Run locally with:  streamlit run app/app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import queue
import sys
import threading
import uuid
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Allow `import src.xxx` when Streamlit runs this file directly (Streamlit
# sets the working directory to the file's folder, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guardrails_and_logging import subscribe_to_run, unsubscribe_from_run  # noqa: E402
from src.llm_factory import active_provider  # noqa: E402
from src.mock_data_generator import generate_applicant_case  # noqa: E402
from src.orchestrator import TriageResult, run_triage  # noqa: E402

st.set_page_config(
    page_title="KYC Triage Multi-Agent System",
    page_icon="🕵️",
    layout="wide",
)

DECISION_STYLE = {
    "approve": ("✅ APPROVED", "success"),
    "reject": ("⛔ REJECTED", "error"),
    "manual_review": ("🟡 MANUAL REVIEW REQUIRED", "warning"),
}

LEVEL_COLOR = {"info": "#58a6ff", "warning": "#d29922", "error": "#f85149"}


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------


def _format_log_line(event: dict) -> str:
    level = event.get("level", "info")
    color = LEVEL_COLOR.get(level, "#c9d1d9")
    timestamp = str(event.get("timestamp", ""))[11:19] or "--:--:--"
    event_name = html.escape(str(event.get("event", "")))
    extras = {k: v for k, v in event.items() if k not in ("timestamp", "level", "event")}
    extras_str = " ".join(f"{html.escape(k)}={html.escape(_short(v))}" for k, v in extras.items())
    return (
        f'<span style="color:{color}">[{timestamp}] {level.upper():<7}</span> '
        f'<b style="color:#e6edf3">{event_name}</b> '
        f'<span style="color:#8b949e">{extras_str}</span>'
    )


def _short(value, limit: int = 160) -> str:
    text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_terminal(lines: list[str]) -> str:
    body = "<br>".join(lines[-400:]) or '<span style="color:#8b949e">Waiting for agents to start...</span>'
    return (
        '<div style="background:#0d1117;color:#c9d1d9;font-family:Consolas,\'Courier New\',monospace;'
        "padding:14px;border-radius:8px;height:340px;overflow-y:auto;"
        f'font-size:13px;line-height:1.6;border:1px solid #30363d;">{body}</div>'
    )


def render_risk_gauge(risk_score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f6feb"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(46,160,67,0.25)"},
                    {"range": [30, 70], "color": "rgba(210,153,34,0.25)"},
                    {"range": [70, 100], "color": "rgba(248,81,73,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_dashboard(result: TriageResult) -> None:
    label, style = DECISION_STYLE[result.verdict.decision]
    getattr(st, style)(f"**Final Decision: {label}**  \n{result.verdict.reason}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{result.fraud_result.risk_score} / 100")
    col2.metric("Document Quality", "Pass" if result.document_quality_passed else "Fail")
    col3.metric("Watchlist Hit", "Yes" if result.watchlist_check["is_on_watchlist"] else "No")
    col4.metric("Prior Flags on File", len(result.watchlist_check["prior_flags"]))

    gauge_col, flags_col = st.columns([1, 1])
    with gauge_col:
        st.plotly_chart(render_risk_gauge(result.fraud_result.risk_score), use_container_width=True)
    with flags_col:
        st.markdown("**Flagged Anomalies**")
        all_flags = result.fraud_result.risk_flags + (
            ["low_document_quality"] if not result.document_quality_passed else []
        )
        if all_flags:
            for flag in all_flags:
                st.markdown(f"- 🚩 `{flag}`")
        else:
            st.markdown("_No anomalies flagged._")
        st.markdown("**Rule matched:** `" + result.verdict.matched_rule + "`")

    st.markdown("#### Compliance Explanation")
    st.info(result.explanation)

    with st.expander("📄 Document Analysis Agent — full output"):
        st.json(result.document_result.model_dump())
    with st.expander("🔎 MCP tool results (identity comparison + watchlist check)"):
        st.json({"identity_comparison": result.identity_comparison, "watchlist_check": result.watchlist_check})
    with st.expander("🕵️ Fraud & Anomaly Detection Agent — full output"):
        st.json(result.fraud_result.model_dump())

    st.download_button(
        "⬇️ Download full case file (JSON)",
        data=json.dumps(result.to_case_file_dict(), indent=2),
        file_name=f"kyc_case_{result.case_id}.json",
        mime="application/json",
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.code(f"LLM provider: {active_provider()}", language=None)
    st.caption(
        "Set GEMINI_API_KEY or GROQ_API_KEY as an environment variable / "
        "HF Space secret to use a real model. Without one, the app runs "
        "fully offline using heuristic mock responses so every screen "
        "still works — clearly labeled '[offline mock]' wherever it appears."
    )
    st.markdown("---")
    st.markdown("### ℹ️ About this demo")
    st.markdown(
        "- **All data is synthetic**, generated with `Faker`. No real "
        "customers, documents, or transactions are involved.\n"
        "- **Document AI is simulated.** Uploaded files are never OCR'd — "
        "their bytes only seed a deterministic synthetic identity.\n"
        "- **Do not upload real personal documents.** None are needed."
    )
    st.markdown("---")
    st.markdown("### 🧩 Architecture")
    st.markdown(
        "`Document Analysis` → `MCP tool calls` → `Fraud Detection` → "
        "`Compliance Rule Engine` → `Compliance Explanation`\n\n"
        "Built with Google ADK, the Model Context Protocol, and Streamlit."
    )

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------

st.title("🕵️ KYC Triage — Multi-Agent System")
st.caption("Google Agent Development Kit • Model Context Protocol • Simulated Document AI • Financial Services")

st.warning(
    "🔒 **Privacy notice — this is a public demo.** Do not upload real IDs, "
    "utility bills, or any real personal documents. Uploaded file *content* "
    "is never read or stored; only a hash of the bytes is used to "
    "deterministically seed a synthetic (fake) identity via `Faker`.",
    icon="🔒",
)

input_col, scenario_col = st.columns(2)
with input_col:
    uploaded_file = st.file_uploader(
        "Upload a mock ID or utility bill",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Any file works — its content is not read. Its byte-hash seeds a synthetic applicant.",
    )
with scenario_col:
    scenario = st.selectbox(
        "Scenario to simulate",
        ["random", "clean", "fraud_mismatch", "watchlist_hit", "incomplete"],
        help=(
            "clean = should approve · fraud_mismatch / incomplete = should manual-review · "
            "watchlist_hit = should reject"
        ),
    )

run_clicked = st.button("▶️ Run KYC Triage", type="primary")

if run_clicked:
    seed = hashlib.sha256(uploaded_file.getvalue()).hexdigest() if uploaded_file else None
    case = generate_applicant_case(seed=seed, scenario=scenario)
    st.session_state["last_case"] = case

    with st.expander("🔍 Simulated input data fed to the pipeline", expanded=False):
        st.json({"document": case.document.to_dict(), "history": case.history.to_dict() if case.history else None})

    st.markdown("#### 🖥️ Live Agent Orchestration Log")
    terminal_placeholder = st.empty()
    terminal_placeholder.markdown(render_terminal([]), unsafe_allow_html=True)

    run_id = str(uuid.uuid4())
    log_queue: queue.Queue = queue.Queue()
    result_box: dict = {}

    def _on_log_event(event: dict) -> None:
        log_queue.put(("log", event))

    def _worker() -> None:
        try:
            result_box["result"] = run_triage(case, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            result_box["error"] = str(exc)
        finally:
            log_queue.put(("done", None))

    subscribe_to_run(run_id, _on_log_event)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    lines: list[str] = []
    while True:
        kind, payload = log_queue.get()
        if kind == "done":
            break
        lines.append(_format_log_line(payload))
        terminal_placeholder.markdown(render_terminal(lines), unsafe_allow_html=True)
    thread.join()
    unsubscribe_from_run(run_id)

    if "error" in result_box:
        st.error(f"Triage run failed: {result_box['error']}")
    else:
        st.session_state["last_result"] = result_box["result"]
        st.success("Triage run complete.")

if "last_result" in st.session_state:
    st.markdown("---")
    st.markdown("### 📊 Risk Score & Decision Dashboard")
    render_dashboard(st.session_state["last_result"])
