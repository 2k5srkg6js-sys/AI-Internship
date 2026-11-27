"""Session 3 UI: chat with the ADK knowledge-base agent and see its Think -> Act -> Observe loop.

Run from the repo root:
    streamlit run ai-engineering-bootcamp-v2/week-3/agent/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# week-3/ first, so `agent` resolves to the package and not to agent/agent.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from agent.agent import MODEL  # noqa: E402
from agent.runtime import MAX_LLM_CALLS, MAX_TOOL_CALLS, AgentResult, AgentSession  # noqa: E402

EXAMPLES = [
    "Until when can I return a holiday purchase, and who must approve a $600 refund?",
    "What are the rules for duplicate charges and price-match requests?",
    "What was Northwind's total revenue in 2025?",
]

STATUS_LABEL = {
    "ok": ("success", "Answered with verified citations"),
    "refused": ("info", "Refused: not covered by the indexed documents"),
    "limit_llm": ("warning", f"Fail-closed: LLM-call limit ({MAX_LLM_CALLS}) reached"),
    "limit_tools": ("warning", f"Fail-closed: tool-call limit ({MAX_TOOL_CALLS}) reached"),
    "uncited": ("warning", "Fail-closed: answer had no citations"),
    "invalid_citation": ("warning", "Fail-closed: answer cited chunks that were never retrieved"),
    "no_search": ("warning", "Fail-closed: agent answered without searching, even after retry"),
    "timeout": ("warning", "Fail-closed: turn timed out"),
    "error": ("error", "Fail-closed: internal error"),
}


def render_trace(trace: list[dict]) -> None:
    """Show the run as numbered Think -> Act -> Observe cycles."""

    if not trace:
        st.caption("No steps recorded.")
        return
    cycle, prev = 0, None
    for step in trace:
        kind = step["kind"]
        # A new cycle starts with the first Think/Act, or after an Observe.
        if kind in ("think", "act") and (prev is None or prev == "observe"):
            cycle += 1
            st.markdown(f"**Cycle {cycle}**")
        if kind == "think":
            st.info(f"🧠 **Think** — {step['text']}")
        elif kind == "act":
            args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            st.warning(f"🔧 **Act** — `{step['tool']}({args})`")
        elif kind == "observe":
            if step.get("status") != "ok":
                st.error(f"👀 **Observe** — tool error: {step.get('error')}")
            else:
                lines = [f"- `{h['chunk_id']}` · score {h['score']} · {h['preview']}…" for h in step["hits"]]
                st.success(f"👀 **Observe** — {len(step['hits'])} chunks\n\n" + "\n".join(lines))
        elif kind == "answer":
            st.markdown("✅ **Answer drafted** (checked by citation guard)")
        elif kind == "guard":
            st.error(f"⛔ **Guard** — {step['text']}")
        prev = kind


def render_result(r: AgentResult) -> None:
    level, label = STATUS_LABEL.get(r.status, ("error", r.status))
    st.markdown(r.answer)
    getattr(st, level)(label, icon=None)
    retry = " · search gate forced a retry" if r.retried else ""
    st.caption(f"tool calls: {r.tool_calls} · model responses: {r.llm_calls} · "
               f"cited: {len(r.cited_ids)}{retry}")
    with st.expander("Agent trace: Think → Act → Observe", expanded=False):
        render_trace(r.trace)


st.set_page_config(page_title="KB Research Agent", page_icon="🔎", layout="centered")
st.title("Knowledge-Base Research Agent")
st.caption("Google ADK agent over the Session 2 Pinecone index: multi-part and follow-up questions, "
           "re-queries when evidence is thin, cited answer or clear refusal.")

if "agent_session" not in st.session_state:
    st.session_state.agent_session = AgentSession()
    st.session_state.messages = []  # (role, payload)

with st.sidebar:
    st.header("Status")
    for var in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
        (st.success if (os.getenv(var) or "").strip() else st.error)(var)
    st.markdown("---")
    st.markdown(f"**Model:** `{MODEL}`")
    st.markdown(f"**Hard limits per turn:** {MAX_LLM_CALLS} LLM calls · {MAX_TOOL_CALLS} tool calls")
    if st.button("New conversation", width="stretch"):
        st.session_state.agent_session = AgentSession()
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Examples**")
    example = None
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            example = ex

for role, payload in st.session_state.messages:
    with st.chat_message(role):
        render_result(payload) if role == "assistant" else st.markdown(payload)

prompt = st.chat_input("Ask about your indexed documents…") or example
if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching and checking evidence…"):
            result = st.session_state.agent_session.ask(prompt)
        st.session_state.messages.append(("assistant", result))
        render_result(result)
