"""Minimal Streamlit UI for the week-1 RAG API: ingest a document, ask a question.

Run from anywhere:
    streamlit run streamlit_app.py

Configuration (week-1/.env, loaded by explicit path):
    API_BASE_URL       e.g. the Render URL; editable in the sidebar
    INGEST_DEMO_TOKEN  optional; read server-side only, sent as X-Ingest-Token to /ingest.
                       Never shown in the UI or logged.
"""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").strip()
# Render's free tier can take ~1 minute to wake up, so allow long requests.
TIMEOUT = httpx.Timeout(120.0, connect=30.0)


def ingest_token() -> str | None:
    """Read the token at call time; the value never leaves this function except as a header."""

    token = (os.getenv("INGEST_DEMO_TOKEN") or "").strip()
    return token or None


def error_message(resp: httpx.Response) -> str:
    """Server error detail only: no request headers, no token."""

    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return f"HTTP {resp.status_code}: {detail}"


st.set_page_config(page_title="Week 1 RAG Demo", page_icon="📚", layout="centered")
st.title("Week 1 RAG Demo")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API base URL", value=DEFAULT_API_BASE_URL).strip().rstrip("/")
    st.caption("Default comes from API_BASE_URL in week-1/.env.")
    if ingest_token():
        st.success("Ingest token configured.")
    else:
        st.warning("INGEST_DEMO_TOKEN not set: ingest is unavailable, Ask still works.")

# ----------------------------------------------------------------------------- Ingest
st.header("Ingest")
token_available = ingest_token() is not None
if not token_available:
    st.info(
        "Ingest is unavailable because INGEST_DEMO_TOKEN is not set in week-1/.env. "
        "Add it there and restart Streamlit. You can still use Ask below."
    )

with st.form("ingest_form"):
    document_id = st.text_input("Document ID", placeholder="e.g. northwind-faq")
    source = st.text_input("Source (optional)", placeholder="e.g. faq.txt")
    text = st.text_area("Document text", height=200)
    ingest_clicked = st.form_submit_button("Ingest", disabled=not token_available)

if ingest_clicked:
    if not document_id.strip() or not text.strip():
        st.error("Document ID and text are both required.")
    else:
        payload = {"text": text, "document_id": document_id.strip()}
        if source.strip():
            payload["metadata"] = {"source": source.strip()}
        try:
            with st.spinner("Chunking, embedding and indexing..."):
                resp = httpx.post(
                    f"{api_base}/ingest",
                    json=payload,
                    headers={"X-Ingest-Token": ingest_token() or ""},
                    timeout=TIMEOUT,
                )
        except httpx.HTTPError as exc:
            st.error(f"Could not reach {api_base}: {type(exc).__name__}")
        else:
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Status: {data.get('status')}")
                c1, c2 = st.columns(2)
                c1.metric("Document ID", data.get("document_id", ""))
                c2.metric("Chunks indexed", data.get("chunks_indexed", 0))
            elif resp.status_code == 401:
                st.error("The API rejected the ingest token. Check INGEST_DEMO_TOKEN matches Render.")
            else:
                st.error(error_message(resp))

st.divider()

# ----------------------------------------------------------------------------- Ask
st.header("Ask")
with st.form("ask_form"):
    question = st.text_input("Question", placeholder="What is the return window for electronics?")
    ask_clicked = st.form_submit_button("Ask")

if ask_clicked:
    if not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving evidence and answering..."):
                resp = httpx.post(
                    f"{api_base}/ask", json={"question": question.strip()}, timeout=TIMEOUT
                )
        except httpx.HTTPError as exc:
            st.error(f"Could not reach {api_base}: {type(exc).__name__}")
        else:
            if resp.status_code != 200:
                st.error(error_message(resp))
            else:
                data = resp.json()
                answer = data.get("answer", {})
                st.subheader("Answer")
                st.markdown(answer.get("answer", ""))

                citations = data.get("citations", [])
                st.markdown(
                    "**Cited documents:** " + (", ".join(f"`{c}`" for c in citations) or "none")
                )
                retrieved = data.get("retrieved_chunk_ids", [])
                with st.expander(f"Retrieved chunks ({len(retrieved)}) - retrieved is not cited"):
                    for cid in retrieved:
                        st.markdown(f"- `{cid}`")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Confidence", f"{answer.get('confidence', 0):.2f}")
                c2.metric("Tokens", data.get("tokens_used", 0))
                c3.metric("Cost (USD)", f"${data.get('cost_usd', 0):.6f}")
                c4.metric("Latency", f"{data.get('latency_ms', 0)} ms")
                st.caption(f"Model: {data.get('model', '')}")

                with st.expander("Raw JSON response"):
                    st.json(data)
