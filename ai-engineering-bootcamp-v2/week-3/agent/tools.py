"""Session 3 agent tool: search the Pinecone knowledge base built in Session 2.

Wraps rag.retrieve() from the sibling folder ../week-1 (no HTTP, no LLM call):
one OpenAI embedding for the query plus one Pinecone query. The agent gets raw
chunks with scores and IDs so it can judge the evidence and cite it itself.

Dependency: week-3 reuses week-1/rag.py and week-1/vector_store.py instead of
copying them. Secrets come from week-1/.env (git-ignored, never committed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ai-engineering-bootcamp-v2/week-3/agent/tools.py -> ai-engineering-bootcamp-v2/week-1
_WEEK1_DIR = Path(__file__).resolve().parent.parent.parent / "week-1"
_ENV_PATH = _WEEK1_DIR / ".env"

for _required in ("rag.py", "vector_store.py"):
    if not (_WEEK1_DIR / _required).is_file():
        raise ImportError(f"week-3 needs {_WEEK1_DIR / _required} (shared RAG code from week-1).")

# The single real .env lives in week-1/ (GOOGLE_API_KEY, OPENAI_API_KEY, PINECONE_*).
# Load it explicitly; on a host without .env the process environment is used.
load_dotenv(_ENV_PATH)

# Appended (not prepended) so week-1/ can never shadow the `agent` package of week-3.
if str(_WEEK1_DIR) not in sys.path:
    sys.path.append(str(_WEEK1_DIR))

from openai import OpenAI  # noqa: E402

import rag  # noqa: E402  (week-1/rag.py)
from vector_store import VectorStoreError  # noqa: E402  (week-1/vector_store.py)

MAX_TOP_K = 10
# Cosine score below which retrieval is flagged as weak. Calibrated by hand on the
# Northwind index with text-embedding-3-small (clear hits ~0.6, loose ones ~0.3-0.4);
# re-check it when the indexed documents change.
WEAK_EVIDENCE_SCORE = 0.45

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def search_knowledge_base(query: str, top_k: int = 8) -> dict[str, Any]:
    """Search the user's indexed documents and return the most similar text chunks.

    Use one focused query per sub-question. If the results do not explicitly
    contain the needed fact, call again with refined terms (synonyms, section
    names, numbers).

    Args:
        query: A short, standalone search query (no pronouns like "it" or "that").
        top_k: Number of chunks to return (1-10, default 8).

    Returns:
        A dict with "status" ("ok" or "error"), the "query", "evidence" ("strong"
        or "weak"; weak means: refine the query before relying on the results),
        a "hint", and "results": a list
        of chunks, each with "chunk_id" (use it for citations as [chunk_id]),
        "source", "score" (cosine similarity, higher is more similar) and "text".
    """

    query = (query or "").strip()
    if not query:
        return {"status": "error", "error": "Empty query.", "results": []}
    top_k = max(1, min(int(top_k or 8), MAX_TOP_K))

    try:
        hits, _tokens = rag.retrieve(_openai(), query, top_k)
    except VectorStoreError as exc:
        return {"status": "error", "error": f"Knowledge base unavailable: {exc}", "results": []}
    except Exception as exc:  # network/auth errors from OpenAI or Pinecone
        return {"status": "error", "error": f"Search failed: {type(exc).__name__}", "results": []}

    top_score = max((h["score"] or 0.0 for h in hits), default=0.0)
    weak = top_score < WEAK_EVIDENCE_SCORE
    return {
        "status": "ok",
        "query": query,
        "evidence": "weak" if weak else "strong",
        "hint": (
            "Top score is low: these chunks may only be loosely related. Re-query with "
            "the document's own vocabulary before relying on them."
            if weak else "Verify that a chunk states the exact fact before citing it."
        ),
        "results": [
            {
                "chunk_id": h["id"],
                "source": h["source"],
                "score": h["score"],
                "text": h["text"],
            }
            for h in hits
        ],
    }
