# Week 3 – Knowledge-Base Research Agent (Google ADK)

A single Google ADK agent with a Streamlit UI. When a user asks a multi-part or
follow-up question about the indexed documents, the agent searches the
Pinecone-backed knowledge base from week 1/2, evaluates whether the retrieved
evidence is sufficient (re-querying with refined terms if needed), and returns
a cited answer or a clear refusal.

## Dependency on week-1

week-3 does **not** duplicate the RAG code. `agent/tools.py` imports
`rag.py` and `vector_store.py` from the sibling folder
`ai-engineering-bootcamp-v2/week-1/`. Both folders must be present side by side.

## Setup

From the repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -r ai-engineering-bootcamp-v2/week-1/requirements.txt -r ai-engineering-bootcamp-v2/week-3/agent/requirements.txt
```

Python 3.12+ is required by google-adk.

### Environment variables

The real `.env` belongs in **`ai-engineering-bootcamp-v2/week-1/.env`** (git-ignored,
never committed). Do not create a `.env` in week-3. A template is in
[`agent/.env.example`](agent/.env.example).

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Gemini model for the ADK agent |
| `OPENAI_API_KEY` | yes | Query embeddings (`text-embedding-3-small`) |
| `PINECONE_API_KEY` | yes | Pinecone access |
| `PINECONE_INDEX_NAME` | yes | Existing dense index, dimension 1536, metric cosine |
| `PINECONE_NAMESPACE` | no | Defaults to Pinecone's default namespace |
| `AGENT_MODEL` | no | Defaults to `gemini-2.5-flash` |

Documents are ingested with the week-1/2 service (`POST /ingest`); the agent only reads.

## Run

Streamlit UI, from the repo root:

```bash
streamlit run ai-engineering-bootcamp-v2/week-3/agent/streamlit_app.py
```

Smoke test (live calls to Gemini, OpenAI and Pinecone), from the repo root:

```bash
python ai-engineering-bootcamp-v2/week-3/agent/smoke_test.py
```

The content checks in the smoke test assume the Northwind support policy manual
is the indexed document.

## How it works

| File | Role |
|---|---|
| `agent/agent.py` | `root_agent`: Gemini with one tool and thought summaries enabled |
| `agent/tools.py` | `search_knowledge_base(query, top_k)`: wraps `week-1/rag.retrieve()`, flags weak evidence |
| `agent/runtime.py` | Runs the agent with hard limits, the search gate and the citation guard |
| `agent/streamlit_app.py` | Chat UI; the trace shows each step as Think → Act → Observe |
| `agent/smoke_test.py` | Offline gate tests plus live end-to-end checks |

Safeguards, all enforced in code and failing closed (fixed refusal, never a partial answer):

- **Hard step limit:** `RunConfig(max_llm_calls=8)` per user turn, plus a tool-call counter (max 10) that aborts the run before an extra tool call executes.
- **Search gate:** a substantive answer given without a search in the current turn, or citing chunks not retrieved in that turn, is discarded and retried once with a reminder. The retry shares the same budget. If it fails again, the answer is withheld.
- **Citation guard:** an answer must be a refusal or cite at least one `[chunk_id]` retrieved in the current turn.

Known limitation: the guard checks where a citation comes from, not whether the
cited chunk actually supports the sentence.
