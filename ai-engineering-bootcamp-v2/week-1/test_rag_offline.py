"""Offline tests for /ingest, /debug/retrieve and grounded /ask.

OpenAI and Pinecone are stubbed: no network, no tokens, no keys needed.
Run:  python test_rag_offline.py   (or: pytest test_rag_offline.py)
"""

import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-not-a-real-key")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

TOKEN = "test-token-123"
CHUNKS = [
    {"id": "doc-a::chunk_0", "score": 0.8, "document_id": "doc-a", "chunk_index": 0,
     "source": "a.txt", "text": "Electronics return window: 14 days."},
    {"id": "doc-b::chunk_0", "score": 0.4, "document_id": "doc-b", "chunk_index": 0,
     "source": "b.txt", "text": "Remote work: up to 3 days per week."},
]


def _stub_rag():
    main.rag.ingest_document = lambda client, doc_id, text, md: {
        "chunks_indexed": 2, "chunks_replaced": 0, "embedding_tokens": 10}
    main.rag.retrieve = lambda client, q, top_k: (CHUNKS, 7)


def _stub_llm(grounded: "main.GroundedAnswer"):
    main.call_model_structured = lambda messages, model: (grounded, 100, 80, 20)


def _env(**values):
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


client = TestClient(main.app)
INGEST_BODY = {"text": "hello world", "document_id": "doc-x"}


def test_ingest_disabled_without_server_token():
    _stub_rag(); _env(INGEST_DEMO_TOKEN=None)
    r = client.post("/ingest", json=INGEST_BODY, headers={"X-Ingest-Token": TOKEN})
    assert r.status_code == 503, r.text


def test_ingest_rejects_missing_or_wrong_token():
    _stub_rag(); _env(INGEST_DEMO_TOKEN=TOKEN)
    assert client.post("/ingest", json=INGEST_BODY).status_code == 401
    r = client.post("/ingest", json=INGEST_BODY, headers={"X-Ingest-Token": "wrong"})
    assert r.status_code == 401


def test_ingest_ok_with_token():
    _stub_rag(); _env(INGEST_DEMO_TOKEN=TOKEN)
    r = client.post("/ingest", json=INGEST_BODY, headers={"X-Ingest-Token": TOKEN})
    assert r.status_code == 200, r.text
    assert r.json() == {"document_id": "doc-x", "chunks_indexed": 2, "status": "indexed"}


def test_ingest_blank_text_is_400():
    _stub_rag(); _env(INGEST_DEMO_TOKEN=TOKEN)
    r = client.post("/ingest", json={"text": "  ", "document_id": "d"},
                    headers={"X-Ingest-Token": TOKEN})
    assert r.status_code == 400


def test_debug_retrieve_local_only():
    _stub_rag(); _env(RENDER=None)
    assert client.get("/debug/retrieve", params={"q": "x"}).status_code == 200
    _env(RENDER="true")
    assert client.get("/debug/retrieve", params={"q": "x"}).status_code == 404
    _env(RENDER=None)
    remote = TestClient(main.app, client=("203.0.113.5", 50000))
    assert remote.get("/debug/retrieve", params={"q": "x"}).status_code == 404


def test_ask_cites_only_used_documents():
    _stub_rag()
    _stub_llm(main.GroundedAnswer(answer="14 days.", confidence=0.9, sources_needed=False,
                                  supported=True, cited_chunk_ids=["doc-a::chunk_0"]))
    data = client.post("/ask", json={"question": "electronics?"}).json()
    assert set(data) == {"answer", "tokens_used", "model", "latency_ms", "cost_usd",
                         "citations", "retrieved_chunk_ids"}
    assert data["citations"] == ["doc-a"]
    assert data["retrieved_chunk_ids"] == ["doc-a::chunk_0", "doc-b::chunk_0"]
    assert data["tokens_used"] == 107  # chat 100 + query embedding 7


def test_ask_refuses_invented_citation():
    _stub_rag()
    _stub_llm(main.GroundedAnswer(answer="Made up.", confidence=0.9, sources_needed=False,
                                  supported=True, cited_chunk_ids=["doc-zzz::chunk_9"]))
    data = client.post("/ask", json={"question": "parental leave?"}).json()
    assert data["answer"]["answer"] == main.REFUSAL
    assert data["citations"] == []


def test_ask_refuses_when_unsupported():
    _stub_rag()
    _stub_llm(main.GroundedAnswer(answer="x", confidence=0.1, sources_needed=True,
                                  supported=False, cited_chunk_ids=["doc-a::chunk_0"]))
    data = client.post("/ask", json={"question": "parental leave?"}).json()
    assert data["answer"]["answer"] == main.REFUSAL
    assert data["citations"] == []


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {name}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
