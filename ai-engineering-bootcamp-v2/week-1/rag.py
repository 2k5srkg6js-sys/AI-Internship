"""RAG helpers: chunking, OpenAI embeddings, Pinecone upsert and query.

Pinecone connection and index validation live in vector_store.py; this module
builds on it. The OpenAI client is passed in so main.py's single client is reused.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from vector_store import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, get_index, load_config

# Configurable via env; defaults follow the Session 2 assignment.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = 100  # chunks per OpenAI embeddings request
UPSERT_BATCH_SIZE = 100  # vectors per Pinecone upsert request

# Keys we set ourselves; caller metadata cannot override them.
RESERVED_METADATA_KEYS = {"document_id", "chunk_index", "source", "text"}


def chunk_id(document_id: str, index: int) -> str:
    """Stable ID, so re-ingesting a document overwrites its chunks instead of duplicating."""

    return f"{document_id}::chunk_{index}"


def _namespace() -> str:
    return load_config().namespace or ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [c for c in splitter.split_text(text) if c.strip()]


def embed_texts(client: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed texts in batches. Returns (vectors, total embedding tokens)."""

    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIMENSIONS
        )
        # The API returns items with an index; sort to be safe.
        vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        tokens += resp.usage.total_tokens if resp.usage else 0
    for v in vectors:
        if len(v) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding has {len(v)} dims, expected {EMBEDDING_DIMENSIONS}")
    return vectors, tokens


def _delete_existing_chunks(document_id: str, namespace: str) -> int:
    """Remove a document's old chunks so a shorter re-ingest leaves no stale ones."""

    index = get_index()
    ids: list[str] = []
    for page in index.list(prefix=f"{document_id}::chunk_", namespace=namespace):
        ids.extend(v.id if hasattr(v, "id") else str(v) for v in page.vectors)
    for start in range(0, len(ids), 1000):
        index.delete(ids=ids[start : start + 1000], namespace=namespace)
    return len(ids)


def ingest_document(
    client: OpenAI,
    document_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chunk, embed and upsert one document. Returns counts for the API response."""

    metadata = metadata or {}
    extra = {k: v for k, v in metadata.items() if k not in RESERVED_METADATA_KEYS and v is not None}
    source = str(metadata.get("source") or document_id)

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Text produced no chunks")

    vectors, embedding_tokens = embed_texts(client, chunks)
    namespace = _namespace()
    replaced = _delete_existing_chunks(document_id, namespace)

    records = [
        {
            "id": chunk_id(document_id, i),
            "values": vec,
            "metadata": {
                **extra,
                "document_id": document_id,
                "chunk_index": i,
                "source": source,
                "text": chunk,
            },
        }
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    index = get_index()
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(
            vectors=records[start : start + UPSERT_BATCH_SIZE],
            namespace=namespace,
            show_progress=False,
        )

    return {
        "chunks_indexed": len(records),
        "chunks_replaced": replaced,
        "embedding_tokens": embedding_tokens,
    }


def retrieve(client: OpenAI, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], int]:
    """Embed the query and return (top-k chunks with metadata, query embedding tokens). No LLM call."""

    [vector], embedding_tokens = embed_texts(client, [query])
    result = get_index().query(
        vector=vector, top_k=top_k, namespace=_namespace(), include_metadata=True
    )
    hits = []
    for m in result.matches:
        md = dict(m.metadata or {})
        chunk_index = md.get("chunk_index")
        hits.append(
            {
                "id": m.id,
                "score": round(m.score, 4) if m.score is not None else None,
                "document_id": md.get("document_id"),
                # Pinecone stores numbers as floats; return the index as int.
                "chunk_index": int(chunk_index) if chunk_index is not None else None,
                "source": md.get("source"),
                "text": md.get("text"),
            }
        )
    return hits, embedding_tokens
