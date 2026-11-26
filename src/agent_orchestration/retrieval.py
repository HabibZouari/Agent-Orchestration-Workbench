"""Document storage and hybrid (keyword + semantic) retrieval.

Two interchangeable stores implement the same `DocumentStore` interface:

* `SQLiteStore`   - zero-setup default. FTS5 (BM25, Porter stemming) for keyword
                    search, cosine similarity in NumPy over stored embeddings.
* `PgVectorStore` - PostgreSQL full-text search plus a pgvector HNSW index.

Both rank the two candidate lists with Reciprocal Rank Fusion (RRF), so scores
mean the same thing regardless of the backend.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agent_orchestration.schemas import DocumentInput, DocumentSummary, SearchResult
from agent_orchestration.settings import get_settings

Embedder = Callable[[Sequence[str]], np.ndarray]

RRF_K = 60
CANDIDATE_POOL = 50
PREVIEW_CHARS = 160

_TOKEN = re.compile(r"\w+", re.UNICODE)


class StoreUnavailableError(RuntimeError):
    """The configured document store cannot be reached."""


class DocumentStore(Protocol):
    def upsert(self, documents: Sequence[DocumentInput]) -> int: ...

    def delete(self, document_id: str) -> bool: ...

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...

    def list_documents(self, limit: int = 100) -> list[DocumentSummary]: ...


SAMPLE_DOCUMENTS: tuple[DocumentInput, ...] = (
    DocumentInput(
        id="demo-policy",
        content=(
            "Company Data Retention Policy: All customer personal data and system transaction "
            "logs must be retained for exactly 365 days (1 year) in encrypted cold storage. "
            "After 365 days, automated purging scripts securely shred the records unless an "
            "active legal hold applies. Department managers must audit retention compliance "
            "quarterly."
        ),
        metadata={"category": "compliance", "owner": "legal", "sample": True},
    ),
    DocumentInput(
        id="security-guidelines",
        content=(
            "Security Incident Response Guidelines: In the event of a suspected credential "
            "compromise or unauthorized access attempt, engineers must immediately notify the "
            "security team via #sec-alerts, rotate all service tokens, and preserve container "
            "logs for forensic analysis. Deleting production documents requires dual "
            "authorization."
        ),
        metadata={"category": "security", "owner": "infosec", "sample": True},
    ),
    DocumentInput(
        id="agent-playbook",
        content=(
            "Agent Orchestration Architecture: The agent system follows a "
            "plan-act-approval-reflect pipeline. High-risk actions like document deletion "
            "require explicit human approval via the approval gate node before execution. "
            "Retrieval combines semantic vector embeddings with lexical keyword matching for "
            "robust hybrid search."
        ),
        metadata={"category": "engineering", "owner": "platform", "sample": True},
    ),
)


def reciprocal_rank_fusion(*rankings: Sequence[str], k: int = RRF_K) -> dict[str, float]:
    """Fuse ranked ID lists: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def query_terms(text: str) -> list[str]:
    """Lower-cased, de-duplicated word tokens; safe to embed in FTS query syntax."""
    return list(dict.fromkeys(_TOKEN.findall(text.lower())))


def _preview(content: str) -> str:
    return content if len(content) <= PREVIEW_CHARS else content[: PREVIEW_CHARS - 1] + "…"


class _HybridStore:
    """Shared embedding, fusion, and seeding logic."""

    def __init__(self, embed: Embedder) -> None:
        self._embed = embed

    def _embed_matrix(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._embed(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def _semantic_ids(self, query_vector: np.ndarray) -> list[str]:
        raise NotImplementedError

    def _lexical_ids(self, terms: list[str]) -> list[str]:
        raise NotImplementedError

    def _fetch(self, ids: Sequence[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        raise NotImplementedError

    def upsert(self, documents: Sequence[DocumentInput]) -> int:
        raise NotImplementedError

    def list_documents(self, limit: int = 100) -> list[DocumentSummary]:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self._embed_matrix([query])[0]
        terms = query_terms(query)
        semantic = self._semantic_ids(query_vector)
        lexical = self._lexical_ids(terms) if terms else []
        fused = reciprocal_rank_fusion(semantic, lexical)
        top = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        rows = self._fetch([doc_id for doc_id, _ in top])
        return [
            SearchResult(id=doc_id, content=rows[doc_id][0], metadata=rows[doc_id][1], score=score)
            for doc_id, score in top
            if doc_id in rows
        ]

    def _seed_if_empty(self) -> None:
        if not self.list_documents(limit=1):
            self.upsert(SAMPLE_DOCUMENTS)


class SQLiteStore(_HybridStore):
    def __init__(self, path: Path, embed: Embedder, *, seed: bool = True) -> None:
        super().__init__(embed)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                    USING fts5(id UNINDEXED, content, tokenize = 'porter unicode61');
                """
            )
        if seed:
            self._seed_if_empty()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert(self, documents: Sequence[DocumentInput]) -> int:
        vectors = self._embed_matrix([doc.content for doc in documents])
        with self._write_lock, self._connection() as conn:
            for doc, vector in zip(documents, vectors, strict=True):
                conn.execute(
                    """
                    INSERT INTO documents (id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        metadata = excluded.metadata,
                        embedding = excluded.embedding
                    """,
                    (doc.id, doc.content, json.dumps(doc.metadata), vector.tobytes()),
                )
                conn.execute("DELETE FROM documents_fts WHERE id = ?", (doc.id,))
                conn.execute(
                    "INSERT INTO documents_fts (id, content) VALUES (?, ?)",
                    (doc.id, doc.content),
                )
        return len(documents)

    def delete(self, document_id: str) -> bool:
        with self._write_lock, self._connection() as conn:
            deleted = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,)).rowcount
            conn.execute("DELETE FROM documents_fts WHERE id = ?", (document_id,))
        return deleted > 0

    def list_documents(self, limit: int = 100) -> list[DocumentSummary]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, content, metadata FROM documents ORDER BY created_at, id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            DocumentSummary(id=doc_id, preview=_preview(content), metadata=json.loads(metadata))
            for doc_id, content, metadata in rows
        ]

    def _semantic_ids(self, query_vector: np.ndarray) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute("SELECT id, embedding FROM documents").fetchall()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        similarities = matrix @ query_vector
        order = np.argsort(-similarities)[:CANDIDATE_POOL]
        return [rows[index][0] for index in order]

    def _lexical_ids(self, terms: list[str]) -> list[str]:
        match = " OR ".join(f'"{term}"' for term in terms)
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts)
                LIMIT ?
                """,
                (match, CANDIDATE_POOL),
            ).fetchall()
        return [row[0] for row in rows]

    def _fetch(self, ids: Sequence[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT id, content, metadata FROM documents WHERE id IN ({placeholders})",
                list(ids),
            ).fetchall()
        return {doc_id: (content, json.loads(metadata)) for doc_id, content, metadata in rows}


class PgVectorStore(_HybridStore):
    def __init__(self, dsn: str, embed: Embedder, *, dim: int, seed: bool = True) -> None:
        super().__init__(embed)
        self._dsn = dsn
        self._dim = int(dim)
        self._ensure_schema()
        if seed:
            self._seed_if_empty()

    @contextmanager
    def _connection(self, *, register: bool = True) -> Iterator[Any]:
        import psycopg
        from pgvector.psycopg import register_vector

        try:
            conn = psycopg.connect(self._dsn, connect_timeout=5)
        except psycopg.OperationalError as error:
            raise StoreUnavailableError(
                "PostgreSQL is unreachable. Start it with 'docker compose up -d postgres' "
                "or set VECTOR_STORE=sqlite."
            ) from error
        try:
            if register:
                register_vector(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection(register=False) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self._dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_content_search_idx "
                "ON documents USING GIN (to_tsvector('english', content))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_embedding_idx "
                "ON documents USING HNSW (embedding vector_cosine_ops)"
            )

    def upsert(self, documents: Sequence[DocumentInput]) -> int:
        from psycopg.types.json import Jsonb

        vectors = self._embed_matrix([doc.content for doc in documents])
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO documents (id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (doc.id, doc.content, Jsonb(doc.metadata), vector)
                    for doc, vector in zip(documents, vectors, strict=True)
                ],
            )
        return len(documents)

    def delete(self, document_id: str) -> bool:
        with self._connection() as conn:
            return conn.execute("DELETE FROM documents WHERE id = %s", (document_id,)).rowcount > 0

    def list_documents(self, limit: int = 100) -> list[DocumentSummary]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, content, metadata FROM documents ORDER BY created_at, id LIMIT %s",
                (limit,),
            ).fetchall()
        return [
            DocumentSummary(id=doc_id, preview=_preview(content), metadata=metadata)
            for doc_id, content, metadata in rows
        ]

    def _semantic_ids(self, query_vector: np.ndarray) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id FROM documents ORDER BY embedding <=> %s LIMIT %s",
                (query_vector, CANDIDATE_POOL),
            ).fetchall()
        return [row[0] for row in rows]

    def _lexical_ids(self, terms: list[str]) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id
                FROM documents, to_tsquery('english', %s) AS query
                WHERE to_tsvector('english', content) @@ query
                ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
                LIMIT %s
                """,
                (" | ".join(terms), CANDIDATE_POOL),
            ).fetchall()
        return [row[0] for row in rows]

    def _fetch(self, ids: Sequence[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not ids:
            return {}
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, content, metadata FROM documents WHERE id = ANY(%s)",
                (list(ids),),
            ).fetchall()
        return {doc_id: (content, metadata) for doc_id, content, metadata in rows}


class FastEmbedEmbedder:
    """Loads the FastEmbed ONNX model on first use (downloads it once)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self._model_name)
        return np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)


_store_lock = threading.Lock()


@lru_cache
def _build_store() -> DocumentStore:
    settings = get_settings()
    embed = FastEmbedEmbedder(settings.embedding_model)
    if settings.vector_store == "pgvector":
        return PgVectorStore(
            settings.database_url,
            embed,
            dim=settings.embedding_dim,
            seed=settings.seed_sample_documents,
        )
    return SQLiteStore(settings.sqlite_path, embed, seed=settings.seed_sample_documents)


def get_store() -> DocumentStore:
    """Process-wide store for the configured backend (created on first use)."""
    with _store_lock:
        return _build_store()
