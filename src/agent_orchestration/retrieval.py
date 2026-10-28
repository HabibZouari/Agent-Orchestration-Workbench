import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
from fastembed import TextEmbedding

from agent_orchestration.schemas import DocumentInput, SearchResult
from agent_orchestration.settings import get_settings

DB_PATH = Path(__file__).parents[2] / "local_documents.db"


@lru_cache
def _embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


def _embed(texts: list[str]) -> list[np.ndarray]:
    return list(_embedding_model().embed(texts))


def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()
    # Auto-seed default sample documents if table is empty
    cursor = conn.execute("SELECT COUNT(*) FROM documents")
    if cursor.fetchone()[0] == 0:
        conn.close()
        seed_default_documents()
        conn = sqlite3.connect(DB_PATH)
    return conn


def seed_default_documents() -> int:
    sample_docs = [
        DocumentInput(
            id="demo-policy",
            content=(
                "Company Data Retention Policy: All customer personal data and system transaction "
                "logs must be retained for exactly 365 days (1 year) in encrypted cold storage. "
                "After 365 days, automated purging scripts securely shred the records unless an "
                "active legal hold applies. Department managers must audit retention compliance quarterly."
            ),
            metadata={"category": "compliance", "owner": "legal"},
        ),
        DocumentInput(
            id="security-guidelines",
            content=(
                "Security Incident Response Guidelines: In the event of a suspected credential "
                "compromise or unauthorized access attempt, engineers must immediately notify the "
                "security team via #sec-alerts, rotate all service tokens, and preserve container "
                "logs for forensic analysis. Deleting production documents requires dual authorization."
            ),
            metadata={"category": "security", "owner": "infosec"},
        ),
        DocumentInput(
            id="agent-playbook",
            content=(
                "Agent Orchestration Architecture: The agent system follows a plan-act-approval-reflect "
                "pipeline. High-risk actions like document deletion require explicit human approval "
                "via the approval gate node before execution. Retrieval combines semantic vector "
                "embeddings with lexical keyword matching for robust hybrid search."
            ),
            metadata={"category": "engineering", "owner": "platform"},
        ),
    ]
    return upsert_documents(sample_docs)


def upsert_documents(documents: list[DocumentInput]) -> int:
    embeddings = _embed([doc.content for doc in documents])
    conn = _init_db()
    try:
        for doc, emb in zip(documents, embeddings, strict=True):
            blob = emb.astype(np.float32).tobytes()
            conn.execute(
                """
                INSERT INTO documents (id, content, metadata, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    metadata = excluded.metadata,
                    embedding = excluded.embedding
                """,
                (doc.id, doc.content, "{}", blob),
            )
        conn.commit()
    finally:
        conn.close()
    return len(documents)


def delete_document(document_id: str) -> bool:
    conn = _init_db()
    try:
        cursor = conn.execute("DELETE FROM documents WHERE id = ? RETURNING id", (document_id,))
        row = cursor.fetchone()
        conn.commit()
        return row is not None
    except sqlite3.OperationalError:
        # Fallback for older sqlite without RETURNING
        cursor = conn.execute("SELECT id FROM documents WHERE id = ?", (document_id,))
        if cursor.fetchone():
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def search_documents(query: str, top_k: int = 5) -> list[SearchResult]:
    query_emb = _embed([query])[0].astype(np.float32)
    conn = _init_db()
    try:
        cursor = conn.execute("SELECT id, content, metadata, embedding FROM documents")
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    scored = []
    for row in rows:
        doc_id, content, metadata_str, emb_blob = row
        doc_emb = np.frombuffer(emb_blob, dtype=np.float32)

        # Cosine similarity
        norm_q = np.linalg.norm(query_emb)
        norm_d = np.linalg.norm(doc_emb)
        if norm_q > 0 and norm_d > 0:
            similarity = float(np.dot(query_emb, doc_emb) / (norm_q * norm_d))
        else:
            similarity = 0.0

        # Lexical keyword match bonus (simple substring / word match)
        query_words = set(query.lower().split())
        content_lower = content.lower()
        lexical_match = sum(1 for w in query_words if w in content_lower) / max(len(query_words), 1)

        # Hybrid score (0.7 semantic + 0.3 lexical)
        score = 0.7 * max(0.0, similarity) + 0.3 * lexical_match
        scored.append((score, doc_id, content, {}))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    return [
        SearchResult(
            id=doc_id,
            content=content,
            metadata=metadata,
            score=score,
        )
        for score, doc_id, content, metadata in top_results
    ]
