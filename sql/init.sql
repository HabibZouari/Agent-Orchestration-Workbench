CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_content_search_idx
    ON documents USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING HNSW (embedding vector_cosine_ops);
