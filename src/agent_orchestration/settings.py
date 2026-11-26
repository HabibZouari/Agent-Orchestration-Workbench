from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables and `.env`."""

    # OpenAI-compatible chat-completions endpoint (OmniRoute, OpenAI, vLLM, Ollama, ...).
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:20128/v1"
    llm_model: str = "gemma-2-2b"

    # "local": in-process thread pool + in-memory LangGraph checkpoints (no services needed).
    # "celery": Celery workers with a Redis broker + Redis LangGraph checkpoints.
    task_backend: Literal["local", "celery"] = "local"
    redis_url: str = "redis://localhost:6379/0"

    # "sqlite": embedded store with FTS5 keyword search + NumPy vector similarity.
    # "pgvector": PostgreSQL full-text search + pgvector HNSW index.
    vector_store: Literal["sqlite", "pgvector"] = "sqlite"
    sqlite_path: Path = Path("data/documents.db")
    database_url: str = "postgresql://agent:agent@localhost:5432/agent_orchestration"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    seed_sample_documents: bool = True

    # Optional shared secret for approval requests. Without it, approvals are loopback-only.
    approval_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
