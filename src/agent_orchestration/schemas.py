from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)


class AgentRunResponse(BaseModel):
    task_id: str
    thread_id: str
    status: Literal["queued"] = "queued"


class ApprovalRequest(BaseModel):
    approved: bool


class TaskStatusResponse(BaseModel):
    task_id: str
    status: Literal["PENDING", "STARTED", "SUCCESS", "FAILURE"]
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    task_backend: str
    vector_store: str
    llm_provider: str
    llm_model: str
    llm_credential_configured: bool
    approval_token_required: bool
    broker_connected: bool
    redis_search_available: bool
    worker_connected: bool


class DocumentInput(BaseModel):
    id: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUpsertRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1, max_length=100)


class DocumentSummary(BaseModel):
    id: str
    preview: str
    metadata: dict[str, Any]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
    score: float
