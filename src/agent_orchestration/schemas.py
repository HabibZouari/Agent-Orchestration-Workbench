from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = Field(default=None, max_length=200)


class AgentRunResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str


class ApprovalRequest(BaseModel):
    approved: bool


class DocumentInput(BaseModel):
    id: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)


class DocumentUpsertRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1, max_length=100)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict[str, object]
    score: float
