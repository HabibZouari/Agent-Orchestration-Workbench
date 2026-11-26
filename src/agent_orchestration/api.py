import ipaddress
import logging
import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from agent_orchestration import __version__
from agent_orchestration.executors import (
    ExecutorUnavailableError,
    TaskExecutor,
    UnknownTaskError,
    get_executor,
)
from agent_orchestration.retrieval import DocumentStore, StoreUnavailableError, get_store
from agent_orchestration.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ApprovalRequest,
    DocumentSummary,
    DocumentUpsertRequest,
    HealthResponse,
    SearchRequest,
    SearchResult,
    TaskStatusResponse,
)
from agent_orchestration.settings import Settings, get_settings

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Agent Orchestration API",
    version=__version__,
    description="Human-in-the-loop LangGraph agent with hybrid document retrieval.",
)

SettingsDep = Annotated[Settings, Depends(get_settings)]
ExecutorDep = Annotated[TaskExecutor, Depends(get_executor)]
StoreDep = Annotated[DocumentStore, Depends(get_store)]


@app.exception_handler(ExecutorUnavailableError)
@app.exception_handler(StoreUnavailableError)
async def _service_unavailable(_: Request, error: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


def _require_ready(executor: TaskExecutor) -> None:
    health = executor.health()
    if not health.broker_connected:
        detail = "Redis is unavailable. Start it with 'docker compose up -d redis'."
    elif not health.redis_search_available:
        detail = (
            "Redis is missing the Search module required for LangGraph checkpoints. "
            "Use Redis Stack: 'docker compose up -d --force-recreate redis'."
        )
    elif not health.worker_connected:
        detail = "No Celery worker is responding. Start one (see README) and retry."
    else:
        return
    raise HTTPException(status_code=503, detail=detail)


@app.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep, executor: ExecutorDep) -> HealthResponse:
    runtime = executor.health()
    return HealthResponse(
        status="ok" if runtime.ready else "degraded",
        version=__version__,
        task_backend=settings.task_backend,
        vector_store=settings.vector_store,
        llm_provider="OpenAI-compatible",
        llm_model=settings.llm_model,
        llm_credential_configured=bool(settings.llm_api_key),
        approval_token_required=bool(settings.approval_api_key),
        broker_connected=runtime.broker_connected,
        redis_search_available=runtime.redis_search_available,
        worker_connected=runtime.worker_connected,
    )


@app.post("/agents/run", response_model=AgentRunResponse, status_code=202)
def submit_agent_run(body: AgentRunRequest, executor: ExecutorDep) -> AgentRunResponse:
    _require_ready(executor)
    thread_id = body.thread_id or str(uuid4())
    task_id = executor.submit_run(body.request, thread_id)
    return AgentRunResponse(task_id=task_id, thread_id=thread_id)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str, executor: ExecutorDep) -> TaskStatusResponse:
    try:
        return executor.status(task_id)
    except UnknownTaskError as error:
        raise HTTPException(status_code=404, detail="Unknown task ID.") from error


def _authorize_approval(request: Request, token: str | None, settings: Settings) -> None:
    expected = settings.approval_api_key
    if expected:
        if not token or not secrets.compare_digest(token.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Invalid approval token.")
        return
    client_host = request.client.host if request.client else ""
    try:
        is_local = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_local = False
    if not is_local:
        raise HTTPException(
            status_code=403,
            detail="Configure APPROVAL_API_KEY before approving over a network.",
        )


@app.post("/agents/{thread_id}/approval", response_model=AgentRunResponse, status_code=202)
def submit_approval(
    thread_id: str,
    body: ApprovalRequest,
    request: Request,
    settings: SettingsDep,
    executor: ExecutorDep,
    x_approval_token: Annotated[str | None, Header()] = None,
) -> AgentRunResponse:
    _authorize_approval(request, x_approval_token, settings)
    _require_ready(executor)
    task_id = executor.submit_resume(thread_id, body.approved)
    return AgentRunResponse(task_id=task_id, thread_id=thread_id)


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    store: StoreDep, limit: Annotated[int, Query(ge=1, le=500)] = 100
) -> list[DocumentSummary]:
    return store.list_documents(limit)


@app.put("/documents")
def index_documents(body: DocumentUpsertRequest, store: StoreDep) -> dict[str, int]:
    return {"indexed": store.upsert(body.documents)}


@app.post("/search", response_model=list[SearchResult])
def search(body: SearchRequest, store: StoreDep) -> list[SearchResult]:
    return store.search(body.query, body.top_k)
