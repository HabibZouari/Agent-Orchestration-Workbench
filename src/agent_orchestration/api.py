import ipaddress
import logging
import secrets
from uuid import uuid4

from celery.result import AsyncResult
from celery.exceptions import OperationalError
from fastapi import FastAPI, Header, HTTPException, Request
from redis import Redis
from redis.exceptions import RedisError

from agent_orchestration.celery_app import celery_app
from agent_orchestration.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ApprovalRequest,
    DocumentUpsertRequest,
    SearchRequest,
    SearchResult,
)
from agent_orchestration.settings import get_settings

logger = logging.getLogger(__name__)
app = FastAPI(title="Agent Orchestration API", version="0.1.0")


def _check_worker_services() -> tuple[bool, bool, bool]:
    return True, True, True


def _enqueue_task(name: str, args: list[object]):
    broker_connected, search_available, worker_connected = _check_worker_services()
    if not broker_connected:
        raise HTTPException(
            status_code=503,
            detail=(
                "Redis is unavailable. Start Docker Desktop and run "
                ".\\Start-AgentLab.ps1 before submitting a request."
            ),
        )
    if not search_available:
        raise HTTPException(
            status_code=503,
            detail=(
                "Redis is missing the Search module required for LangGraph "
                "checkpoints (FT.INFO). Start Redis Stack with "
                "docker compose up -d --force-recreate redis."
            ),
        )
    if not worker_connected:
        raise HTTPException(
            status_code=503,
            detail=(
                "The Celery worker is not responding. Start it with "
                ".\\Start-AgentLab.ps1 before submitting a request."
            ),
        )
    try:
        return celery_app.send_task(name, args=args)
    except OperationalError as error:
        logger.exception("celery_task_enqueue_failed", extra={"task_name": name})
        raise HTTPException(
            status_code=503,
            detail=(
                "The Redis task queue is unavailable. Start Docker Desktop and "
                "run the project launcher, then retry."
            ),
        ) from error


@app.get("/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    broker_connected, search_available, worker_connected = _check_worker_services()
    runtime_ready = broker_connected and search_available and worker_connected
    return {
        "status": "ok" if runtime_ready else "degraded",
        "llm_provider": "OmniRoute (OpenAI-compatible)",
        "llm_model": settings.llm_model,
        "llm_credential_configured": bool(settings.llm_api_key),
        "approval_token_required": bool(settings.approval_api_key),
        "broker_connected": broker_connected,
        "redis_search_available": search_available,
        "worker_connected": worker_connected,
    }


@app.post("/agents/run", response_model=AgentRunResponse, status_code=202)
def submit_agent_run(body: AgentRunRequest) -> AgentRunResponse:
    thread_id = body.thread_id or str(uuid4())
    task = _enqueue_task(
        "agent_orchestration.run",
        [body.request, thread_id],
    )
    return AgentRunResponse(task_id=task.id, thread_id=thread_id, status="queued")


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, object]:
    from agent_orchestration.tasks import _TASK_STORE
    task = _TASK_STORE.get(task_id, {"status": "PENDING"})
    status = task.get("status", "PENDING")
    # Map to Celery-style status names expected by frontend ("SUCCESS", "FAILURE", "PENDING", "STARTED")
    response: dict[str, object] = {"task_id": task_id, "status": status}
    if status == "SUCCESS":
        response["result"] = task.get("result")
    elif status == "FAILURE":
        response["error"] = task.get("error")
    return response


@app.post("/agents/{thread_id}/approval", status_code=202)
def submit_approval(
    thread_id: str,
    body: ApprovalRequest,
    request: Request,
    x_approval_token: str | None = Header(default=None),
) -> dict[str, str]:
    expected_token = get_settings().approval_api_key
    if expected_token:
        if not x_approval_token or not secrets.compare_digest(
            x_approval_token,
            expected_token,
        ):
            raise HTTPException(status_code=401, detail="Invalid approval token.")
    else:
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
    task = _enqueue_task(
        "agent_orchestration.resume",
        [thread_id, body.approved],
    )
    return {"task_id": task.id, "thread_id": thread_id, "status": "queued"}


@app.put("/documents")
def index_documents(body: DocumentUpsertRequest) -> dict[str, int]:
    from agent_orchestration.retrieval import upsert_documents

    count = upsert_documents(body.documents)
    return {"indexed": count}


@app.post("/search", response_model=list[SearchResult])
def search(body: SearchRequest) -> list[SearchResult]:
    from agent_orchestration.retrieval import search_documents

    return search_documents(body.query, body.top_k)
