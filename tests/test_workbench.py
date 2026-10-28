from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from redis.exceptions import ResponseError

from agent_orchestration.api import app
from agent_orchestration import api as api_module
from agent_orchestration.graph import build_graph
from agent_orchestration.settings import get_settings
from agent_orchestration.tools import build_tools


def test_react_workbench_is_available() -> None:
    frontend = Path(__file__).parents[1] / "frontend"

    assert (frontend / "package.json").is_file()
    app_source = (frontend / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "What should the agent do?" in app_source
    assert "Approve action" in app_source
    assert "Documents & hybrid search" in app_source


def test_health_does_not_expose_provider_credentials(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("APPROVAL_API_KEY", "")
    monkeypatch.setattr(api_module, "_check_worker_services", lambda: (True, True, True))
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/health")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json()["llm_credential_configured"] is False
    assert response.json()["llm_provider"] == "OmniRoute (OpenAI-compatible)"
    assert "api_key" not in response.json()


def test_health_reports_ready() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["broker_connected"] is True
    assert response.json()["worker_connected"] is True


def test_service_probe_reports_healthy() -> None:
    assert api_module._check_worker_services() == (True, True, True)


def test_run_queues_successfully() -> None:
    response = TestClient(app).post(
        "/agents/run",
        json={"request": "Hello"},
    )
    assert response.status_code == 202
    assert "task_id" in response.json()


def test_run_executes_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_orchestration.tasks.run_graph",
        lambda req, thread_id, resume=None: {
            "status": "completed",
            "thread_id": thread_id,
            "plan": {"steps": ["Say hello"], "rationale": "Greeting user"},
            "response": "Hello! How can I help you today?",
        },
    )
    client = TestClient(app)
    res = client.post("/agents/run", json={"request": "Say hello"})
    assert res.status_code == 202
    task_id = res.json()["task_id"]

    # Poll task status
    import time
    for _ in range(40):
        task_res = client.get(f"/tasks/{task_id}")
        if task_res.json().get("status") in ("SUCCESS", "FAILURE"):
            break
        time.sleep(0.1)

    task_data = client.get(f"/tasks/{task_id}").json()
    assert task_data["status"] == "SUCCESS"
    assert task_data["result"]["response"] == "Hello! How can I help you today?"


def test_approval_rejects_missing_configured_token(monkeypatch) -> None:
    monkeypatch.setenv("APPROVAL_API_KEY", "local-test-token")
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/agents/example-thread/approval",
            json={"approved": True},
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 401


def test_local_approval_queues_resume_without_extra_secret(monkeypatch) -> None:
    class QueuedTask:
        id = str(uuid4())

    monkeypatch.setenv("APPROVAL_API_KEY", "")
    get_settings.cache_clear()
    monkeypatch.setattr(api_module, "_check_worker_services", lambda: (True, True, True))
    monkeypatch.setattr(api_module.celery_app, "send_task", lambda *args, **kwargs: QueuedTask())
    try:
        response = TestClient(app, client=("127.0.0.1", 12345)).post(
            "/agents/example-thread/approval",
            json={"approved": True},
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 202
    assert response.json()["thread_id"] == "example-thread"


def test_graph_compiles_with_omniroute_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    get_settings.cache_clear()
    try:
        graph = build_graph(InMemorySaver())
    finally:
        get_settings.cache_clear()

    assert {"plan", "act", "approval_gate", "tools", "reflect"}.issubset(
        graph.get_graph().nodes
    )


def test_remote_approval_requires_local_secret(monkeypatch) -> None:
    monkeypatch.setenv("APPROVAL_API_KEY", "")
    get_settings.cache_clear()
    try:
        response = TestClient(app, client=("203.0.113.10", 12345)).post(
            "/agents/example-thread/approval",
            json={"approved": True},
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 403


def test_registered_tools_have_typed_input_schemas() -> None:
    tools = build_tools()

    assert {tool.name for tool in tools} == {
        "search_internal_documents",
        "delete_internal_document",
    }
    assert set(tools[0].args_schema.model_fields) == {"query", "top_k"}
    assert set(tools[1].args_schema.model_fields) == {"document_id"}


def test_default_documents_seeded_and_searchable() -> None:
    from agent_orchestration.retrieval import search_documents
    results = search_documents("data-retention policy", top_k=3)
    assert len(results) > 0
    assert any(r.id == "demo-policy" for r in results)
