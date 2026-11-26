import time

from fastapi.testclient import TestClient

from agent_orchestration.api import app
from agent_orchestration.executors import RuntimeHealth
from agent_orchestration.retrieval import StoreUnavailableError


def _wait_for(client: TestClient, task_id: str) -> dict:
    for _ in range(100):
        task = client.get(f"/tasks/{task_id}").json()
        if task["status"] in ("SUCCESS", "FAILURE"):
            return task
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not finish")


def test_health_reports_config_without_exposing_secrets(client, settings) -> None:
    settings.llm_api_key = "sk-test-secret"
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["task_backend"] == "local"
    assert body["vector_store"] == "sqlite"
    assert body["llm_credential_configured"] is True
    assert "sk-test-secret" not in str(body)


def test_run_is_queued_and_completes(client) -> None:
    response = client.post("/agents/run", json={"request": "Say hello"})

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    task = _wait_for(client, response.json()["task_id"])
    assert task["status"] == "SUCCESS"
    assert task["result"]["response"] == "echo: Say hello"


def test_run_keeps_a_supplied_thread_id(client) -> None:
    response = client.post("/agents/run", json={"request": "Hi", "thread_id": "my-thread"})

    assert response.json()["thread_id"] == "my-thread"


def test_failed_task_reports_its_error(client, executor) -> None:
    def explode(request, thread_id):
        raise RuntimeError("model unreachable")

    executor._run = explode
    task = _wait_for(client, client.post("/agents/run", json={"request": "x"}).json()["task_id"])

    assert task == {
        "task_id": task["task_id"],
        "status": "FAILURE",
        "result": None,
        "error": "model unreachable",
    }


def test_unknown_task_is_404(client) -> None:
    assert client.get("/tasks/does-not-exist").status_code == 404


def test_run_is_refused_when_worker_is_down(client, executor) -> None:
    executor.health = lambda: RuntimeHealth(True, True, False)

    response = client.post("/agents/run", json={"request": "Hi"})

    assert response.status_code == 503
    assert "Celery worker" in response.json()["detail"]
    assert client.get("/health").json()["status"] == "degraded"


def test_empty_request_is_rejected(client) -> None:
    assert client.post("/agents/run", json={"request": ""}).status_code == 422


def test_local_approval_without_token(client) -> None:
    response = client.post("/agents/t1/approval", json={"approved": True})

    assert response.status_code == 202
    assert _wait_for(client, response.json()["task_id"])["result"]["response"] == "approved=True"


def test_remote_approval_requires_configured_token(client) -> None:
    remote = TestClient(app, client=("203.0.113.10", 50000))

    assert remote.post("/agents/t1/approval", json={"approved": True}).status_code == 403


def test_configured_token_is_enforced(client, settings) -> None:
    settings.approval_api_key = "local-test-token"

    missing = client.post("/agents/t1/approval", json={"approved": True})
    wrong = client.post(
        "/agents/t1/approval", json={"approved": True}, headers={"X-Approval-Token": "nope"}
    )
    right = client.post(
        "/agents/t1/approval",
        json={"approved": True},
        headers={"X-Approval-Token": "local-test-token"},
    )

    assert (missing.status_code, wrong.status_code, right.status_code) == (401, 401, 202)


def test_index_list_and_search_documents(client) -> None:
    indexed = client.put(
        "/documents",
        json={"documents": [{"id": "vpn", "content": "VPN setup guide", "metadata": {"t": 1}}]},
    )
    listed = client.get("/documents").json()
    results = client.post("/search", json={"query": "vpn guide", "top_k": 2}).json()

    assert indexed.json() == {"indexed": 1}
    assert "vpn" in {doc["id"] for doc in listed}
    assert results[0]["id"] == "vpn"
    assert results[0]["metadata"] == {"t": 1}


def test_store_outage_maps_to_503(client, store) -> None:
    def unavailable(*args, **kwargs):
        raise StoreUnavailableError("PostgreSQL is unreachable.")

    store.search = unavailable

    response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 503
    assert response.json()["detail"] == "PostgreSQL is unreachable."
