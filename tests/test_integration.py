"""Tests against live Redis Stack / PostgreSQL. Skipped unless the URLs are provided:

    docker compose up -d
    TEST_DATABASE_URL=postgresql://agent:agent@localhost:5432/agent_orchestration \
    TEST_REDIS_URL=redis://localhost:6379/0 pytest -m integration

(RediSearch, used by the LangGraph checkpointer, only indexes Redis database 0.)
"""

import os
import subprocess
import sys
import time
import uuid

import pytest

from agent_orchestration.executors import CeleryExecutor
from agent_orchestration.retrieval import PgVectorStore
from agent_orchestration.schemas import DocumentInput
from conftest import EMBEDDING_DIM, hashing_embedder

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
REDIS_URL = os.environ.get("TEST_REDIS_URL")


@pytest.fixture
def pg_store():
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"CREATE SCHEMA {schema}")
    separator = "&" if "?" in DATABASE_URL else "?"
    dsn = f"{DATABASE_URL}{separator}options=-csearch_path%3D{schema},public"
    try:
        yield PgVectorStore(dsn, hashing_embedder, dim=EMBEDDING_DIM, seed=True)
    finally:
        with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_pgvector_store_round_trip(pg_store) -> None:
    assert {doc.id for doc in pg_store.list_documents()} == {
        "demo-policy",
        "security-guidelines",
        "agent-playbook",
    }
    assert pg_store.search("data retention policy", top_k=3)[0].id == "demo-policy"

    pg_store.upsert([DocumentInput(id="faq", content="Giraffe facts", metadata={"v": 2})])
    hit = pg_store.search("giraffe", top_k=1)[0]
    assert (hit.id, hit.metadata) == ("faq", {"v": 2})

    assert pg_store.delete("faq") is True
    assert pg_store.delete("faq") is False


@pytest.fixture
def celery_worker():
    if not REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    env = {
        **os.environ,
        "TASK_BACKEND": "celery",
        "REDIS_URL": REDIS_URL,
        "LLM_API_KEY": "integration-test-placeholder",
    }
    worker = subprocess.Popen(
        [
            *(sys.executable, "-m", "celery", "-A", "agent_orchestration.celery_app"),
            *("worker", "--pool=solo", "--loglevel=warning"),
        ],
        env=env,
    )
    executor = CeleryExecutor(REDIS_URL, health_ttl_seconds=0)
    try:
        deadline = time.monotonic() + 60
        while not executor.health().worker_connected:
            if time.monotonic() > deadline:
                pytest.fail("Celery worker did not come online")
            time.sleep(1)
        yield executor
    finally:
        worker.terminate()
        worker.wait(timeout=30)


def test_celery_worker_round_trip(celery_worker) -> None:
    health = celery_worker.health()
    assert health.ready, health

    # Resuming a thread that never paused exercises the queue, the worker, the Redis
    # checkpointer and the result backend without calling the LLM.
    task_id = celery_worker.submit_resume(f"no-such-thread-{uuid.uuid4()}", True)
    deadline = time.monotonic() + 60
    while (status := celery_worker.status(task_id)).status not in ("SUCCESS", "FAILURE"):
        assert time.monotonic() < deadline, "task did not finish"
        time.sleep(0.5)

    assert status.status == "FAILURE"
    assert "no action awaiting approval" in status.error
