"""Celery application for TASK_BACKEND=celery.

Start a worker with:
    celery -A agent_orchestration.celery_app worker --loglevel=info --pool=solo
(`--pool=solo` is required on Windows; use the default prefork pool on Linux.)
"""

from celery import Celery
from redis import Redis

from agent_orchestration.settings import get_settings

RUN_TASK = "agent_orchestration.run"
RESUME_TASK = "agent_orchestration.resume"
THREAD_LOCK_SECONDS = 15 * 60


def create_celery_app(redis_url: str | None = None) -> Celery:
    url = redis_url or get_settings().redis_url
    app = Celery("agent_orchestration", broker=url, backend=url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        result_expires=24 * 60 * 60,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
    )
    return app


celery_app = create_celery_app()


def _thread_lock(thread_id: str):
    """One run per LangGraph thread at a time, across all workers."""
    client = Redis.from_url(get_settings().redis_url)
    return client.lock(
        f"agent_orchestration:thread:{thread_id}",
        timeout=THREAD_LOCK_SECONDS,
        blocking_timeout=THREAD_LOCK_SECONDS,
    )


@celery_app.task(name=RUN_TASK)
def run_agent(request: str, thread_id: str) -> dict[str, object]:
    from agent_orchestration.graph import run_graph

    with _thread_lock(thread_id):
        return run_graph(request, thread_id)


@celery_app.task(name=RESUME_TASK)
def resume_agent(thread_id: str, approved: bool) -> dict[str, object]:
    from agent_orchestration.graph import resume_graph

    with _thread_lock(thread_id):
        return resume_graph(thread_id, approved)
