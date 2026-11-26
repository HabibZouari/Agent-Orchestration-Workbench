import threading
import time

import pytest

from agent_orchestration.executors import CeleryExecutor, LocalExecutor, UnknownTaskError


def _wait(executor: LocalExecutor, task_id: str):
    for _ in range(200):
        status = executor.status(task_id)
        if status.status in ("SUCCESS", "FAILURE"):
            return status
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_runs_on_the_same_thread_are_serialized() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def slow_run(request, thread_id):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"status": "completed"}

    executor = LocalExecutor(run=slow_run, resume=lambda *_: {})
    ids = [executor.submit_run("x", "same-thread") for _ in range(3)]

    assert all(_wait(executor, task_id).status == "SUCCESS" for task_id in ids)
    assert peak == 1


def test_old_task_records_are_evicted() -> None:
    executor = LocalExecutor(run=lambda *_: {}, resume=lambda *_: {}, max_tracked_tasks=2)
    first = executor.submit_run("a", "t1")
    _wait(executor, first)
    for thread in ("t2", "t3"):
        _wait(executor, executor.submit_run("b", thread))

    with pytest.raises(UnknownTaskError):
        executor.status(first)


def test_celery_health_reports_unreachable_redis() -> None:
    health = CeleryExecutor("redis://127.0.0.1:1/0").health()

    assert (health.broker_connected, health.redis_search_available, health.worker_connected) == (
        False,
        False,
        False,
    )
    assert health.ready is False
