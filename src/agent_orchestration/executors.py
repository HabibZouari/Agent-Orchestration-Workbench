"""Background execution of agent runs.

`LocalExecutor` runs graph invocations on a thread pool inside the API process;
`CeleryExecutor` hands them to Celery workers over Redis. The API only depends
on the shared `TaskExecutor` interface.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Protocol
from uuid import uuid4

from agent_orchestration.schemas import TaskStatusResponse
from agent_orchestration.settings import get_settings

logger = logging.getLogger(__name__)

RunFn = Callable[[str, str], dict[str, Any]]
ResumeFn = Callable[[str, bool], dict[str, Any]]


class ExecutorUnavailableError(RuntimeError):
    """The task backend cannot accept or report work right now."""


class UnknownTaskError(KeyError):
    """No task with this ID exists."""


@dataclass(frozen=True)
class RuntimeHealth:
    broker_connected: bool
    redis_search_available: bool
    worker_connected: bool

    @property
    def ready(self) -> bool:
        return self.broker_connected and self.redis_search_available and self.worker_connected


class TaskExecutor(Protocol):
    def submit_run(self, request: str, thread_id: str) -> str: ...

    def submit_resume(self, thread_id: str, approved: bool) -> str: ...

    def status(self, task_id: str) -> TaskStatusResponse: ...

    def health(self) -> RuntimeHealth: ...


def _default_run(request: str, thread_id: str) -> dict[str, Any]:
    from agent_orchestration.graph import run_graph

    return run_graph(request, thread_id)


def _default_resume(thread_id: str, approved: bool) -> dict[str, Any]:
    from agent_orchestration.graph import resume_graph

    return resume_graph(thread_id, approved)


class LocalExecutor:
    """Thread-pool executor; task state lives in memory for the API's lifetime."""

    def __init__(
        self,
        run: RunFn = _default_run,
        resume: ResumeFn = _default_resume,
        *,
        max_workers: int = 4,
        max_tracked_tasks: int = 1_000,
    ) -> None:
        self._run = run
        self._resume = resume
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-task")
        self._tasks: OrderedDict[str, TaskStatusResponse] = OrderedDict()
        self._tasks_lock = threading.Lock()
        self._thread_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._max_tracked_tasks = max_tracked_tasks

    def submit_run(self, request: str, thread_id: str) -> str:
        return self._submit(thread_id, self._run, request, thread_id)

    def submit_resume(self, thread_id: str, approved: bool) -> str:
        return self._submit(thread_id, self._resume, thread_id, approved)

    def status(self, task_id: str) -> TaskStatusResponse:
        with self._tasks_lock:
            if task_id not in self._tasks:
                raise UnknownTaskError(task_id)
            return self._tasks[task_id]

    def health(self) -> RuntimeHealth:
        return RuntimeHealth(True, True, True)

    def _set(self, task_id: str, **fields: Any) -> None:
        with self._tasks_lock:
            self._tasks[task_id] = TaskStatusResponse(task_id=task_id, **fields)
            while len(self._tasks) > self._max_tracked_tasks:
                self._tasks.popitem(last=False)

    def _submit(self, thread_id: str, fn: Callable[..., dict[str, Any]], *args: Any) -> str:
        task_id = str(uuid4())
        self._set(task_id, status="PENDING")
        self._pool.submit(self._execute, task_id, thread_id, fn, *args)
        return task_id

    def _execute(
        self, task_id: str, thread_id: str, fn: Callable[..., dict[str, Any]], *args: Any
    ) -> None:
        with self._tasks_lock:
            thread_lock = self._thread_locks[thread_id]
        with thread_lock:
            self._set(task_id, status="STARTED")
            try:
                self._set(task_id, status="SUCCESS", result=fn(*args))
            except Exception as error:
                logger.exception("agent_task_failed", extra={"task_id": task_id})
                self._set(task_id, status="FAILURE", error=str(error) or type(error).__name__)


class CeleryExecutor:
    """Dispatches to Celery workers; task state lives in the Redis result backend."""

    _STATUS_MAP: ClassVar[dict[str, str]] = {
        "PENDING": "PENDING",
        "RECEIVED": "PENDING",
        "STARTED": "STARTED",
        "RETRY": "STARTED",
        "SUCCESS": "SUCCESS",
        "FAILURE": "FAILURE",
        "REVOKED": "FAILURE",
    }

    def __init__(self, redis_url: str, *, health_ttl_seconds: float = 3.0) -> None:
        from agent_orchestration.celery_app import create_celery_app

        self._app = create_celery_app(redis_url)
        self._redis_url = redis_url
        self._health_ttl = health_ttl_seconds
        self._health_cache: tuple[float, RuntimeHealth] | None = None

    def submit_run(self, request: str, thread_id: str) -> str:
        from agent_orchestration.celery_app import RUN_TASK

        return self._send(RUN_TASK, [request, thread_id])

    def submit_resume(self, thread_id: str, approved: bool) -> str:
        from agent_orchestration.celery_app import RESUME_TASK

        return self._send(RESUME_TASK, [thread_id, approved])

    def _send(self, name: str, args: list[object]) -> str:
        from kombu.exceptions import OperationalError

        try:
            return self._app.send_task(name, args=args).id
        except OperationalError as error:
            logger.exception("celery_task_enqueue_failed", extra={"task_name": name})
            raise ExecutorUnavailableError("The Redis task queue is unavailable.") from error

    def status(self, task_id: str) -> TaskStatusResponse:
        from celery.result import AsyncResult
        from redis.exceptions import RedisError

        result = AsyncResult(task_id, app=self._app)
        try:
            state = result.state
            payload = result.result
        except RedisError as error:
            raise ExecutorUnavailableError("The Redis result backend is unavailable.") from error
        status = self._STATUS_MAP.get(state, "PENDING")
        if status == "SUCCESS":
            return TaskStatusResponse(task_id=task_id, status=status, result=payload)
        if status == "FAILURE":
            return TaskStatusResponse(task_id=task_id, status=status, error=str(payload))
        return TaskStatusResponse(task_id=task_id, status=status)

    def health(self) -> RuntimeHealth:
        now = time.monotonic()
        if self._health_cache and now - self._health_cache[0] < self._health_ttl:
            return self._health_cache[1]
        health = self._probe()
        self._health_cache = (now, health)
        return health

    def _probe(self) -> RuntimeHealth:
        from redis import Redis
        from redis.exceptions import RedisError, ResponseError

        client = Redis.from_url(self._redis_url, socket_connect_timeout=1, socket_timeout=2)
        try:
            client.ping()
        except RedisError:
            return RuntimeHealth(False, False, False)
        try:
            client.execute_command("FT._LIST")
            search_available = True
        except ResponseError:
            search_available = False
        try:
            worker_connected = bool(self._app.control.ping(timeout=1.0))
        except Exception:
            logger.warning("celery_worker_ping_failed", exc_info=True)
            worker_connected = False
        return RuntimeHealth(True, search_available, worker_connected)


@lru_cache
def get_executor() -> TaskExecutor:
    settings = get_settings()
    if settings.task_backend == "celery":
        return CeleryExecutor(settings.redis_url)
    return LocalExecutor()
