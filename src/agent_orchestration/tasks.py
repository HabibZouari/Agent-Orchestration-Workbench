import logging
import threading
from typing import Any
from uuid import uuid4

from agent_orchestration.graph import run_graph

logger = logging.getLogger(__name__)

_TASK_STORE: dict[str, dict[str, Any]] = {}


class AsyncResultMock:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.id = task_id


class CeleryAppMock:
    class Control:
        def inspect(self, timeout=None):
            class Inspector:
                def ping(self):
                    return {"worker@local": {"ok": "pong"}}
            return Inspector()

    def __init__(self):
        self.control = self.Control()
        self.backend = None

    def send_task(self, name: str, args: list[Any] = None) -> AsyncResultMock:
        task_id = str(uuid4())
        args = args or []
        if name == "agent_orchestration.run":
            req, thread_id = args[0], args[1]
            t = threading.Thread(target=_run_task_thread, args=(task_id, run_graph, req, thread_id), daemon=True)
            t.start()
        elif name == "agent_orchestration.resume":
            thread_id, approved = args[0], args[1]
            t = threading.Thread(target=_run_task_thread, args=(task_id, run_graph, "", thread_id, {"approved": approved}), daemon=True)
            t.start()
        return AsyncResultMock(task_id)


def _run_task_thread(task_id: str, func, *args):
    _TASK_STORE[task_id] = {"status": "STARTED"}
    try:
        res = func(*args)
        _TASK_STORE[task_id] = {"status": "SUCCESS", "result": res}
    except Exception as e:
        logger.exception("task_failed", extra={"task_id": task_id})
        _TASK_STORE[task_id] = {"status": "FAILURE", "error": str(e)}


celery_app = CeleryAppMock()
