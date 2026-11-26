"""Shared fixtures. Tests run offline: no LLM calls, no model downloads, no services."""

import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage

from agent_orchestration.api import app
from agent_orchestration.executors import LocalExecutor, get_executor
from agent_orchestration.retrieval import SQLiteStore, get_store
from agent_orchestration.settings import Settings, get_settings

EMBEDDING_DIM = 64


def hashing_embedder(texts: Sequence[str]) -> np.ndarray:
    """Deterministic bag-of-words embedding: texts sharing words are similar."""
    vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
            vectors[row, bucket] += 1.0
    return vectors


class _Runnable:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
        return self._fn(messages)


class ScriptedChatModel:
    """Stands in for ChatOpenAI: returns a fixed plan, fixed tool calls, fixed answer."""

    def __init__(
        self, tool_calls: list[dict[str, Any]] | None = None, answer: str = "Done."
    ) -> None:
        self.tool_calls = tool_calls or []
        self.answer = answer
        self.planned_requests: list[str] = []

    def with_structured_output(self, schema, **_: Any) -> _Runnable:
        def plan(messages: list[BaseMessage]):
            self.planned_requests.append(str(messages[-1].content))
            return schema(steps=["Use the right tool"], rationale="Scripted plan.")

        return _Runnable(plan)

    def bind_tools(self, _tools) -> _Runnable:
        return _Runnable(lambda _messages: AIMessage(content="", tool_calls=self.tool_calls))

    def invoke(self, _messages: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(content=self.answer)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_api_key="", approval_api_key="")


@pytest.fixture
def embedder():
    return hashing_embedder


@pytest.fixture
def store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "documents.db", hashing_embedder, seed=True)


@pytest.fixture
def executor() -> LocalExecutor:
    return LocalExecutor(
        run=lambda request, thread_id: {
            "status": "completed",
            "thread_id": thread_id,
            "plan": {"steps": ["Answer"], "rationale": "Scripted."},
            "response": f"echo: {request}",
        },
        resume=lambda thread_id, approved: {
            "status": "completed",
            "thread_id": thread_id,
            "plan": {"steps": ["Resume"], "rationale": "Scripted."},
            "response": f"approved={approved}",
        },
    )


@pytest.fixture
def client(settings, store, executor) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_executor] = lambda: executor
    try:
        yield TestClient(app, client=("127.0.0.1", 50000))
    finally:
        app.dependency_overrides.clear()
