import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent_orchestration.graph import (
    NoPendingApprovalError,
    build_graph,
    resume_graph,
    run_graph,
)
from agent_orchestration.settings import get_settings
from agent_orchestration.tools import TOOL_RISK, build_tools
from conftest import ScriptedChatModel


def _delete_call(document_id: str) -> list[dict]:
    return [{"name": "delete_internal_document", "args": {"document_id": document_id}, "id": "c1"}]


def _graph(model, store):
    return build_graph(InMemorySaver(), model=model, tools=build_tools(store))


def test_graph_compiles_with_openai_compatible_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    get_settings.cache_clear()
    try:
        graph = build_graph(InMemorySaver())
    finally:
        get_settings.cache_clear()

    assert {"plan", "act", "approval_gate", "tools", "denied_tools", "reflect"} <= set(
        graph.get_graph().nodes
    )


def test_every_tool_has_a_declared_risk_level(store) -> None:
    tools = build_tools(store)

    assert {tool.name for tool in tools} == set(TOOL_RISK)
    assert set(tools[0].args_schema.model_fields) == {"query", "top_k"}
    assert set(tools[1].args_schema.model_fields) == {"document_id"}


def test_plain_question_completes_without_tools(store) -> None:
    outcome = run_graph("Hi", "t-plain", graph=_graph(ScriptedChatModel(answer="Hello!"), store))

    assert outcome["status"] == "completed"
    assert outcome["response"] == "Hello!"
    assert outcome["plan"]["steps"] == ["Use the right tool"]


def test_low_risk_search_runs_without_approval(store) -> None:
    model = ScriptedChatModel(
        tool_calls=[
            {"name": "search_internal_documents", "args": {"query": "retention"}, "id": "c1"}
        ]
    )

    outcome = run_graph("Find the retention policy", "t-search", graph=_graph(model, store))

    assert outcome["status"] == "completed"


def test_delete_pauses_for_approval_then_runs_when_approved(store) -> None:
    graph = _graph(ScriptedChatModel(tool_calls=_delete_call("demo-policy")), store)

    paused = run_graph("Delete demo-policy", "t-approve", graph=graph)

    assert paused["status"] == "awaiting_approval"
    assert paused["interrupts"][0]["tool_calls"] == [
        {"name": "delete_internal_document", "arguments": {"document_id": "demo-policy"}}
    ]
    assert "demo-policy" in {doc.id for doc in store.list_documents()}

    finished = resume_graph("t-approve", approved=True, graph=graph)

    assert finished["status"] == "completed"
    assert "demo-policy" not in {doc.id for doc in store.list_documents()}


def test_rejected_delete_leaves_document_in_place(store) -> None:
    graph = _graph(ScriptedChatModel(tool_calls=_delete_call("demo-policy")), store)
    run_graph("Delete demo-policy", "t-reject", graph=graph)

    finished = resume_graph("t-reject", approved=False, graph=graph)

    assert finished["status"] == "completed"
    assert "demo-policy" in {doc.id for doc in store.list_documents()}


def test_resume_without_pending_approval_is_refused(store) -> None:
    graph = _graph(ScriptedChatModel(tool_calls=_delete_call("demo-policy")), store)
    run_graph("Delete demo-policy", "t-twice", graph=graph)
    resume_graph("t-twice", approved=True, graph=graph)

    with pytest.raises(NoPendingApprovalError):
        resume_graph("t-twice", approved=True, graph=graph)
    with pytest.raises(NoPendingApprovalError):
        resume_graph("never-started", approved=True, graph=graph)


def test_follow_up_request_is_planned_on_the_latest_message(store) -> None:
    model = ScriptedChatModel()
    graph = _graph(model, store)

    run_graph("First question", "t-follow", graph=graph)
    run_graph("Second question", "t-follow", graph=graph)

    assert model.planned_requests == ["First question", "Second question"]
