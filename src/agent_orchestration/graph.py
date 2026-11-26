"""LangGraph workflow: plan -> act -> (approval gate) -> tools -> reflect."""

import logging
import threading
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent_orchestration.settings import get_settings
from agent_orchestration.tools import TOOL_RISK, build_tools

logger = logging.getLogger(__name__)


class NoPendingApprovalError(LookupError):
    """Raised when resuming a thread that is not waiting for approval."""


class AgentPlan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=10)
    rationale: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: AgentPlan
    final_response: str
    approval_required: bool
    approval_granted: bool


def _build_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY must be configured for the OpenAI-compatible provider.")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


def _latest_request(messages: Sequence[BaseMessage]) -> HumanMessage:
    """The newest user message, so follow-up requests on a thread are handled."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return HumanMessage(content="")


def build_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    model: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
):
    model = model if model is not None else _build_model()
    tools = tools if tools is not None else build_tools()
    model_with_tools = model.bind_tools(tools)
    tool_executor = ToolNode(tools)

    def log_step(step: str, config: RunnableConfig, **fields: object) -> None:
        logger.info(
            "agent_step",
            extra={
                "step": step,
                "thread_id": config.get("configurable", {}).get("thread_id"),
                **fields,
            },
        )

    def plan_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        request = _latest_request(state["messages"])
        plan = model.with_structured_output(AgentPlan, method="json_schema").invoke(
            [
                SystemMessage(
                    content="Create a concise, safe execution plan. Do not execute actions."
                ),
                HumanMessage(content=str(request.content)),
            ]
        )
        log_step("plan", config, step_count=len(plan.steps))
        return {"plan": plan}

    def act_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        "Carry out the request using the available tools. "
                        "Use 'search_internal_documents' to search internal documents and "
                        "'delete_internal_document' to delete one. When the user asks about "
                        "documents, policies, or internal information, you MUST call "
                        "'search_internal_documents'. Do not claim you lack access to internal "
                        "documents. Do not claim an action succeeded unless a tool confirms it. "
                        f"Execution plan: {state['plan'].model_dump_json()}"
                    )
                ),
                _latest_request(state["messages"]),
            ]
        )
        log_step("act", config, tool_names=[call["name"] for call in response.tool_calls])
        return {"messages": [response], "approval_required": False, "approval_granted": False}

    def approval_gate_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected an AIMessage with tool calls at the approval gate.")
        high_risk_calls = [
            call
            for call in last_message.tool_calls
            if TOOL_RISK.get(call["name"], "high") == "high"
        ]
        if not high_risk_calls:
            log_step("approval_gate", config, required=False)
            return {"approval_required": False, "approval_granted": True}

        log_step(
            "approval_gate",
            config,
            required=True,
            tools=[call["name"] for call in high_risk_calls],
        )
        decision = interrupt(
            {
                "reason": "High-risk tool call requires human approval.",
                "tool_calls": [
                    {"name": call["name"], "arguments": call["args"]} for call in high_risk_calls
                ],
            }
        )
        approved = (
            bool(decision.get("approved", False)) if isinstance(decision, dict) else bool(decision)
        )
        log_step("approval_decision", config, approved=approved)
        return {"approval_required": True, "approval_granted": approved}

    def denied_tools_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        # A rejected batch is denied as a whole: every pending tool call gets a
        # "not approved" result so the model never sees a dangling tool call.
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected an AIMessage with tool calls at the approval gate.")
        denied = [
            ToolMessage(content="Action was not approved by a human.", tool_call_id=call["id"])
            for call in last_message.tool_calls
        ]
        log_step("tool_execution_denied", config, count=len(denied))
        return {"messages": denied}

    def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected an AIMessage with tool calls at the tool executor.")
        tool_names = [call["name"] for call in last_message.tool_calls]
        try:
            result = tool_executor.invoke(state, config)
        except Exception:
            logger.exception(
                "agent_tool_execution_failed",
                extra={
                    "thread_id": config.get("configurable", {}).get("thread_id"),
                    "tool_names": tool_names,
                },
            )
            raise
        log_step("tool_execution_completed", config, tool_names=tool_names)
        return result

    def reflect_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Reflect on the plan and tool results. Give the user a concise, "
                        "truthful answer. Mention any denied or unsuccessful action."
                    )
                ),
                *state["messages"],
            ]
        )
        log_step("reflect", config)
        return {"final_response": str(response.content)}

    def route_after_act(state: AgentState) -> str:
        message = state["messages"][-1]
        has_calls = isinstance(message, AIMessage) and message.tool_calls
        return "approval_gate" if has_calls else "reflect"

    def route_after_gate(state: AgentState) -> str:
        if state.get("approval_required") and not state.get("approval_granted"):
            return "denied_tools"
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("act", act_node)
    builder.add_node("approval_gate", approval_gate_node)
    builder.add_node("tools", tools_node)
    builder.add_node("denied_tools", denied_tools_node)
    builder.add_node("reflect", reflect_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "act")
    builder.add_conditional_edges("act", route_after_act, ["approval_gate", "reflect"])
    builder.add_conditional_edges("approval_gate", route_after_gate, ["tools", "denied_tools"])
    builder.add_edge("tools", "reflect")
    builder.add_edge("denied_tools", "reflect")
    builder.add_edge("reflect", END)
    return builder.compile(checkpointer=checkpointer)


def create_checkpointer() -> BaseCheckpointSaver:
    """In-memory checkpoints for the local backend; Redis when work is distributed.

    Celery workers run in separate processes, so a paused thread must be stored
    somewhere every worker can read when the approval decision arrives.
    """
    settings = get_settings()
    if settings.task_backend == "celery":
        from langgraph.checkpoint.redis import RedisSaver

        saver = RedisSaver(redis_url=settings.redis_url)
        saver.setup()
        return saver
    return InMemorySaver()


_graph_lock = threading.Lock()
_graph: Any = None


def get_graph():
    """Compile the workflow once per process."""
    global _graph
    with _graph_lock:
        if _graph is None:
            _graph = build_graph(create_checkpointer())
        return _graph


def _summarize(thread_id: str, result: dict[str, Any]) -> dict[str, object]:
    plan = result["plan"].model_dump()
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "plan": plan,
            "interrupts": [item.value for item in interrupts],
        }
    return {
        "status": "completed",
        "thread_id": thread_id,
        "plan": plan,
        "response": result.get("final_response", ""),
    }


def run_graph(request: str, thread_id: str, graph=None) -> dict[str, object]:
    graph = graph if graph is not None else get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=request)]}, config=config)
    return _summarize(thread_id, result)


def resume_graph(thread_id: str, approved: bool, graph=None) -> dict[str, object]:
    graph = graph if graph is not None else get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not any(task.interrupts for task in snapshot.tasks):
        raise NoPendingApprovalError(f"Thread {thread_id!r} has no action awaiting approval.")
    result = graph.invoke(Command(resume={"approved": approved}), config=config)
    return _summarize(thread_id, result)
