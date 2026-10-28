import logging
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent_orchestration.settings import get_settings
from agent_orchestration.tools import TOOL_RISK, build_tools

logger = logging.getLogger(__name__)


def _build_model() -> BaseChatModel:
    settings = get_settings()
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY must be configured for the OmniRoute provider.")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


class AgentPlan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=10)
    rationale: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: AgentPlan
    final_response: str
    approval_required: bool
    approval_granted: bool


def build_graph(checkpointer: BaseCheckpointSaver):
    model = _build_model()
    tools = build_tools()
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
        request = next(
            (message.content for message in state["messages"] if isinstance(message, HumanMessage)),
            "",
        )
        plan = model.with_structured_output(
            AgentPlan,
            method="json_schema",
        ).invoke(
            [
                SystemMessage(
                    content="Create a concise, safe execution plan. Do not execute actions."
                ),
                HumanMessage(content=str(request)),
            ]
        )
        log_step("plan", config, step_count=len(plan.steps))
        return {"plan": plan}

    def act_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        request = next(
            (message for message in state["messages"] if isinstance(message, HumanMessage)),
            HumanMessage(content=""),
        )
        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        "Carry out the request using the available tools. "
                        "You have access to the tool 'search_internal_documents' to search internal documents, "
                        "and 'delete_internal_document' to delete documents. "
                        "When the user asks to search documents, policies, or internal information, you MUST call "
                        "the 'search_internal_documents' tool. Do not claim you lack access to internal documents. "
                        "Do not claim an action succeeded unless a tool confirms it. "
                        f"Execution plan: {state['plan'].model_dump_json()}"
                    )
                ),
                request,
            ]
        )
        log_step(
            "act",
            config,
            tool_names=[call["name"] for call in response.tool_calls],
        )
        return {"messages": [response], "approval_required": False, "approval_granted": False}

    def approval_gate_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected an AIMessage with tool calls at approval gate.")
        calls = last_message.tool_calls
        high_risk_calls = [
            call for call in calls if TOOL_RISK.get(call["name"], "high") == "high"
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
                    {"name": call["name"], "arguments": call["args"]}
                    for call in high_risk_calls
                ],
            }
        )
        approved = (
            decision.get("approved", False)
            if isinstance(decision, dict)
            else bool(decision)
        )
        log_step("approval_decision", config, approved=approved)
        return {"approval_required": True, "approval_granted": approved}

    def denied_tools_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected an AIMessage with tool calls at approval gate.")
        denied = [
            ToolMessage(
                content="Action was not approved by a human.",
                tool_call_id=call["id"],
            )
            for call in last_message.tool_calls
        ]
        log_step("tool_execution_denied", config, count=len(denied))
        return {"messages": denied}

    def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, object]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Expected an AIMessage with tool calls at tool executor.")
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
        return "approval_gate" if isinstance(message, AIMessage) and message.tool_calls else "reflect"

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
    builder.add_conditional_edges("act", route_after_act)
    builder.add_conditional_edges("approval_gate", route_after_gate)
    builder.add_edge("tools", "reflect")
    builder.add_edge("denied_tools", "reflect")
    builder.add_edge("reflect", END)
    return builder.compile(checkpointer=checkpointer)


# Global in-memory checkpointer so threads persist in memory during the session
from langgraph.checkpoint.memory import InMemorySaver
_global_checkpointer = InMemorySaver()

def run_graph(request: str, thread_id: str, resume: object | None = None) -> dict[str, object]:
    graph = build_graph(_global_checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        result = graph.invoke(Command(resume=resume), config=config)
    else:
        result = graph.invoke(
            {"messages": [HumanMessage(content=request)]},
            config=config,
        )

    interrupts = result.get("__interrupt__", [])
    if interrupts:
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "plan": result["plan"].model_dump(),
            "interrupts": [item.value for item in interrupts],
        }
    return {
        "status": "completed",
        "thread_id": thread_id,
        "plan": result["plan"].model_dump(),
        "response": result.get("final_response", ""),
    }
