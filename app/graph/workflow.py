"""LangGraph StateGraph wiring."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, StateGraph

from app.agents import (
    chitchat_agent,
    customer_service_agent,
    data_agent,
    handoff_to_human,
    order_agent,
    route_intent,
    supervisor_aggregate,
    ticket_agent,
)
from app.config import get_settings
from app.graph.state import AgentState
from app.memory import get_checkpointer

_compiled_graph = None


def _route_after_intent(state: AgentState) -> str:
    if state.get("needs_human") and state.get("intent") in {"unclear", None}:
        return "handoff"
    intent = state.get("intent") or "product_query"
    confidence = float(state.get("confidence") or 0)
    if confidence < 0.45:
        return "handoff"
    mapping = {
        "product_query": "customer_service",
        "technical": "customer_service",
        "ticket": "ticket",
        "order_query": "order",
        "data_query": "data",
        "chitchat": "chitchat",
        "unclear": "handoff",
    }
    return mapping.get(intent, "customer_service")


def _after_order(state: AgentState) -> str:
    meta = state.get("meta") or {}
    if meta.get("continue_to_ticket"):
        return "ticket"
    return "supervisor"


def _after_ticket(state: AgentState) -> str:
    settings = get_settings()
    path = state.get("route_path") or []
    if path and path[-1] == "retry" and int(state.get("tool_retries") or 0) <= settings.max_tool_retries:
        return "ticket"
    if state.get("needs_human"):
        return "handoff"
    return "supervisor"


def _after_cs(state: AgentState) -> str:
    if state.get("needs_human"):
        return "handoff"
    return "supervisor"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent_router", route_intent)
    graph.add_node("customer_service", customer_service_agent)
    graph.add_node("ticket", ticket_agent)
    graph.add_node("order", order_agent)
    graph.add_node("data", data_agent)
    graph.add_node("chitchat", chitchat_agent)
    graph.add_node("supervisor", supervisor_aggregate)
    graph.add_node("handoff", handoff_to_human)

    graph.set_entry_point("intent_router")
    graph.add_conditional_edges(
        "intent_router",
        _route_after_intent,
        {
            "customer_service": "customer_service",
            "ticket": "ticket",
            "order": "order",
            "data": "data",
            "chitchat": "chitchat",
            "handoff": "handoff",
        },
    )
    graph.add_conditional_edges(
        "customer_service",
        _after_cs,
        {"supervisor": "supervisor", "handoff": "handoff"},
    )
    graph.add_conditional_edges(
        "order",
        _after_order,
        {"ticket": "ticket", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "ticket",
        _after_ticket,
        {"ticket": "ticket", "supervisor": "supervisor", "handoff": "handoff"},
    )
    graph.add_edge("data", "supervisor")
    graph.add_edge("chitchat", "supervisor")
    graph.add_edge("supervisor", END)
    graph.add_edge("handoff", END)

    return graph.compile(checkpointer=get_checkpointer())


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    *,
    query: str,
    thread_id: str,
    trace_id: str,
    user_id: str = "u001",
    role: str = "user",
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    graph = get_graph()
    settings = get_settings()
    history = list(messages or [])
    history.append({"role": "user", "content": query})
    initial: AgentState = {
        "messages": history,
        "user_id": user_id,
        "role": role,
        "thread_id": thread_id,
        "trace_id": trace_id,
        "query": query,
        "route_path": [],
        "evidence": [],
        "tool_retries": 0,
        "token_usage": 0,
        "needs_human": False,
        "meta": {},
    }
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_graph_steps,
    }
    result = graph.invoke(initial, config=config)
    final = result.get("final_answer") or result.get("agent_answer") or ""
    history.append({"role": "assistant", "content": final})
    result["messages"] = history
    return result
