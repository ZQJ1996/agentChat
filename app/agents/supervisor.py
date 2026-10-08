"""Supervisor aggregation and handoff helpers."""

from __future__ import annotations

from typing import Any

from app.graph.state import AgentState
from app.models import ChatMessage, get_model_provider
from app.observability import TraceContext, trace_node
from app.prompts import get_prompt


def supervisor_aggregate(state: AgentState) -> AgentState:
    answer = state.get("agent_answer") or ""
    needs_human = bool(state.get("needs_human"))
    evidence = state.get("evidence") or []
    ticket_id = state.get("ticket_id")
    trace = TraceContext(
        trace_id=state.get("trace_id") or "unknown",
        thread_id=state.get("thread_id") or "unknown",
    )

    parts = [answer]
    if ticket_id:
        parts.append(f"（关联工单：#{ticket_id}）")
    if evidence:
        cites = "；".join(e.get("citation", "") for e in evidence[:3] if e.get("citation"))
        if cites and "参考来源" not in answer:
            parts.append(f"\n来源：{cites}")
    if needs_human:
        parts.append("\n该问题已标记转人工，后续由人工客服跟进。")

    draft = "\n".join(p for p in parts if p).strip()

    with trace_node(trace, "supervisor", input_preview=draft[:200]) as bucket:
        provider = get_model_provider()
        result = provider.chat(
            [
                ChatMessage(role="system", content=get_prompt("system_supervisor")),
                ChatMessage(role="user", content=draft),
            ]
        )
        bucket["tokens"] = result.total_tokens
        final_answer = result.content.strip() or draft
        bucket["output_preview"] = final_answer

    return {
        "final_answer": final_answer,
        "route_path": ["supervisor"],
        "token_usage": int(state.get("token_usage") or 0) + result.total_tokens,
    }


def handoff_to_human(state: AgentState) -> AgentState:
    msg = state.get("agent_answer") or "抱歉，当前无法自动处理，已为您转接人工客服。"
    if "转人工" not in msg and "拦截" not in msg:
        msg = f"{msg}\n已转人工处理。"
    return {
        "needs_human": True,
        "final_answer": msg,
        "agent_answer": msg,
        "route_path": ["handoff_to_human"],
    }


def chitchat_agent(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    provider = get_model_provider()
    result = provider.chat(
        [
            ChatMessage(role="system", content="你是友好的企业助手，简短寒暄并引导用户提出业务问题。"),
            ChatMessage(role="user", content=query),
        ]
    )
    return {
        "agent_answer": result.content,
        "route_path": ["chitchat"],
        "needs_human": False,
        "token_usage": int(state.get("token_usage") or 0) + result.total_tokens,
    }


def order_agent(state: AgentState) -> AgentState:
    """Order lookup; may continue to ticket via graph edge."""
    import re

    from app.tools.order_tools import list_orders, query_order

    query = state.get("query") or ""
    m = re.search(r"ORD\d{3,}", query, re.IGNORECASE)
    if m:
        order = query_order(m.group(0).upper())
        if not order:
            return {
                "agent_answer": f"未找到订单 {m.group(0).upper()}。",
                "route_path": ["order_agent"],
                "needs_human": False,
            }
        answer = (
            f"订单 {order['order_id']} 状态={order['status']}，"
            f"金额={order['amount']}，是否超时={order['is_overdue']}。"
        )
        if order.get("is_overdue") and any(k in query for k in ("工单", "投诉", "开")):
            return {
                "order_id": order["order_id"],
                "agent_answer": answer + " 检测到超时且需要开单，转工单处理。",
                "route_path": ["order_agent", "to_ticket"],
                "meta": {**(state.get("meta") or {}), "continue_to_ticket": True},
            }
        return {
            "order_id": order["order_id"],
            "agent_answer": answer,
            "route_path": ["order_agent"],
            "needs_human": False,
        }

    orders = list_orders(user_id=state.get("user_id"), limit=5)
    if not orders:
        orders = list_orders(user_id=None, limit=5)
    if not orders:
        return {"agent_answer": "暂无订单数据。", "route_path": ["order_agent"]}
    lines = [
        f"- {o['order_id']}: status={o['status']}, overdue={o['is_overdue']}" for o in orders
    ]
    return {
        "agent_answer": "近期订单：\n" + "\n".join(lines),
        "route_path": ["order_agent"],
        "needs_human": False,
    }
