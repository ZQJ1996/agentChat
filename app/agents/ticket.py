"""Ticket agent with tool calling and retry."""

from __future__ import annotations

import re
from typing import Any

from app.config import get_settings
from app.graph.state import AgentState
from app.models import ChatMessage, get_model_provider
from app.observability import TraceContext, trace_node
from app.prompts import get_prompt
from app.tools.order_tools import query_order
from app.tools.permission import require_permission
from app.tools.ticket_tools import create_ticket, query_ticket, update_ticket_priority, update_ticket_status


def ticket_agent(state: AgentState) -> AgentState:
    settings = get_settings()
    query = state.get("query") or ""
    role = state.get("role") or "user"
    retries = int(state.get("tool_retries") or 0)
    trace = TraceContext(
        trace_id=state.get("trace_id") or "unknown",
        thread_id=state.get("thread_id") or "unknown",
    )

    # Compound path: order overdue -> create complaint ticket
    order_id = state.get("order_id") or _extract_order_id(query)
    order_info = None
    if order_id:
        order_info = query_order(order_id)

    with trace_node(trace, "ticket_agent", input_preview=query) as bucket:
        try:
            if order_info and order_info.get("is_overdue"):
                require_permission(role, "ticket:write")
                ticket = create_ticket(
                    title=f"订单超时投诉 {order_id}",
                    desc=f"用户反馈订单 {order_id} 超时未送达。订单状态={order_info.get('status')}，已超时 {order_info.get('overdue_days')} 天。原始诉求：{query}",
                    priority="high",
                    category="complaint",
                )
                answer = (
                    f"已查询订单 {order_id}：状态={order_info.get('status')}，"
                    f"已超时 {order_info.get('overdue_days')} 天。"
                    f"已为您创建投诉工单 #{ticket['id']}（优先级：{ticket['priority']}）。"
                )
                bucket["output_preview"] = answer
                return {
                    "order_id": order_id,
                    "ticket_id": ticket["id"],
                    "agent_answer": answer,
                    "route_path": ["ticket_agent", "create_ticket"],
                    "tool_retries": 0,
                    "needs_human": False,
                }

            provider = get_model_provider()
            messages = [
                ChatMessage(role="system", content=get_prompt("system_ticket")),
                ChatMessage(role="user", content=query),
            ]
            result = provider.chat(messages)
            bucket["tokens"] = result.total_tokens
            plan = result.content
            answer, ticket_id = _execute_plan(plan, query, role, order_info)
            bucket["output_preview"] = answer
            return {
                "order_id": order_id,
                "ticket_id": ticket_id,
                "agent_answer": answer,
                "route_path": ["ticket_agent"],
                "tool_retries": 0,
                "token_usage": int(state.get("token_usage") or 0) + result.total_tokens,
                "needs_human": False,
            }
        except Exception as exc:  # noqa: BLE001
            if retries < settings.max_tool_retries:
                return {
                    "tool_retries": retries + 1,
                    "error": str(exc),
                    "route_path": ["ticket_agent", "retry"],
                    "agent_answer": f"工具调用失败，正在重试：{exc}",
                }
            return {
                "tool_retries": retries,
                "error": str(exc),
                "needs_human": True,
                "route_path": ["ticket_agent", "fail"],
                "agent_answer": f"工单处理失败（{exc}），已转人工。",
            }


def _extract_order_id(text: str) -> str | None:
    m = re.search(r"(ORD\d{3,}|订单[:：\s]*([A-Za-z0-9-]+))", text, re.IGNORECASE)
    if not m:
        # also accept bare ORD####
        m2 = re.search(r"ORD\d{3,}", text, re.IGNORECASE)
        return m2.group(0).upper() if m2 else None
    if m.group(0).upper().startswith("ORD"):
        m_ord = re.search(r"ORD\d{3,}", m.group(0), re.IGNORECASE)
        return m_ord.group(0).upper() if m_ord else None
    return (m.group(2) or m.group(1) or "").upper() or None


def _execute_plan(
    plan: str,
    query: str,
    role: str,
    order_info: dict[str, Any] | None,
) -> tuple[str, str | None]:
    # Prefer explicit TOOL_CALL directives from mock/LLM
    create_m = re.search(
        r'TOOL_CALL\s+create_ticket.*?title="([^"]*)".*?desc="([^"]*)".*?priority="([^"]*)".*?category="([^"]*)"',
        plan,
        re.IGNORECASE | re.DOTALL,
    )
    query_m = re.search(r'TOOL_CALL\s+query_ticket.*?ticket_id="([^"]*)"', plan, re.IGNORECASE)
    update_m = re.search(
        r'TOOL_CALL\s+update_ticket_status.*?ticket_id="([^"]*)".*?status="([^"]*)"',
        plan,
        re.IGNORECASE,
    )

    if create_m or any(k in query for k in ("创建", "开", "投诉", "升级")):
        require_permission(role, "ticket:write")
        title = create_m.group(1) if create_m else "用户请求工单"
        desc = create_m.group(2) if create_m else query
        priority = create_m.group(3) if create_m else ("high" if "高" in query or "投诉" in query else "medium")
        category = create_m.group(4) if create_m else "general"
        if "超时" in query or (order_info and order_info.get("is_overdue")):
            priority = "high"
            category = "complaint"
        ticket = create_ticket(title=title, desc=desc, priority=priority, category=category)
        if "高优先" in query or "升级" in query:
            ticket = update_ticket_priority(ticket["id"], "high")
        return (
            f"已创建工单 #{ticket['id']}，优先级={ticket['priority']}，状态={ticket['status']}。",
            ticket["id"],
        )

    if query_m or re.search(r"T\d{3,}", query):
        require_permission(role, "ticket:read")
        tid_m = re.search(r"T\d{3,}", query)
        tid = query_m.group(1) if query_m else (tid_m.group(0) if tid_m else "")
        ticket = query_ticket(tid)
        if not ticket:
            return f"未找到工单 {tid}。", None
        return (
            f"工单 #{ticket['id']}：状态={ticket['status']}，优先级={ticket['priority']}，标题={ticket['title']}。",
            ticket["id"],
        )

    if update_m:
        require_permission(role, "ticket:update")
        ticket = update_ticket_status(update_m.group(1), update_m.group(2))
        return f"已更新工单 #{ticket['id']} 状态为 {ticket['status']}。", ticket["id"]

    # default create for ticket intent
    require_permission(role, "ticket:write")
    ticket = create_ticket(title="用户工单请求", desc=query, priority="medium", category="general")
    return f"已为您创建工单 #{ticket['id']}（优先级={ticket['priority']}）。", ticket["id"]
