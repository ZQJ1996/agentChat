"""Data analysis agent with whitelist NL2SQL."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from app.config import get_settings
from app.graph.state import AgentState
from app.models import ChatMessage, get_model_provider
from app.observability import TraceContext, trace_node
from app.prompts import get_prompt
from app.tools.permission import require_permission

ALLOWED_TABLES = {
    "orders": {"order_id", "user_id", "status", "created_at", "shipped_at", "delivered_at", "amount"},
    "tickets": {"id", "title", "description", "priority", "category", "status", "assignee", "created_at", "updated_at"},
}

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|REPLACE|CREATE|TRUNCATE)\b",
    re.IGNORECASE,
)


def data_agent(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    role = state.get("role") or "user"
    require_permission(role, "data:read")
    trace = TraceContext(
        trace_id=state.get("trace_id") or "unknown",
        thread_id=state.get("thread_id") or "unknown",
    )

    with trace_node(trace, "data_agent", input_preview=query) as bucket:
        sql = _nl2sql(query)
        bucket["output_preview"] = sql
        try:
            rows = _run_readonly_sql(sql)
        except Exception as exc:  # noqa: BLE001
            return {
                "agent_answer": f"数据分析失败：{exc}。如需复杂查询请人工确认。",
                "needs_human": True,
                "route_path": ["data_agent", "sql_fail"],
                "data_result": [],
            }

    summary = _summarize_rows(sql, rows)
    return {
        "agent_answer": summary,
        "data_result": rows,
        "route_path": ["data_agent"],
        "needs_human": False,
        "meta": {**(state.get("meta") or {}), "sql": sql},
    }


def _nl2sql(query: str) -> str:
    # Template shortcuts for mock / reliability
    if any(k in query for k in ("工单", "优先级", "ticket")):
        return "SELECT priority, COUNT(*) AS cnt FROM tickets GROUP BY priority;"
    if any(k in query for k in ("订单", "物流", "销量", "amount", "status")):
        if "金额" in query or "amount" in query.lower():
            return "SELECT status, ROUND(SUM(amount),2) AS total_amount, COUNT(*) AS cnt FROM orders GROUP BY status;"
        return "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status;"

    provider = get_model_provider()
    messages = [
        ChatMessage(role="system", content=get_prompt("system_data")),
        ChatMessage(
            role="user",
            content=(
                "将问题转为一条只读 SQL。仅允许表 orders/tickets 及已知列。"
                f"问题：{query}"
            ),
        ),
    ]
    result = provider.chat(messages, temperature=0)
    sql = result.content.strip().strip("`")
    if "SELECT" not in sql.upper():
        return "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status;"
    # extract first statement
    sql = sql.split(";")[0].strip() + ";"
    return sql


def _run_readonly_sql(sql: str) -> list[dict[str, Any]]:
    if FORBIDDEN.search(sql):
        raise ValueError("仅允许 SELECT 只读查询")
    tables = set(re.findall(r"\bFROM\s+(\w+)", sql, flags=re.IGNORECASE))
    tables |= set(re.findall(r"\bJOIN\s+(\w+)", sql, flags=re.IGNORECASE))
    for t in tables:
        if t.lower() not in ALLOWED_TABLES:
            raise ValueError(f"表不在白名单: {t}")
    settings = get_settings()
    uri = f"file:{settings.sqlite_file}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]
    finally:
        conn.close()


def _summarize_rows(sql: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"查询已执行，无结果。SQL: {sql}"
    preview = rows[:10]
    lines = [f"- {row}" for row in preview]
    more = "" if len(rows) <= 10 else f"\n... 共 {len(rows)} 行"
    return f"查询结果如下（SQL: {sql}）：\n" + "\n".join(lines) + more
