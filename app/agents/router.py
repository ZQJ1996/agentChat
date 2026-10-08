"""Intent router agent."""

from __future__ import annotations

import json
import re
from typing import Any

from app.graph.state import AgentState
from app.models import ChatMessage, get_model_provider
from app.observability import trace_node
from app.observability.tracing import TraceContext
from app.prompts import get_prompt
from app.security import check_user_input


def route_intent(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    trace = TraceContext(
        trace_id=state.get("trace_id") or "unknown",
        thread_id=state.get("thread_id") or "unknown",
    )
    sec = check_user_input(query)
    if not sec.safe:
        return {
            "query": sec.sanitized_text,
            "intent": "unclear",
            "confidence": 0.1,
            "needs_human": True,
            "route_path": ["intent_router", "security_block"],
            "agent_answer": "检测到潜在不安全指令，已拦截并转人工。",
            "meta": {"security_reasons": sec.reasons},
        }

    with trace_node(trace, "intent_router", input_preview=query) as bucket:
        provider = get_model_provider()
        summary = _session_summary(state.get("messages") or [])
        messages = [
            ChatMessage(role="system", content=get_prompt("system_router")),
            ChatMessage(
                role="user",
                content=f"会话摘要:\n{summary}\n\n当前问题:\n{sec.sanitized_text}",
            ),
        ]
        result = provider.chat(messages, temperature=0)
        bucket["tokens"] = result.total_tokens
        bucket["output_preview"] = result.content
        intent, confidence, reason = _parse_route(result.content, sec.sanitized_text)

    needs_human = confidence < 0.45 or intent == "unclear"
    return {
        "query": sec.sanitized_text,
        "intent": intent,
        "confidence": confidence,
        "needs_human": needs_human,
        "route_path": ["intent_router"],
        "token_usage": int(state.get("token_usage") or 0) + result.total_tokens,
        "meta": {**(state.get("meta") or {}), "route_reason": reason},
    }


def _session_summary(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "(空)"
    recent = messages[-4:]
    return "\n".join(f"{m.get('role')}: {m.get('content', '')[:80]}" for m in recent)


def _parse_route(content: str, query: str) -> tuple[str, float, str]:
    try:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            intent = str(data.get("intent") or "product_query")
            confidence = float(data.get("confidence") or 0.5)
            reason = str(data.get("reason") or "")
            return intent, confidence, reason
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # heuristic fallback
    from app.models.mock_provider import _route_intent

    return _route_intent(query)
