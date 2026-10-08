"""Customer service RAG agent with refuse-to-answer."""

from __future__ import annotations

import json
import re

from app.config import get_settings
from app.graph.state import AgentState
from app.models import ChatMessage, get_model_provider
from app.observability import TraceContext, trace_node
from app.prompts import get_prompt
from app.rag import get_knowledge_store


def customer_service_agent(state: AgentState) -> AgentState:
    settings = get_settings()
    query = state.get("query") or ""
    store = get_knowledge_store()
    evidence = store.hybrid_search(query)
    trace = TraceContext(
        trace_id=state.get("trace_id") or "unknown",
        thread_id=state.get("thread_id") or "unknown",
    )

    evidence_dicts = [
        {
            "rank": e.rank,
            "chunk_id": e.chunk_id,
            "text": e.text,
            "source": e.source,
            "title": e.title,
            "section": e.section,
            "score": e.score,
            "citation": e.citation(),
        }
        for e in evidence
    ]

    max_score = max((e.score for e in evidence), default=0.0)
    # also require minimal lexical overlap so unrelated top chunks don't pass
    overlap_ok = _query_evidence_overlap(query, evidence) >= 0.12
    if not evidence or max_score < settings.evidence_min_score or not overlap_ok:
        answer = "当前知识库证据不足，无法可靠回答该问题。已为您转接人工客服。"
        return {
            "evidence": evidence_dicts,
            "agent_answer": answer,
            "needs_human": True,
            "route_path": ["customer_service", "refuse"],
        }

    evidence_block = "\n\n".join(
        f"[{e.rank}] ({e.title}/{e.section}) score={e.score:.3f}\n{e.text}" for e in evidence
    )

    with trace_node(trace, "customer_service", input_preview=query) as bucket:
        provider = get_model_provider()
        messages = [
            ChatMessage(role="system", content=get_prompt("system_customer_service")),
            ChatMessage(
                role="user",
                content=f"用户问题：{query}\n\n证据：\n{evidence_block}",
            ),
        ]
        result = provider.chat(messages)
        bucket["tokens"] = result.total_tokens
        answer = result.content
        bucket["output_preview"] = answer

    verified, reason = _verify_answer(answer, evidence_block)
    if not verified:
        return {
            "evidence": evidence_dicts,
            "agent_answer": "回答未通过证据校验，为避免幻觉已转人工处理。",
            "needs_human": True,
            "route_path": ["customer_service", "verify_fail"],
            "token_usage": int(state.get("token_usage") or 0) + result.total_tokens,
            "meta": {**(state.get("meta") or {}), "verify_reason": reason},
        }

    citations = "\n".join(str(e.get("citation") or "") for e in evidence_dicts)
    if "[1]" not in answer and evidence_dicts:
        answer = f"{answer}\n\n参考来源：\n{citations}"

    return {
        "evidence": evidence_dicts,
        "agent_answer": answer,
        "needs_human": False,
        "route_path": ["customer_service"],
        "token_usage": int(state.get("token_usage") or 0) + result.total_tokens,
    }


def _query_evidence_overlap(query: str, evidence) -> float:
    from app.rag.store import _tokenize

    q = set(_tokenize(query))
    if not q or not evidence:
        return 0.0
    best = 0.0
    for e in evidence:
        t = set(_tokenize(e.text))
        best = max(best, len(q & t) / len(q))
    return best


def _verify_answer(answer: str, evidence: str) -> tuple[bool, str]:
    refuse_markers = ("无法确定", "证据不足", "转人工", "无法可靠")
    if any(m in answer for m in refuse_markers):
        return True, "explicit refuse"
    provider = get_model_provider()
    messages = [
        ChatMessage(role="system", content=get_prompt("system_verify")),
        ChatMessage(role="user", content=f"证据:\n{evidence}\n\n回答:\n{answer}"),
    ]
    result = provider.chat(messages, temperature=0)
    try:
        m = re.search(r"\{.*\}", result.content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return bool(data.get("ok")), str(data.get("reason") or "")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # lexical overlap heuristic
    ans_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", answer))
    ev_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", evidence))
    if not ans_tokens:
        return False, "empty answer"
    overlap = len(ans_tokens & ev_tokens) / len(ans_tokens)
    return overlap >= 0.15, f"overlap={overlap:.2f}"
