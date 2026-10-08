"""FastAPI routers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import  APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from app.config import get_settings
from app.eval import run_evaluation, write_report
from app.graph import run_agent
from app.observability import log_dialog, new_trace_id
from app.rag import get_knowledge_store
from app.security import mask_sensitive
from app.tools.ticket_tools import query_ticket

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    store = get_knowledge_store()
    ready = len(store._corpus) > 0  # noqa: SLF001
    return HealthResponse(
        status="ok",
        model_provider=settings.model_provider,
        knowledge_ready=ready,
    )


@router.post("/knowledge/ingest", response_model=IngestResponse)
def ingest_knowledge() -> IngestResponse:
    store = get_knowledge_store()
    n = store.ingest_directory()
    return IngestResponse(ingested_chunks=n)


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> dict[str, Any]:
    ticket = query_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="ticket not found")
    return ticket


@router.post("/eval/run")
def eval_run() -> dict[str, Any]:
    report = run_evaluation()
    path = write_report(report)
    return {
        "report_path": str(path),
        "total": report.total,
        "intent_accuracy": report.intent_accuracy,
        "answer_accuracy": report.answer_accuracy,
        "refuse_rational_rate": report.refuse_rational_rate,
        "ticket_completion_rate": report.ticket_completion_rate,
        "avg_recall_at_k": report.avg_recall_at_k,
    }


def _invoke_chat(req: ChatRequest) -> dict[str, Any]:
    trace_id = new_trace_id()
    safe_message = mask_sensitive(req.message)
    log_dialog(
        trace_id=trace_id,
        thread_id=req.thread_id,
        role="user",
        content=safe_message,
        user_id=req.user_id,
    )
    result = run_agent(
        query=safe_message,
        thread_id=req.thread_id,
        trace_id=trace_id,
        user_id=req.user_id,
        role=req.role,
    )
    answer = mask_sensitive(result.get("final_answer") or result.get("agent_answer") or "")
    log_dialog(
        trace_id=trace_id,
        thread_id=req.thread_id,
        role="assistant",
        content=answer,
        user_id=req.user_id,
        agent_path=list(result.get("route_path") or []),
        meta={
            "intent": result.get("intent"),
            "ticket_id": result.get("ticket_id"),
            "needs_human": result.get("needs_human"),
        },
    )
    result["final_answer"] = answer
    result["trace_id"] = trace_id
    return result


@router.post("/chat")
async def chat(req: ChatRequest):
    if req.stream:
        return EventSourceResponse(_sse_chat(req))

    result = await asyncio.to_thread(_invoke_chat, req)
    return ChatResponse(
        answer=result.get("final_answer") or "",
        intent=result.get("intent"),
        confidence=result.get("confidence"),
        route_path=list(result.get("route_path") or []),
        evidence=list(result.get("evidence") or []),
        ticket_id=result.get("ticket_id"),
        needs_human=bool(result.get("needs_human")),
        trace_id=result.get("trace_id") or "",
        token_usage=int(result.get("token_usage") or 0),
        data_result=list(result.get("data_result") or []),
        meta=dict(result.get("meta") or {}),
    )


async def _sse_chat(req: ChatRequest) -> AsyncIterator[dict[str, str]]:
    result = await asyncio.to_thread(_invoke_chat, req)
    answer = result.get("final_answer") or ""
    meta = {
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
        "route_path": list(result.get("route_path") or []),
        "ticket_id": result.get("ticket_id"),
        "needs_human": bool(result.get("needs_human")),
        "trace_id": result.get("trace_id"),
        "token_usage": int(result.get("token_usage") or 0),
        "evidence": list(result.get("evidence") or []),
        "data_result": list(result.get("data_result") or []),
    }
    yield {"event": "meta", "data": json.dumps(meta, ensure_ascii=False)}
    # chunk answer for SSE UX
    step = 24
    for i in range(0, len(answer), step):
        yield {"event": "token", "data": answer[i : i + step]}
        await asyncio.sleep(0.01)
    yield {"event": "done", "data": "[DONE]"}
