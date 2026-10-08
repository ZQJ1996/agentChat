"""Pydantic API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str = "default"
    user_id: str = "u001"
    role: str = "user"
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    intent: str | None = None
    confidence: float | None = None
    route_path: list[str] = []
    evidence: list[dict[str, Any]] = []
    ticket_id: str | None = None
    needs_human: bool = False
    trace_id: str
    token_usage: int = 0
    data_result: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}


class IngestResponse(BaseModel):
    ingested_chunks: int


class HealthResponse(BaseModel):
    status: str
    model_provider: str
    knowledge_ready: bool
