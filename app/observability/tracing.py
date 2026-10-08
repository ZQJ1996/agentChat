"""Structured tracing and cost logging."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.tools.db import db_session

logger = logging.getLogger("agent.trace")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class TraceContext:
    trace_id: str
    thread_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0

    def add_tokens(self, n: int) -> None:
        self.total_tokens += max(0, n)


@contextmanager
def trace_node(
    ctx: TraceContext,
    node: str,
    *,
    input_preview: str = "",
) -> Generator[dict[str, Any]]:
    start = time.perf_counter()
    bucket: dict[str, Any] = {"output_preview": "", "tokens": 0}
    try:
        yield bucket
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        event = {
            "trace_id": ctx.trace_id,
            "node": node,
            "latency_ms": round(latency_ms, 2),
            "tokens": int(bucket.get("tokens") or 0),
            "input_preview": (input_preview or "")[:300],
            "output_preview": str(bucket.get("output_preview") or "")[:300],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        ctx.events.append(event)
        ctx.add_tokens(int(event["tokens"]))
        logger.info(json.dumps(event, ensure_ascii=False))
        try:
            with db_session() as conn:
                conn.execute(
                    """
                    INSERT INTO trace_events(trace_id, node, latency_ms, tokens, input_preview, output_preview, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["trace_id"],
                        event["node"],
                        event["latency_ms"],
                        event["tokens"],
                        event["input_preview"],
                        event["output_preview"],
                        event["created_at"],
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to persist trace: %s", exc)


def log_dialog(
    *,
    trace_id: str,
    thread_id: str,
    role: str,
    content: str,
    user_id: str | None = None,
    agent_path: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO dialog_logs(trace_id, thread_id, user_id, role, content, agent_path, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                thread_id,
                user_id,
                role,
                content,
                ",".join(agent_path or []),
                json.dumps(meta or {}, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
