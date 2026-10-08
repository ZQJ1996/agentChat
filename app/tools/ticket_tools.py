"""Ticket CRUD tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.tools.db import db_session, next_ticket_id, row_to_dict


ALLOWED_PRIORITIES = {"low", "medium", "high", "urgent"}
ALLOWED_STATUS = {"open", "in_progress", "resolved", "closed", "escalated"}


def create_ticket(
    title: str,
    desc: str,
    priority: str = "medium",
    category: str = "general",
    assignee: str | None = None,
) -> dict[str, Any]:
    priority = (priority or "medium").lower()
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError(f"invalid priority: {priority}")
    if not title.strip() or not desc.strip():
        raise ValueError("title and desc are required")
    now = datetime.utcnow().isoformat()
    with db_session() as conn:
        ticket_id = next_ticket_id(conn)
        conn.execute(
            """
            INSERT INTO tickets(id, title, description, priority, category, status, assignee, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, title.strip(), desc.strip(), priority, category, "open", assignee, now, now),
        )
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return row_to_dict(row) or {}


def query_ticket(ticket_id: str) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return row_to_dict(row)


def update_ticket_status(ticket_id: str, status: str) -> dict[str, Any]:
    status = status.lower()
    if status not in ALLOWED_STATUS:
        raise ValueError(f"invalid status: {status}")
    now = datetime.utcnow().isoformat()
    with db_session() as conn:
        cur = conn.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, ticket_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"ticket not found: {ticket_id}")
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return row_to_dict(row) or {}


def update_ticket_priority(ticket_id: str, priority: str) -> dict[str, Any]:
    priority = priority.lower()
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError(f"invalid priority: {priority}")
    now = datetime.utcnow().isoformat()
    with db_session() as conn:
        cur = conn.execute(
            "UPDATE tickets SET priority = ?, updated_at = ? WHERE id = ?",
            (priority, now, ticket_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"ticket not found: {ticket_id}")
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return row_to_dict(row) or {}
