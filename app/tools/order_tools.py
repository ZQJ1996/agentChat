"""Order query tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.tools.db import db_session, row_to_dict


def query_order(order_id: str) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    data = row_to_dict(row)
    if not data:
        return None
    data["overdue_days"] = _overdue_days(data)
    data["is_overdue"] = data["overdue_days"] is not None and data["overdue_days"] > 7
    return data


def list_orders(user_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    with db_session() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    result = []
    for row in rows:
        data = row_to_dict(row) or {}
        data["overdue_days"] = _overdue_days(data)
        data["is_overdue"] = data["overdue_days"] is not None and data["overdue_days"] > 7
        result.append(data)
    return result


def _overdue_days(order: dict[str, Any]) -> int | None:
    if order.get("status") in {"delivered", "cancelled"}:
        return None
    created = order.get("created_at")
    if not created:
        return None
    try:
        created_dt = datetime.fromisoformat(created)
    except ValueError:
        return None
    return (datetime.utcnow() - created_dt).days
