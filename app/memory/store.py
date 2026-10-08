"""Short-term checkpoint and long-term preference memory."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.tools.db import db_session, row_to_dict

_checkpointer: MemorySaver | None = None


def get_checkpointer() -> MemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def get_user_prefs(user_id: str) -> dict[str, Any]:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM user_prefs WHERE user_id = ?", (user_id,)).fetchone()
    data = row_to_dict(row)
    if not data:
        return {}
    try:
        return json.loads(data.get("prefs_json") or "{}")
    except json.JSONDecodeError:
        return {}


def upsert_user_prefs(user_id: str, prefs: dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()
    payload = json.dumps(prefs, ensure_ascii=False)
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO user_prefs(user_id, prefs_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET prefs_json=excluded.prefs_json, updated_at=excluded.updated_at
            """,
            (user_id, payload, now),
        )


def summarize_messages(messages: list[dict[str, str]], limit: int = 6) -> str:
    recent = messages[-limit:]
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')[:120]}" for m in recent]
    return "\n".join(lines)
