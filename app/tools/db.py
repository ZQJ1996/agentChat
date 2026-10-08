"""SQLite persistence for tickets, orders, dialog logs, preferences."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import get_settings


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    path = settings.sqlite_file
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session() -> Generator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                assignee TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                shipped_at TEXT,
                delivered_at TEXT,
                amount REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dialog_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_path TEXT,
                meta_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_prefs (
                user_id TEXT PRIMARY KEY,
                prefs_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                node TEXT NOT NULL,
                latency_ms REAL,
                tokens INTEGER,
                input_preview TEXT,
                output_preview TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        _seed_orders(conn)


def _seed_orders(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()
    if row and row["c"] > 0:
        return
    now = datetime.utcnow()
    samples = [
        ("ORD1001", "u001", "delivered", now - timedelta(days=20), now - timedelta(days=18), now - timedelta(days=15), 199.0),
        ("ORD1002", "u001", "shipped", now - timedelta(days=12), now - timedelta(days=10), None, 399.0),
        ("ORD1003", "u002", "pending", now - timedelta(days=9), None, None, 88.0),
        ("ORD1004", "u003", "shipped", now - timedelta(days=2), now - timedelta(days=1), None, 520.0),
        ("ORD1005", "u002", "cancelled", now - timedelta(days=30), None, None, 66.0),
    ]
    for order_id, user_id, status, created, shipped, delivered, amount in samples:
        conn.execute(
            """
            INSERT INTO orders(order_id, user_id, status, created_at, shipped_at, delivered_at, amount)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                user_id,
                status,
                created.isoformat(),
                shipped.isoformat() if shipped else None,
                delivered.isoformat() if delivered else None,
                amount,
            ),
        )


def next_ticket_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT COUNT(*) AS c FROM tickets").fetchone()
    n = int(row["c"]) + 1001 if row else 1001
    return f"T{n}"


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}
