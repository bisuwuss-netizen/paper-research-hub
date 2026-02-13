from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple


def _now() -> int:
    return int(time.time())


def enqueue_paper(conn, paper_id: int, delay_seconds: int = 0) -> None:
    next_run = _now() + max(0, delay_seconds)
    conn.execute(
        """
        INSERT INTO sync_queue (paper_id, status, attempts, next_run_at)
        VALUES (?, 'queued', 0, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
          next_run_at = excluded.next_run_at,
          status = CASE
            WHEN sync_queue.status = 'running' THEN sync_queue.status
            ELSE 'queued'
          END
        """,
        (paper_id, next_run),
    )


def enqueue_all(conn) -> int:
    rows = conn.execute("SELECT id FROM papers").fetchall()
    count = 0
    for r in rows:
        enqueue_paper(conn, r["id"], delay_seconds=0)
        count += 1
    return count


def fetch_due(conn, limit: int) -> List[int]:
    now = _now()
    rows = conn.execute(
        """
        SELECT paper_id FROM sync_queue
        WHERE (status IN ('queued','idle','failed'))
          AND (next_run_at IS NULL OR next_run_at <= ?)
        ORDER BY next_run_at ASC
        LIMIT ?
        """,
        (now, limit),
    ).fetchall()
    return [r["paper_id"] for r in rows]


def mark_running(conn, paper_id: int) -> None:
    conn.execute(
        "UPDATE sync_queue SET status='running', last_run_at=? WHERE paper_id = ?",
        (_now(), paper_id),
    )


def mark_success(conn, paper_id: int) -> None:
    interval = int(os.getenv("SYNC_RESYNC_INTERVAL", "86400"))
    conn.execute(
        """
        UPDATE sync_queue
        SET status='idle', attempts=0, last_error=NULL, next_run_at=?
        WHERE paper_id = ?
        """,
        (_now() + interval, paper_id),
    )


def mark_failure(conn, paper_id: int, error: str) -> None:
    base = int(os.getenv("SYNC_RETRY_BASE", "600"))
    row = conn.execute(
        "SELECT attempts FROM sync_queue WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    attempts = (row["attempts"] if row else 0) + 1
    backoff = min(base * (2 ** min(attempts, 5)), 86400)
    conn.execute(
        """
        UPDATE sync_queue
        SET status='failed', attempts=?, last_error=?, next_run_at=?
        WHERE paper_id = ?
        """,
        (attempts, error[:500], _now() + backoff, paper_id),
    )


def queue_stats(conn) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) as c FROM sync_queue GROUP BY status"
    ).fetchall()
    out = {r["status"]: r["c"] for r in rows}
    return out
