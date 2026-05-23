"""SQLite checkpointer.

Persists one row per ``prompt_id`` with status, last error, attempt count,
and the latest result (if any). Resume = "read existing rows, only run the
ones still pending."

The schema is intentionally narrow so the SQLite file stays small and the
common operations are O(1) per row.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PromptStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    DLQ = "dlq"  # exhausted retries or hit a PermanentError


@dataclass
class PromptState:
    prompt_id: str
    status: PromptStatus
    attempts: int
    result: Any
    error: str | None
    updated_at: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    prompt_id   TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    error       TEXT,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS prompts_status_idx ON prompts(status);
"""


class SqliteCheckpointer:
    """Thread-safe SQLite-backed prompt state store.

    All methods are synchronous. Callers from async code keep state
    operations small (one row at a time) and run them on the event-loop
    thread. The bench shows this is not a bottleneck even at 1000 prompts.
    """

    def __init__(self, path: str | Path | None = ":memory:") -> None:
        self._path = ":memory:" if path is None or path == ":memory:" else str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so we can use it from asyncio tasks (single loop, no shared state).
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def enqueue(self, prompt_ids: list[str]) -> int:
        """Insert any missing prompt_ids as PENDING. Existing rows untouched.

        Returns the number of new rows inserted.
        """
        now = time.time()
        with self._lock:
            cursor = self._conn.executemany(
                "INSERT OR IGNORE INTO prompts (prompt_id, status, attempts, updated_at) VALUES (?, ?, 0, ?)",
                [(pid, PromptStatus.PENDING.value, now) for pid in prompt_ids],
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # status transitions
    # ------------------------------------------------------------------
    def mark_in_progress(self, prompt_id: str) -> None:
        self._update(prompt_id, status=PromptStatus.IN_PROGRESS)

    def mark_success(self, prompt_id: str, result: Any) -> None:
        self._update(
            prompt_id,
            status=PromptStatus.SUCCESS,
            result_json=json.dumps(result, ensure_ascii=False),
            error=None,
        )

    def mark_dlq(self, prompt_id: str, error: str) -> None:
        self._update(prompt_id, status=PromptStatus.DLQ, error=error)

    def increment_attempt(self, prompt_id: str, error: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE prompts SET attempts = attempts + 1, error = ?, status = ?, updated_at = ? WHERE prompt_id = ?",
                (error, PromptStatus.PENDING.value, time.time(), prompt_id),
            )

    # Hardcoded allowlist of columns that _update is permitted to write. Field
    # NAMES (unlike values) are not parameterized in SQL, so an attacker-shaped
    # kwarg name could otherwise be interpolated into the UPDATE statement. We
    # accept only the writable columns from _SCHEMA and reject anything else.
    _UPDATABLE_COLUMNS = frozenset({"status", "attempts", "result_json", "error", "updated_at"})

    def _update(self, prompt_id: str, **fields: Any) -> None:
        if not fields:
            return
        invalid = set(fields) - self._UPDATABLE_COLUMNS
        if invalid:
            raise ValueError(
                f"_update rejected non-allowlisted column name(s): {sorted(invalid)}. "
                f"Allowed: {sorted(self._UPDATABLE_COLUMNS)}"
            )
        fields["updated_at"] = time.time()
        columns = ", ".join(f"{k} = ?" for k in fields)
        values = [v.value if isinstance(v, PromptStatus) else v for v in fields.values()]
        with self._lock:
            self._conn.execute(
                f"UPDATE prompts SET {columns} WHERE prompt_id = ?",
                [*values, prompt_id],
            )

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def get(self, prompt_id: str) -> PromptState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT prompt_id, status, attempts, result_json, error, updated_at FROM prompts WHERE prompt_id = ?",
                (prompt_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def pending(self) -> list[PromptState]:
        return self._select_status((PromptStatus.PENDING.value, PromptStatus.IN_PROGRESS.value))

    def successes(self) -> list[PromptState]:
        return self._select_status((PromptStatus.SUCCESS.value,))

    def dead_letters(self) -> list[PromptState]:
        return self._select_status((PromptStatus.DLQ.value,))

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) FROM prompts GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}

    def all_pending_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT prompt_id FROM prompts WHERE status IN (?, ?)",
                (PromptStatus.PENDING.value, PromptStatus.IN_PROGRESS.value),
            ).fetchall()
        return [r[0] for r in rows]

    def reset_in_progress(self) -> int:
        """On resume, anything stuck in IN_PROGRESS should go back to PENDING."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE prompts SET status = ? WHERE status = ?",
                (PromptStatus.PENDING.value, PromptStatus.IN_PROGRESS.value),
            )
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    def _select_status(self, statuses: tuple[str, ...]) -> list[PromptState]:
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT prompt_id, status, attempts, result_json, error, updated_at FROM prompts WHERE status IN ({placeholders})",
                statuses,
            ).fetchall()
        return [self._row_to_state(r) for r in rows]

    @staticmethod
    def _row_to_state(row: tuple) -> PromptState:
        prompt_id, status, attempts, result_json, error, updated_at = row
        result = json.loads(result_json) if result_json else None
        return PromptState(
            prompt_id=prompt_id,
            status=PromptStatus(status),
            attempts=int(attempts),
            result=result,
            error=error,
            updated_at=float(updated_at),
        )
