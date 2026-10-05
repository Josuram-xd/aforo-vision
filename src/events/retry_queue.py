"""Persistent FIFO queue of events waiting to reach the backend (SQLite, stdlib).

Every event is written here before any send attempt, so a crash or a network drop never loses one.
Events leave in the order they were queued. The backend is idempotent by eventId, so a retry after an
ambiguous failure (sent but the answer was lost) does not double-count.

The file holds event JSON, including personName for identified students: it lives under data/,
which is git-ignored, and is deleted with the rest of the pilot data.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path


class RetryQueue:
    def __init__(self, path: str | Path):
        path = str(path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the pipeline thread (push) and the sender thread (peek/ack)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS queue ("
                " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
                " event_id TEXT NOT NULL UNIQUE,"
                " payload TEXT NOT NULL,"
                " queued_at REAL NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS rejected ("
                " event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, reason TEXT, rejected_at REAL NOT NULL)"
            )

    def push(self, event: dict) -> bool:
        """Queue an event at the back. Returns False if an event with that eventId is already waiting."""
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO queue (event_id, payload, queued_at) VALUES (?, ?, ?)",
                (event["eventId"], json.dumps(event, ensure_ascii=False), time.time()),
            )
            return cursor.rowcount == 1

    def peek(self) -> dict | None:
        """The oldest waiting event, or None. It stays queued until `ack` or `reject`."""
        with self._lock:
            row = self._db.execute("SELECT payload FROM queue ORDER BY seq LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def ack(self, event_id: str) -> None:
        """The backend has the event: remove it from the queue."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM queue WHERE event_id = ?", (event_id,))

    def reject(self, event_id: str, reason: str = "") -> None:
        """The backend refuses the event for good: move it aside so it stops blocking the queue."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO rejected (event_id, payload, reason, rejected_at)"
                " SELECT event_id, payload, ?, ? FROM queue WHERE event_id = ?",
                (reason, time.time(), event_id),
            )
            self._db.execute("DELETE FROM queue WHERE event_id = ?", (event_id,))

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM queue").fetchone()[0]

    def rejected_count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM rejected").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()
