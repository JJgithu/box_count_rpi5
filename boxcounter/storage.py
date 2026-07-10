"""Persistence: SQLite event log + daily CSV files.

SQLite runs in WAL mode with NORMAL synchronous writes, which is gentle on
SD cards at conveyor rates (a few events per second at most). The running
total survives restarts and can be reset from the web UI; resets are recorded
as a timestamp marker, not by deleting history.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .counter import CountEvent

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    iso TEXT NOT NULL,
    track_id INTEGER,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    area REAL,
    direction INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class CountStore:
    def __init__(self, data_dir: str, use_sqlite: bool = True, use_csv: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_sqlite = use_sqlite
        self.use_csv = use_csv
        self._lock = threading.Lock()   # web thread calls total()/reset()
        self._conn: Optional[sqlite3.Connection] = None
        if use_sqlite:
            self._conn = sqlite3.connect(self.data_dir / "boxcount.db",
                                         check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _last_reset_id(self) -> int:
        # Use the monotonic rowid, not wall-clock: an offline Pi with no RTC
        # can have its clock jump backwards after a power cut, which would make
        # a ts-based marker resurrect old events or hide new ones.
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='last_reset_id'").fetchone()
        return int(row[0]) if row else 0

    def record(self, ev: CountEvent) -> None:
        """Persist one event. Storage errors (disk full, SD wear) are logged
        and swallowed: losing a log row must never take down counting, GPIO
        output or the dashboard — none of which need the disk."""
        iso = datetime.fromtimestamp(ev.ts).isoformat(timespec="seconds")
        try:
            with self._lock:
                if self._conn is not None:
                    x, y, w, h = ev.bbox
                    self._conn.execute(
                        "INSERT INTO events (ts, iso, track_id, x, y, w, h, area, direction)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (ev.ts, iso, ev.track_id, x, y, w, h, ev.area, ev.direction))
                    self._conn.commit()
        except Exception:
            log.exception("SQLite write failed; continuing without persisting event")
        if self.use_csv:
            try:
                self._append_csv(ev, iso)
            except OSError:
                log.exception("CSV write failed; continuing")

    def _append_csv(self, ev: CountEvent, iso: str) -> None:
        day = datetime.fromtimestamp(ev.ts).strftime("%Y-%m-%d")
        path = self.data_dir / f"events_{day}.csv"
        new_file = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp", "track_id", "x", "y", "w", "h",
                                 "area", "direction"])
            x, y, w, h = ev.bbox
            writer.writerow([iso, ev.track_id, x, y, w, h, int(ev.area), ev.direction])

    def total(self) -> int:
        """Count of events since the last reset (0 if SQLite disabled)."""
        with self._lock:
            if self._conn is None:
                return 0
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE id > ?",
                (self._last_reset_id(),)).fetchone()
            return int(row[0])

    def reset(self) -> None:
        """Restart the running total; history stays in the database/CSVs."""
        with self._lock:
            if self._conn is not None:
                row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_reset_id', ?)",
                    (str(int(row[0])),))
                self._conn.commit()
        log.info("Count total reset")

    def recent(self, n: int = 20) -> List[dict]:
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                "SELECT iso, track_id, w, h, direction FROM events"
                " ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"time": r[0], "track_id": r[1], "w": r[2], "h": r[3],
                 "direction": r[4]} for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None
