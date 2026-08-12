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
    direction INTEGER,
    pieces INTEGER,
    pack_seconds REAL
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
            # migrate pre-packing databases in place
            for col, typ in (("pieces", "INTEGER"), ("pack_seconds", "REAL")):
                try:
                    self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass    # column already exists
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
                        "INSERT INTO events (ts, iso, track_id, x, y, w, h, area,"
                        " direction, pieces, pack_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (ev.ts, iso, ev.track_id, x, y, w, h, ev.area, ev.direction,
                         ev.pieces,
                         round(ev.pack_seconds, 2) if ev.pack_seconds is not None else None))
                    self._conn.commit()
        except Exception:
            log.exception("SQLite write failed; continuing without persisting event")
        if self.use_csv:
            try:
                self._append_csv(ev, iso)
            except OSError:
                log.exception("CSV write failed; continuing")

    # One row per counted box. The four columns that matter come first so the
    # file is readable as-is in a spreadsheet; the rest is detection detail
    # kept for diagnostics.
    _CSV_HEADER = ["box", "timestamp", "pads", "pack_seconds",
                   "track_id", "x", "y", "w", "h", "area", "direction"]

    def csv_path_for(self, ts: float) -> Path:
        """Path of the daily CSV a given timestamp belongs to."""
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        path = self.data_dir / f"events_{day}.csv"
        # A day file left by an older version has a different header; roll to
        # a sibling file instead of appending mismatched rows under it.
        if path.exists():
            try:
                with open(path, newline="") as f:
                    if next(csv.reader(f), None) != self._CSV_HEADER:
                        path = self.data_dir / f"events_{day}_v2.csv"
            except OSError:
                pass
        return path

    def _append_csv(self, ev: CountEvent, iso: str) -> None:
        path = self.csv_path_for(ev.ts)
        new_file = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self._CSV_HEADER)
            x, y, w, h = ev.bbox
            writer.writerow([
                "" if ev.box_number is None else ev.box_number,
                iso,
                "" if ev.pieces is None else ev.pieces,
                "" if ev.pack_seconds is None else round(ev.pack_seconds, 2),
                ev.track_id, x, y, w, h, int(ev.area), ev.direction])

    def summary(self) -> dict:
        """Totals and averages since the last reset, for the status display."""
        with self._lock:
            if self._conn is None:
                return {"total": 0, "avg_pieces": None, "avg_pack_seconds": None,
                        "pieces_total": 0}
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(pieces), 0), AVG(pieces),"
                " AVG(pack_seconds) FROM events WHERE id > ?",
                (self._last_reset_id(),)).fetchone()
        return {
            "total": int(row[0]),
            "pieces_total": int(row[1]),
            "avg_pieces": row[2],
            "avg_pack_seconds": row[3],
        }

    def total(self) -> int:
        """Count of events since the last reset (0 if SQLite disabled)."""
        with self._lock:
            if self._conn is None:
                return 0
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE id > ?",
                (self._last_reset_id(),)).fetchone()
            return int(row[0])

    def pieces_total(self) -> int:
        """Sum of pieces since the last reset (0 if SQLite disabled)."""
        with self._lock:
            if self._conn is None:
                return 0
            row = self._conn.execute(
                "SELECT COALESCE(SUM(pieces), 0) FROM events WHERE id > ?",
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
                "SELECT iso, track_id, w, h, direction, pieces, pack_seconds"
                " FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"time": r[0], "track_id": r[1], "w": r[2], "h": r[3],
                 "direction": r[4], "pieces": r[5], "pack_seconds": r[6]}
                for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None
