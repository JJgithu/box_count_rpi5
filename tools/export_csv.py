"""Export counted boxes to a single CSV — one row per box.

The counter already writes a CSV per day (data/events_YYYY-MM-DD.csv). This
tool pulls from the database instead, so you can get any date range as one
file, e.g. a whole week or a single shift, with a per-box summary.

    python3 tools/export_csv.py                       # everything -> stdout
    python3 tools/export_csv.py -o report.csv         # everything -> file
    python3 tools/export_csv.py --today -o today.csv
    python3 tools/export_csv.py --since 2026-08-01 --until 2026-08-08 -o week.csv
    python3 tools/export_csv.py --summary             # totals and averages only
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter.config import load_config             # noqa: E402

HEADER = ["box", "timestamp", "date", "time", "pads", "pack_seconds"]


def _day_bounds(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d").timestamp()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--db", help="path to boxcount.db (default: from config)")
    ap.add_argument("--out", "-o", help="output file (default: stdout)")
    ap.add_argument("--since", help="first day to include, YYYY-MM-DD")
    ap.add_argument("--until", help="last day to include, YYYY-MM-DD (inclusive)")
    ap.add_argument("--today", action="store_true", help="today only")
    ap.add_argument("--summary", action="store_true",
                    help="print totals and averages instead of rows")
    args = ap.parse_args()

    if args.db:
        db_path = Path(args.db)
    else:
        cfg = load_config(args.config)
        db_path = Path(cfg.output.data_dir) / "boxcount.db"
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    where, params = [], []
    if args.today:
        start = datetime.now().replace(hour=0, minute=0, second=0,
                                       microsecond=0).timestamp()
        where.append("ts >= ?")
        params.append(start)
    if args.since:
        where.append("ts >= ?")
        params.append(_day_bounds(args.since))
    if args.until:
        where.append("ts < ?")
        params.append((datetime.strptime(args.until, "%Y-%m-%d")
                       + timedelta(days=1)).timestamp())
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT ts, iso, pieces, pack_seconds FROM events{clause} ORDER BY id",
        params).fetchall()

    if args.summary:
        n = len(rows)
        pads = [r[2] for r in rows if r[2] is not None]
        secs = [r[3] for r in rows if r[3] is not None]
        print(f"Boxes counted        : {n}")
        print(f"Pads counted (total) : {sum(pads) if pads else 0}")
        print(f"Average pads per box : "
              f"{sum(pads)/len(pads):.1f}" if pads else "Average pads per box : --")
        print(f"Average pack time    : "
              f"{sum(secs)/len(secs):.1f} s" if secs else "Average pack time    : --")
        if secs:
            print(f"Fastest / slowest    : {min(secs):.1f} s / {max(secs):.1f} s")
        if rows:
            print(f"First box            : {rows[0][1]}")
            print(f"Last box             : {rows[-1][1]}")
        return 0

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer = csv.writer(out)
        writer.writerow(HEADER)
        for i, (ts, iso, pieces, pack_s) in enumerate(rows, start=1):
            date, _, clock = iso.partition("T")
            writer.writerow([i, iso, date, clock,
                             "" if pieces is None else pieces,
                             "" if pack_s is None else round(pack_s, 2)])
    finally:
        if args.out:
            out.close()
            print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
