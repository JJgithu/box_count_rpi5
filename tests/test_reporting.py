"""Tests for the CSV layout, the summary query, and the status panel."""

import csv
import io
import re
import time

from boxcounter.counter import CountEvent
from boxcounter.status import StatusDisplay
from boxcounter.storage import CountStore


def mk_event(ts, box_number, pieces, pack_seconds, track_id=1):
    return CountEvent(ts=ts, track_id=track_id, bbox=(10, 20, 100, 80),
                      area=8000.0, direction=1, pieces=pieces,
                      pack_seconds=pack_seconds, box_number=box_number)


def test_csv_has_one_row_per_box_with_pads_and_time(tmp_path):
    store = CountStore(str(tmp_path), use_sqlite=True, use_csv=True)
    t = time.time()
    store.record(mk_event(t, 1, 3, 11.4))
    store.record(mk_event(t + 20, 2, 2, 9.75))
    store.close()

    files = list(tmp_path.glob("events_*.csv"))
    assert len(files) == 1
    with open(files[0], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["box"] == "1"
    assert rows[0]["pads"] == "3"
    assert rows[0]["pack_seconds"] == "11.4"
    assert rows[1]["box"] == "2"
    assert rows[1]["pads"] == "2"
    assert rows[1]["pack_seconds"] == "9.75"
    # the four useful columns come first
    assert list(rows[0])[:4] == ["box", "timestamp", "pads", "pack_seconds"]


def test_csv_blank_when_packing_disabled(tmp_path):
    store = CountStore(str(tmp_path), use_sqlite=False, use_csv=True)
    store.record(mk_event(time.time(), 1, None, None))
    store.close()
    path = next(tmp_path.glob("events_*.csv"))
    rows = list(csv.DictReader(open(path, newline="")))
    assert rows[0]["pads"] == "" and rows[0]["pack_seconds"] == ""


def test_summary_totals_and_averages(tmp_path):
    store = CountStore(str(tmp_path), use_sqlite=True, use_csv=False)
    t = time.time()
    for i, (pads, secs) in enumerate([(3, 10.0), (2, 12.0), (4, 14.0)], start=1):
        store.record(mk_event(t + i, i, pads, secs))
    s = store.summary()
    assert s["total"] == 3
    assert s["pieces_total"] == 9
    assert abs(s["avg_pieces"] - 3.0) < 1e-6
    assert abs(s["avg_pack_seconds"] - 12.0) < 1e-6
    store.close()


def test_summary_survives_reset_and_reopen(tmp_path):
    store = CountStore(str(tmp_path), use_sqlite=True, use_csv=False)
    t = time.time()
    store.record(mk_event(t, 1, 5, 20.0))
    store.reset()
    store.record(mk_event(t + 10, 1, 3, 10.0))
    s = store.summary()
    assert s["total"] == 1, "reset must exclude earlier boxes"
    assert abs(s["avg_pieces"] - 3.0) < 1e-6
    store.close()

    reopened = CountStore(str(tmp_path), use_sqlite=True, use_csv=False)
    assert reopened.summary()["total"] == 1, "totals must survive a restart"
    assert reopened.total() == 1
    reopened.close()


def test_summary_empty_database(tmp_path):
    store = CountStore(str(tmp_path), use_sqlite=True, use_csv=False)
    s = store.summary()
    assert s["total"] == 0 and s["pieces_total"] == 0
    assert s["avg_pieces"] is None and s["avg_pack_seconds"] is None
    store.close()


def _render(stats, boxes=()):
    buf = io.StringIO()
    d = StatusDisplay(csv_path="data/events.csv", packing=True,
                      refresh_s=0, stream=buf)
    for b in boxes:
        d.add_box(*b)
    d.draw(stats, force=True)
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buf.getvalue())


def test_panel_shows_the_four_headline_numbers():
    out = _render(
        {"total": 42, "avg_pack_seconds": 11.4, "avg_pieces": 3.0,
         "pieces_total": 126, "rate_per_min": 5.2, "fps": 30.0},
        boxes=[(42, 3, 10.9, time.time())])
    assert "Boxes counted" in out and "42" in out
    assert "Average pack time" in out and "11.4 s" in out
    assert "Average pads per box" in out and "3.0" in out
    assert "LAST BOX" in out and "3 pads" in out and "10.9 s" in out


def test_panel_handles_no_data_yet():
    out = _render({"total": 0, "avg_pack_seconds": None, "avg_pieces": None,
                   "pieces_total": 0, "rate_per_min": 0.0, "fps": 0.0})
    assert "--" in out                      # averages render as placeholders
    assert "waiting for the first box" in out
    assert "none yet" in out


def test_panel_lines_stay_within_the_terminal_width():
    out = _render({"total": 123456, "avg_pack_seconds": 123.4, "avg_pieces": 12.3,
                   "pieces_total": 999999, "rate_per_min": 99.9, "fps": 30.0},
                  boxes=[(999, 12, 123.4, time.time())])
    assert max(len(line) for line in out.split("\n")) <= 74


def test_panel_without_packing_omits_pad_rows():
    buf = io.StringIO()
    d = StatusDisplay(csv_path="", packing=False, refresh_s=0, stream=buf)
    d.draw({"total": 7, "rate_per_min": 2.0, "fps": 30.0}, force=True)
    out = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buf.getvalue())
    assert "Boxes counted" in out
    assert "Average pads per box" not in out
    assert "LAST BOX" not in out
