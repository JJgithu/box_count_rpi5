"""--reset / --total: resetting the count from the command line.

There is no browser on a factory-floor Pi, so the terminal must be able to
do this. History must survive: a reset restarts the running total only.
"""

import shutil
import sys
import time
from pathlib import Path

import pytest

from boxcounter.__main__ import main
from boxcounter.counter import CountEvent
from boxcounter.storage import CountStore

REAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture()
def cfg_with_data(tmp_path):
    """A config pointing at a data dir that already holds three boxes."""
    from boxcounter.configedit import set_values
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(REAL_CONFIG, cfg_path)
    data_dir = tmp_path / "data"
    text, _ = set_values(cfg_path.read_text(),
                         {"output.data_dir": str(data_dir)})
    cfg_path.write_text(text)

    store = CountStore(str(data_dir), use_sqlite=True, use_csv=True)
    now = time.time()
    for i, (pads, secs) in enumerate([(3, 10.0), (2, 12.0), (4, 14.0)], start=1):
        store.record(CountEvent(ts=now + i, track_id=i, bbox=(0, 0, 10, 10),
                                area=100.0, direction=1, pieces=pads,
                                pack_seconds=secs, box_number=i))
    store.close()
    return cfg_path, data_dir


def test_total_prints_the_counts(cfg_with_data, capsys):
    cfg_path, _ = cfg_with_data
    assert main(["-c", str(cfg_path), "--total"]) == 0
    out = capsys.readouterr().out
    assert "Boxes counted        : 3" in out
    assert "Pads counted (total) : 9" in out
    assert "3.0" in out          # average pads
    assert "12.0 s" in out       # average pack time


def test_reset_zeroes_the_running_total(cfg_with_data, capsys):
    cfg_path, _ = cfg_with_data
    assert main(["-c", str(cfg_path), "--reset"]) == 0
    assert "3 -> 0" in capsys.readouterr().out

    assert main(["-c", str(cfg_path), "--total"]) == 0
    assert "Boxes counted        : 0" in capsys.readouterr().out


def test_reset_keeps_the_history(cfg_with_data):
    import sqlite3
    cfg_path, data_dir = cfg_with_data
    main(["-c", str(cfg_path), "--reset"])

    rows = sqlite3.connect(data_dir / "boxcount.db").execute(
        "SELECT COUNT(*) FROM events").fetchone()[0]
    assert rows == 3, "reset must not delete history"
    assert list(data_dir.glob("events_*.csv")), "CSV files must survive"


def test_counting_resumes_from_zero_after_reset(cfg_with_data):
    cfg_path, data_dir = cfg_with_data
    main(["-c", str(cfg_path), "--reset"])

    store = CountStore(str(data_dir), use_sqlite=True, use_csv=False)
    try:
        assert store.total() == 0
        store.record(CountEvent(ts=time.time(), track_id=9, bbox=(0, 0, 10, 10),
                                area=100.0, direction=1, pieces=5,
                                pack_seconds=8.0, box_number=1))
        assert store.total() == 1, "new boxes count up from zero"
        assert store.summary()["pieces_total"] == 5
    finally:
        store.close()


def test_reset_twice_is_harmless(cfg_with_data, capsys):
    cfg_path, _ = cfg_with_data
    main(["-c", str(cfg_path), "--reset"])
    capsys.readouterr()
    assert main(["-c", str(cfg_path), "--reset"]) == 0
    assert "0 -> 0" in capsys.readouterr().out


def test_reset_works_on_an_empty_database(tmp_path, capsys):
    from boxcounter.configedit import set_values
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(REAL_CONFIG, cfg_path)
    text, _ = set_values(cfg_path.read_text(),
                         {"output.data_dir": str(tmp_path / "fresh")})
    cfg_path.write_text(text)
    assert main(["-c", str(cfg_path), "--reset"]) == 0
    assert "0 -> 0" in capsys.readouterr().out


def test_reset_does_not_need_the_camera(cfg_with_data, monkeypatch):
    """--reset must work with no camera attached (picamera2 absent)."""
    import boxcounter.__main__ as m
    monkeypatch.setattr(m, "_check_optional_deps",
                        lambda cfg, args: pytest.fail(
                            "--reset must not require camera dependencies"))
    cfg_path, _ = cfg_with_data
    assert main(["-c", str(cfg_path), "--reset"]) == 0
