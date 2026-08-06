"""End-to-end packing-station test: synthetic video -> exact piece counts.

Also guards the counting side: the packer's arm sweeping across the frame
must never inflate the box count.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_packing_video import generate_packing_video     # noqa: E402

from boxcounter.config import AppConfig                   # noqa: E402
from boxcounter.pipeline import Pipeline                  # noqa: E402


def packing_config(video_path: str, data_dir: str) -> AppConfig:
    cfg = AppConfig()
    cfg.camera.source = "video"
    cfg.camera.video_path = video_path
    cfg.processing.roi = [0.0, 0.0, 1.0, 1.0]
    cfg.processing.warmup_frames = 60
    cfg.counting.axis = "y"
    cfg.counting.line_position = 0.78
    cfg.counting.direction = "positive"
    cfg.packing.enabled = True
    cfg.packing.zone = [0.0, 0.05, 1.0, 0.55]
    cfg.tracking.max_disappeared = 30     # survive arm-merge occlusions
    cfg.output.data_dir = data_dir
    cfg.output.csv = False
    cfg.web.enabled = False
    cfg.gpio.enabled = False
    return cfg


def test_pieces_boxes_and_pack_time(tmp_path):
    video = str(tmp_path / "packing.mp4")
    truth = generate_packing_video(video, pieces_per_box=(3, 2), seed=7)

    data_dir = str(tmp_path / "data")
    summary = Pipeline(packing_config(video, data_dir)).run()

    assert summary["session"] == truth["boxes"], (
        f"box count wrong: {summary['session']} != {truth['boxes']} "
        "(an arm may have been counted as a box)")

    rows = sqlite3.connect(Path(data_dir) / "boxcount.db").execute(
        "SELECT pieces, pack_seconds FROM events ORDER BY id").fetchall()
    assert len(rows) == truth["boxes"]
    assert [r[0] for r in rows] == truth["pieces"], (
        f"pieces per box wrong: {[r[0] for r in rows]} != {truth['pieces']}")
    for (pieces, pack_s), expect_s in zip(rows, truth["pack_seconds"]):
        assert pack_s is not None
        assert abs(pack_s - expect_s) < 2.0, (
            f"pack time {pack_s:.1f}s too far from truth {expect_s:.1f}s")


def test_more_pieces_and_single_box(tmp_path):
    video = str(tmp_path / "packing5.mp4")
    truth = generate_packing_video(video, pieces_per_box=(5,), seed=21)
    data_dir = str(tmp_path / "data")
    summary = Pipeline(packing_config(video, data_dir)).run()
    assert summary["session"] == 1
    rows = sqlite3.connect(Path(data_dir) / "boxcount.db").execute(
        "SELECT pieces FROM events").fetchall()
    assert rows[0][0] == 5
