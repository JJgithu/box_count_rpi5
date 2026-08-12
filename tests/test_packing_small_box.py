"""Small boxes and slow packers: the field failure "only one pad is counted".

Three defects combined to produce it, all size-dependent, which is why the
original end-to-end tests (large boxes, brisk half-second reaches) never saw
any of them:

  1. a parked box below the background-freeze threshold was absorbed into
     the background model about a second after stopping (tests/test_parked_box.py);
  2. a hand covering a small box registers almost no motion inside it — the
     hand's high-contrast edges fall outside the interior region — so the
     "this occupant is static furniture" rule fired on an ordinary pause,
     closed the visit and folded the arm into the ring baseline;
  3. with the visit closed, a box still hidden under the hand was declared
     "track lost" and the session was abandoned.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_packing_video import generate_packing_video     # noqa: E402

from boxcounter.config import AppConfig                   # noqa: E402
from boxcounter.pipeline import Pipeline                  # noqa: E402


def run(video: str, data_dir: str) -> list:
    cfg = AppConfig()                    # shipped defaults
    cfg.camera.source = "video"
    cfg.camera.video_path = video
    cfg.processing.roi = [0.0, 0.0, 1.0, 1.0]
    cfg.processing.warmup_frames = 60
    cfg.counting.axis = "y"
    cfg.counting.line_position = 0.78
    cfg.counting.direction = "positive"
    cfg.packing.enabled = True
    cfg.packing.zone = [0.0, 0.05, 1.0, 0.55]
    cfg.output.data_dir = data_dir
    cfg.output.csv = False
    cfg.web.enabled = False
    cfg.gpio.enabled = False
    Pipeline(cfg).run()
    rows = sqlite3.connect(Path(data_dir) / "boxcount.db").execute(
        "SELECT pieces FROM events ORDER BY id").fetchall()
    return [r[0] for r in rows]


@pytest.mark.parametrize("box", [(72, 60), (95, 80)])
def test_small_box_counts_every_pad(tmp_path, box):
    """A small box must not be lost partway through packing."""
    video = str(tmp_path / f"small_{box[0]}.mp4")
    generate_packing_video(video, pieces_per_box=(4,), seed=5,
                           box_w=(box[0], box[0] + 1), box_h=(box[1], box[1] + 1),
                           gap_s=1.0, hold_s=0.6)
    assert run(video, str(tmp_path / "d")) == [4], (
        f"a {box[0]}x{box[1]} box lost pads mid-pack")


def test_slow_packer_resting_a_hand_in_a_small_box(tmp_path):
    """The hand covers the box and rests several seconds per pad — the case
    that used to record one pad (or none) and then lose the box entirely."""
    video = str(tmp_path / "resting.mp4")
    generate_packing_video(video, pieces_per_box=(4,), seed=5,
                           box_w=(72, 73), box_h=(60, 61),
                           gap_s=0.5, hold_s=5.5)
    assert run(video, str(tmp_path / "d")) == [4], (
        "pads were lost while the packer rested a hand in the box")


def test_long_pause_between_pads_on_a_small_box(tmp_path):
    """Several seconds of a perfectly still belt between pads: the window in
    which a stationary small box used to be absorbed into the background."""
    video = str(tmp_path / "paused.mp4")
    generate_packing_video(video, pieces_per_box=(3,), seed=9,
                           box_w=(70, 71), box_h=(58, 59),
                           gap_s=4.0, hold_s=0.5)
    assert run(video, str(tmp_path / "d")) == [3], (
        "the box was lost during the pause between pads")


def test_large_box_still_works(tmp_path):
    """The original geometry must be unaffected by the small-box fixes."""
    video = str(tmp_path / "large.mp4")
    generate_packing_video(video, pieces_per_box=(3, 2), seed=7)
    assert run(video, str(tmp_path / "d")) == [3, 2]
