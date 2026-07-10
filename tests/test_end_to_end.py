"""End-to-end accuracy test: synthetic conveyor video -> exact count.

Generates videos with a known number of boxes (mixed open/closed) and runs
the full pipeline (detector -> tracker -> counter -> storage) on them.
The count must match the ground truth exactly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_test_video import generate_video            # noqa: E402

from boxcounter.config import AppConfig               # noqa: E402
from boxcounter.pipeline import Pipeline              # noqa: E402


def run_pipeline_on(video_path: str, data_dir: str) -> dict:
    cfg = AppConfig()
    cfg.camera.source = "video"
    cfg.camera.video_path = video_path
    cfg.processing.roi = [0.0, 0.0, 1.0, 1.0]
    cfg.processing.warmup_frames = 60
    cfg.counting.axis = "y"
    cfg.counting.line_position = 0.55
    cfg.counting.direction = "positive"
    cfg.output.data_dir = data_dir
    cfg.output.csv = False
    cfg.web.enabled = False
    cfg.gpio.enabled = False
    pipeline = Pipeline(cfg)
    return pipeline.run()


@pytest.mark.parametrize("n_boxes,seed,open_ratio", [
    (12, 42, 0.5),    # mixed open/closed
    (8, 7, 1.0),      # all open boxes (hardest case)
    (10, 99, 0.0),    # all closed boxes
])
def test_exact_count(tmp_path, n_boxes, seed, open_ratio):
    video = str(tmp_path / f"belt_{seed}.mp4")
    truth = generate_video(video, n_boxes=n_boxes, seed=seed,
                           open_ratio=open_ratio, speed_px=6.0)
    summary = run_pipeline_on(video, str(tmp_path / "data"))
    assert summary["session"] == truth, (
        f"counted {summary['session']} of {truth} boxes "
        f"(seed={seed}, open_ratio={open_ratio})")


def test_fast_belt(tmp_path):
    """Double belt speed still counts exactly."""
    video = str(tmp_path / "fast.mp4")
    truth = generate_video(video, n_boxes=8, seed=3, speed_px=12.0)
    summary = run_pipeline_on(video, str(tmp_path / "data"))
    assert summary["session"] == truth


def test_total_resumes_from_storage(tmp_path):
    """Restarting the pipeline must resume the persisted total."""
    video = str(tmp_path / "belt.mp4")
    truth = generate_video(video, n_boxes=5, seed=11)
    data_dir = str(tmp_path / "data")
    s1 = run_pipeline_on(video, data_dir)
    assert s1["session"] == truth
    s2 = run_pipeline_on(video, data_dir)
    assert s2["session"] == truth
    assert s2["total"] == 2 * truth, "total must persist across restarts"
