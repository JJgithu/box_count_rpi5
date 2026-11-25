"""Packing-station mode: boxes stop in front of the paper machine, get filled
with pads, and slide away. Synthetic videos with known ground truth."""

import shutil
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_station_video import OUTLET_ZONE, generate_station_video  # noqa: E402

from boxcounter.camera import flicker_safe_exposure         # noqa: E402
from boxcounter.config import AppConfig, StationConfig, load_config  # noqa: E402
from boxcounter.configedit import add_missing, set_values   # noqa: E402
from boxcounter.pipeline import Pipeline                    # noqa: E402
from boxcounter.station import StationMonitor               # noqa: E402

FPS = 15          # fewer frames to render and process; the logic is time-based


def run_station(path, cfg=None, roi=(0.0, 0.0, 1.0, 1.0)):
    """Feed a video through the station monitor; return the count events."""
    cfg = cfg or StationConfig()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    mon, events, i = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if mon is None:
            mon = StationMonitor(cfg, list(roi), frame.shape[:2])
        events += mon.update(frame, i / fps)
    cap.release()
    return events


def check(events, truth, time_tol=2.0):
    assert len(events) == truth["boxes"], (
        f"boxes: counted {len(events)}, truth {truth['boxes']}")
    pads = [e.pieces for e in events]
    assert pads == truth["pads"], f"pads per box {pads} != truth {truth['pads']}"
    for e, t in zip(events, truth["pack_seconds"]):
        assert abs(e.pack_seconds - t) < time_tol, (
            f"pack time {e.pack_seconds:.1f}s vs truth {t:.1f}s")


def station_pipeline_config(video, data_dir) -> AppConfig:
    cfg = AppConfig()
    cfg.counting.mode = "station"
    cfg.camera.source = "video"
    cfg.camera.video_path = str(video)
    cfg.processing.roi = [0.05, 0.0, 0.90, 1.0]
    cfg.output.data_dir = str(data_dir)
    cfg.web.enabled = False
    cfg.gpio.enabled = False
    return cfg


# -- end to end ---------------------------------------------------------------

def test_boxes_pads_and_pack_time_end_to_end(tmp_path):
    """Three boxes through the whole program: database and CSV rows."""
    video = tmp_path / "station.mp4"
    truth = generate_station_video(str(video), pads_per_box=(3, 2, 4), fps=FPS)
    summary = Pipeline(station_pipeline_config(video, tmp_path / "data")).run()
    assert summary["total"] == truth["boxes"]

    rows = sqlite3.connect(tmp_path / "data" / "boxcount.db").execute(
        "SELECT pieces, pack_seconds FROM events ORDER BY id").fetchall()
    assert [r[0] for r in rows] == truth["pads"]
    for (_, secs), t in zip(rows, truth["pack_seconds"]):
        assert abs(secs - t) < 2.0

    csvs = list((tmp_path / "data").glob("events_*.csv"))
    assert len(csvs) == 1
    lines = csvs[0].read_text().strip().splitlines()
    assert lines[0].startswith("box,timestamp,pads,pack_seconds")
    assert [int(l.split(",")[2]) for l in lines[1:]] == truth["pads"]


def test_hands_alone_are_never_counted(tmp_path):
    """A hand resting on the empty rollers for seconds, then leaving: the
    exact thing the line counter used to count as a box."""
    video = tmp_path / "hands.mp4"
    truth = generate_station_video(str(video), pads_per_box=(), fps=FPS,
                                   rest_hand=True)
    assert truth["boxes"] == 0
    assert run_station(video) == []


@pytest.mark.parametrize("kw", [
    dict(seed=7, pads_per_box=(4, 4), gap_s=0.4),          # fast packer
    dict(seed=8, pads_per_box=(3, 5), gap_s=2.0),          # slow packer
    dict(seed=9, pads_per_box=(2, 3), box_size=(150, 110)),  # small box
    dict(seed=10, pads_per_box=(3, 2), box_size=(260, 190)),  # big box
], ids=["fast-packer", "slow-packer", "small-box", "big-box"])
def test_packing_styles_and_box_sizes(tmp_path, kw):
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), fps=FPS, **kw)
    check(run_station(video), truth)


def test_box_standing_there_at_startup_does_not_break_counting(tmp_path):
    """A box already on the spot when the counter starts becomes part of the
    learned picture; once it is taken away the spot must be repaired, not
    turned into a phantom box, and the following boxes count normally."""
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), pads_per_box=(3, 2), fps=FPS,
                                   seed=13, box_at_start=2)
    check(run_station(video), truth)


def test_shoving_the_box_is_not_a_pad(tmp_path):
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), pads_per_box=(3, 2), fps=FPS,
                                   seed=23, nudge_px=18)
    check(run_station(video), truth)


def test_empty_handed_reach_is_not_a_pad(tmp_path):
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), pads_per_box=(3, 2), fps=FPS,
                                   seed=21, empty_reach=True)
    check(run_station(video), truth)


def test_lighting_change_mid_shift(tmp_path):
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), pads_per_box=(3, 2, 2), fps=FPS,
                                   seed=24, light_step=0.10)
    check(run_station(video), truth)


def test_packer_who_starts_and_finishes_without_pausing(tmp_path):
    """Real packing: the hand goes for the first pad the moment the box
    arrives, and the box is pushed on the moment the last pad is in. There is
    never a quiet moment before the first or after the last pad."""
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), pads_per_box=(2, 3, 2), fps=FPS,
                                   seed=31, settle_s=0.0, end_s=0.0)
    check(run_station(video), truth)


@pytest.mark.parametrize("kw", [
    dict(seed=32, gap_s=0.3),
    dict(seed=33, gap_s=0.0, stay_hand=True),
], ids=["short-gaps", "never-stops"])
def test_hands_working_nonstop_never_split_or_lose_a_box(tmp_path, kw):
    """A hand across the middle of the box splits it into two pieces, each
    big enough to pass for a small box: it must stay ONE box, and a box that
    never sees a quiet moment must still be counted, with its time."""
    video = tmp_path / "v.mp4"
    truth = generate_station_video(str(video), pads_per_box=(2, 3, 2), fps=FPS,
                                   settle_s=0.0, end_s=0.0, **kw)
    events = run_station(video)
    assert len(events) == truth["boxes"]
    for e, t in zip(events, truth["pack_seconds"]):
        assert abs(e.pack_seconds - t) < 2.0


@pytest.mark.parametrize("kw", [
    dict(pads_per_box=(3, 2, 4)),
    # the hand pauses at the outlet, inside the zone, and carries the pad
    # through it: that must stay one reach, not a pad "in the air" plus the
    # pad landing
    dict(seed=1, pads_per_box=(3, 2, 4), fps=30),
    # the zone covers the top of a small box and an arm resting across the
    # rest cuts it in two: the arm must not become part of the box
    dict(seed=9, pads_per_box=(2, 3), box_size=(150, 110), fps=30),
], ids=["default", "pause-at-outlet", "small-box"])
def test_outlet_zone_keeps_counting_exact(tmp_path, kw):
    """With the outlet measured (setup step 3), the hanging paper is left out
    of the box; counting stays exact."""
    video = tmp_path / "v.mp4"
    kw.setdefault("fps", FPS)
    truth = generate_station_video(str(video), **kw)
    cfg = StationConfig()
    cfg.outlet_zone = list(OUTLET_ZONE)
    check(run_station(video, cfg), truth)


def test_pads_not_counted_when_disabled(tmp_path):
    video = tmp_path / "v.mp4"
    generate_station_video(str(video), pads_per_box=(2,), fps=FPS, rest_hand=False)
    cfg = StationConfig()
    cfg.count_pads = False
    events = run_station(video, cfg)
    assert len(events) == 1
    assert events[0].pieces is None and events[0].pack_seconds > 3


# -- configuration --------------------------------------------------------------

def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_station_is_the_default_mode(tmp_path):
    assert load_config(write(tmp_path, "")).counting.mode == "station"
    assert load_config(ROOT / "config" / "config.yaml").counting.mode == "station"


@pytest.mark.parametrize("text", [
    "counting:\n  mode: table\n",
    "station:\n  zone: [0.5, 0.5, 0.8, 0.8]\n",
    "station:\n  min_box_frac: 0.5\n  max_box_frac: 0.4\n",
    "station:\n  box_color: blue\n",
    "station:\n  depart_cover: 0.6\n  present_cover: 0.5\n",
    "station:\n  quiet_seconds: 0\n",
    "camera:\n  flicker_hz: 55\n",
    "station:\n  pad_method: magic\n",
    "station:\n  pad_method: outlet\n",
    "station:\n  outlet_zone: [0.9, 0.9, 0.5, 0.5]\n",
])
def test_bad_station_settings_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, text))


def test_missing_station_section_can_be_added_to_an_old_config(tmp_path):
    """The setup tool must be able to write to a config from before station
    mode existed (no station: section, no counting.mode)."""
    old = ("camera:\n  source: picamera\n\ncounting:\n  axis: y   # belt\n\n"
           "web:\n  port: 8080\n")
    updates = {"counting.mode": "station", "station.zone": [0.2, 0.1, 0.6, 0.7],
               "station.min_box_frac": 0.05, "station.box_color": "any"}
    text, _ = set_values(old, updates)
    text, added = add_missing(text, updates)
    assert len(added) == 4
    cfg = load_config(write(tmp_path, text))
    assert cfg.counting.mode == "station"
    assert cfg.station.zone == [0.2, 0.1, 0.6, 0.7]
    assert cfg.station.box_color == "any"
    assert cfg.web.port == 8080


def test_setup_tool_measures_the_box_and_writes_a_valid_config(tmp_path):
    import station_setup
    video = tmp_path / "v.mp4"
    generate_station_video(str(video), pads_per_box=(1,), fps=FPS, rest_hand=False)
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(ROOT / "config" / "config.yaml", cfg_path)
    cfg = load_config(cfg_path)

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    empty = frames[int(1.0 * FPS)]
    # the box parks at 2.5 s (lead-in) + 1.0 s (slide); hands come 0.8 s later
    box = frames[int(3.7 * FPS)]
    res = station_setup.analyse(empty, box, cfg)
    assert res["found"]
    assert 0.08 < res["area_frac"] < 0.3
    assert res["brown_frac"] > 0.8
    assert res["edges"] == []
    updates = station_setup.propose(res, cfg)
    zx, zy, zw, zh = updates["station.zone"]
    assert zx < 0.5 < zx + zw and zy < 0.4 < zy + zh     # box centre inside
    assert updates["station.min_box_frac"] < res["area_frac"] < updates["station.max_box_frac"]

    text, _ = set_values(cfg_path.read_text(), updates)
    cfg_path.write_text(text)
    assert load_config(cfg_path).station.zone == updates["station.zone"]


def test_setup_tool_finds_the_pad_outlet(tmp_path):
    import station_setup
    video = tmp_path / "v.mp4"
    generate_station_video(str(video), pads_per_box=(2,), fps=FPS, rest_hand=False)
    cfg = load_config(ROOT / "config" / "config.yaml")
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    empty = frames[int(1.0 * FPS)]
    box = frames[int(3.7 * FPS)]
    after = frames[int(6.6 * FPS)]          # first pad taken, next one hanging
    res = station_setup.analyse(empty, box, cfg)
    out = station_setup.find_outlet(box, after, cfg, res)
    assert out is not None
    zx, zy, zw, zh = out["zone"]
    tx, ty, tw, th = OUTLET_ZONE
    # it covers the real outlet's centre
    assert zx <= tx + tw / 2 <= zx + zw and zy <= ty + th / 2 <= zy + zh

    # Saved, and the smallest accepted box is sized by what stays visible.
    updates = station_setup.propose(res, cfg)
    hidden = station_setup.add_outlet(updates, out, res)
    assert 0.0 < hidden <= station_setup.MAX_HIDDEN
    assert updates["station.outlet_zone"] == out["zone"]
    visible = res["area_frac"] * (1.0 - hidden)
    assert updates["station.min_box_frac"] < visible

    # A "pad" hanging over most of the box would hide it: not saved.
    updates = station_setup.propose(res, cfg)
    x, y, w, h = res["bbox"]
    ww, wh = res["size"]
    over_box = {"zone": [x / ww, y / wh, w / ww, h / wh], "frac": 0.1}
    assert station_setup.add_outlet(updates, over_box, res) > station_setup.MAX_HIDDEN
    assert "station.outlet_zone" not in updates


# -- camera -----------------------------------------------------------------------

@pytest.mark.parametrize("exp,gain", [(12000, 2.0), (20000, 1.5), (30000, 4.0),
                                      (40000, 3.0), (9000, 1.0)])
def test_flicker_safe_exposure_is_whole_flicker_periods(exp, gain):
    new_exp, new_gain = flicker_safe_exposure(exp, gain, 60, 1e6 / 30)
    period = 1e6 / 120
    assert abs(new_exp / period - round(new_exp / period)) < 0.01
    assert new_exp <= 1e6 / 30 + 1                         # fits in a frame
    assert new_gain >= 1.0
    # brightness (exposure x gain) kept unless the gain would drop below 1
    if gain * exp / new_exp >= 1.0:
        assert abs(new_exp * new_gain - exp * gain) / (exp * gain) < 0.01


def test_very_short_exposure_is_left_alone():
    assert flicker_safe_exposure(4000, 1.0, 60, 1e6 / 30) == (4000, 1.0)
