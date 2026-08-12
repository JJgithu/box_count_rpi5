"""A box that stops must stay detected for as long as it sits there.

Field bug this guards: a box parked at the packing station was quietly
absorbed into the background model about a second after stopping, so the
packer could place one pad and then the system "stopped recognising the
box". It only happened to boxes smaller than freeze_learning_fg_fraction
(2% of the ROI) — big boxes froze background learning and survived, small
ones did not, which is why it looked size-dependent and intermittent.
"""

import cv2
import numpy as np
import pytest

from boxcounter.config import ProcessingConfig
from boxcounter.detector import BoxDetector

W, H, FPS = 640, 480, 30


def belt_frame(rng):
    fr = np.full((H, W, 3), 108, np.uint8)
    return np.clip(fr.astype(np.int16) + rng.normal(0, 2, (H, W, 1)),
                   0, 255).astype(np.uint8)


def run_parked_box(cfg, bw, bh, seconds=25.0, park_at=3.0):
    """Box slides in, parks in the middle, and never moves again.
    Returns (detected_while_arriving, detected_at_the_end)."""
    rng = np.random.default_rng(1)
    det = BoxDetector(cfg)
    arriving = False
    tail = []
    total = int(seconds * FPS)
    for f in range(total):
        fr = belt_frame(rng)
        if f >= FPS:
            prog = min(1.0, (f - FPS) / (park_at * FPS - FPS))
            y = int(-bh + prog * ((H - bh) // 2 + bh))
            x = (W - bw) // 2
            cv2.rectangle(fr, (x, max(0, y)), (x + bw, max(0, y) + bh),
                          (70, 110, 150), -1)
        found = bool(det.process(fr)[0])
        if f < park_at * FPS:
            arriving = arriving or found
        if f >= total - FPS:            # the final second
            tail.append(found)
    return arriving, (sum(tail) > len(tail) * 0.5)


@pytest.mark.parametrize("bw,bh", [
    (200, 160),      # 11.6% of the ROI — always froze learning, always worked
    (120, 100),      # 4.3%
    (90, 70),        # 2.3% — just above the old 2% freeze threshold
    (75, 60),        # 1.6% — just below it: this is where boxes vanished
    (62, 52),        # 1.2% — the smallest the shipped min_area_frac accepts
])
def test_parked_box_is_still_detected_25s_later(bw, bh):
    cfg = ProcessingConfig()          # shipped defaults
    roi_frac = (bw * bh) / ((cfg.roi[2] * W) * (cfg.roi[3] * H))
    assert roi_frac > cfg.min_area_frac, (
        "test box is below min_area_frac and would never be counted anyway")
    arriving, at_end = run_parked_box(cfg, bw, bh)
    assert arriving, f"{bw}x{bh} ({roi_frac:.4f} of ROI) never detected at all"
    assert at_end, (
        f"{bw}x{bh} ({roi_frac:.4f} of ROI) was absorbed into the background "
        "while parked — the packer would lose the box mid-pack")


def test_empty_belt_still_produces_no_detections():
    """The freeze must not be so eager that an empty belt reads as a box."""
    rng = np.random.default_rng(7)
    det = BoxDetector(ProcessingConfig())
    found = 0
    for _ in range(FPS * 10):
        if det.process(belt_frame(rng))[0]:
            found += 1
    assert found == 0, "empty belt must stay empty"


def test_background_still_adapts_when_no_box_is_present():
    """Freezing on a present box must not stop the model adapting to a
    gradual lighting change on an empty belt."""
    rng = np.random.default_rng(3)
    cfg = ProcessingConfig()
    det = BoxDetector(cfg)
    for _ in range(FPS * 3):                       # settle
        det.process(belt_frame(rng))
    # belt slowly brightens, no box anywhere
    for i in range(FPS * 12):
        fr = np.full((H, W, 3), 108 + min(40, i // 8), np.uint8)
        fr = np.clip(fr.astype(np.int16) + rng.normal(0, 2, (H, W, 1)),
                     0, 255).astype(np.uint8)
        det.process(fr)
    # settled at the new brightness: no phantom detections
    phantom = 0
    for _ in range(FPS * 3):
        fr = np.full((H, W, 3), 148, np.uint8)
        fr = np.clip(fr.astype(np.int16) + rng.normal(0, 2, (H, W, 1)),
                     0, 255).astype(np.uint8)
        if det.process(fr)[0]:
            phantom += 1
    assert phantom < FPS, "model failed to adapt to a lighting change"


def test_a_box_leaving_lets_the_belt_go_quiet_again():
    """After a box departs there must be no lingering ghost detection.

    The belt starts empty, as it does at startup — a box already sitting
    there when the model is first built is a different (documented) case,
    handled by the relearn hatch and the wizard's arrival check.
    """
    rng = np.random.default_rng(5)
    det = BoxDetector(ProcessingConfig())
    bw, bh = 75, 60
    for _ in range(FPS * 4):                       # empty belt: learn it
        det.process(belt_frame(rng))
    for _ in range(FPS * 8):                       # box arrives and parks
        fr = belt_frame(rng)
        cv2.rectangle(fr, (280, 210), (280 + bw, 210 + bh), (70, 110, 150), -1)
        det.process(fr)
    ghost = 0
    for _ in range(FPS * 10):                      # box gone
        if det.process(belt_frame(rng))[0]:
            ghost += 1
    assert ghost < FPS, "the vacated spot kept producing a ghost box"
