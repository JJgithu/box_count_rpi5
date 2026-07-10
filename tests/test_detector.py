import numpy as np

from boxcounter.config import ProcessingConfig
from boxcounter.detector import BoxDetector, merge_close_boxes


def belt_frame(w=640, h=480, gray=110):
    return np.full((h, w, 3), gray, np.uint8)


def test_merge_close_boxes_fuses_fragments():
    boxes = [(100, 100, 50, 40), (160, 105, 50, 40), (400, 300, 60, 60)]
    merged = merge_close_boxes(boxes, gap=20)
    assert len(merged) == 2
    big = max(merged, key=lambda b: b[2] * b[3])
    assert big == (100, 100, 110, 45)


def test_merge_respects_gap():
    boxes = [(0, 0, 40, 40), (100, 0, 40, 40)]
    assert len(merge_close_boxes(boxes, gap=10)) == 2
    assert len(merge_close_boxes(boxes, gap=80)) == 1


def test_merge_chain_collapses():
    boxes = [(0, 0, 30, 30), (40, 0, 30, 30), (80, 0, 30, 30)]
    merged = merge_close_boxes(boxes, gap=15)
    assert len(merged) == 1
    assert merged[0] == (0, 0, 110, 30)


def _run_detector(cfg, frames):
    det = BoxDetector(cfg)
    result = None
    for f in frames:
        result = det.process(f)
    return result


def test_mog2_detects_solid_box():
    cfg = ProcessingConfig(warmup_frames=0, min_area_frac=0.01,
                           close_kernel=31, mog2_var_threshold=25)
    frames = [belt_frame() for _ in range(60)]
    boxed = belt_frame()
    boxed[200:320, 250:420] = (70, 110, 150)   # cardboard box
    frames += [boxed] * 3
    detections, mask = _run_detector(cfg, frames)
    assert len(detections) == 1
    x, y, w, h = detections[0].bbox
    assert abs(x - 250) < 25 and abs(y - 200) < 25
    assert abs(w - 170) < 40 and abs(h - 120) < 40


def test_mog2_detects_open_box_as_single_blob():
    """An open box (interior ~ belt gray) must not split into fragments."""
    cfg = ProcessingConfig(warmup_frames=0, min_area_frac=0.01,
                           close_kernel=31, merge_gap_px=24,
                           mog2_var_threshold=25)
    frames = [belt_frame() for _ in range(60)]
    boxed = belt_frame()
    boxed[180:340, 230:430] = (70, 110, 150)       # rim
    boxed[196:324, 246:414] = (104, 106, 108)      # interior close to belt
    frames += [boxed] * 3
    detections, mask = _run_detector(cfg, frames)
    assert len(detections) == 1, f"open box fragmented: {len(detections)} blobs"
    _, _, w, h = detections[0].bbox
    assert w > 150 and h > 120


def test_static_method(tmp_path):
    import cv2
    bg_path = tmp_path / "bg.png"
    cv2.imwrite(str(bg_path), np.full((480, 640), 110, np.uint8))
    cfg = ProcessingConfig(method="static", background_image=str(bg_path),
                           warmup_frames=0, min_area_frac=0.01)
    boxed = belt_frame()
    boxed[200:320, 250:420] = (70, 110, 150)
    detections, _ = _run_detector(cfg, [boxed])
    assert len(detections) == 1


def test_roi_offsets_detections_to_frame_coords():
    cfg = ProcessingConfig(warmup_frames=0, roi=[0.25, 0.25, 0.5, 0.5],
                           min_area_frac=0.02, mog2_var_threshold=25)
    frames = [belt_frame() for _ in range(60)]
    boxed = belt_frame()
    boxed[240:320, 250:400] = (70, 110, 150)   # inside the centered ROI
    frames += [boxed] * 3
    detections, _ = _run_detector(cfg, frames)
    assert len(detections) == 1
    x, y, w, h = detections[0].bbox
    assert 230 <= x <= 270 and 220 <= y <= 260, "bbox must be in full-frame coords"


def test_freeze_learning_relearns_after_permanent_scene_change():
    """After a permanent lighting step the frozen model must rebuild itself
    and detect boxes again, instead of latching up blind forever."""
    cfg = ProcessingConfig(warmup_frames=0, min_area_frac=0.01,
                           mog2_history=60, relearn_after_freeze_frames=20,
                           freeze_learning_fg_fraction=0.02, mog2_var_threshold=25)
    det = BoxDetector(cfg)
    for _ in range(60):                       # settle on the dark empty belt
        det.process(belt_frame(gray=110))
    # Warehouse lights switch on: the whole ROI now reads as foreground and
    # (without the escape hatch) learning would freeze permanently.
    for _ in range(70):                       # exceeds relearn threshold + blank
        det.process(belt_frame(gray=175))
    # A box on the NEW (bright) belt must be detected again.
    boxed = belt_frame(gray=175)
    boxed[200:320, 250:420] = (70, 110, 150)
    result = None
    for _ in range(5):
        result = det.process(boxed)
    detections, _ = result
    assert len(detections) == 1, "detector failed to recover after scene change"


def test_tiny_and_huge_blobs_filtered():
    cfg = ProcessingConfig(warmup_frames=0, min_area_frac=0.05,
                           max_area_frac=0.5, open_kernel=0, close_kernel=0,
                           mog2_var_threshold=25)
    frames = [belt_frame() for _ in range(60)]
    boxed = belt_frame()
    boxed[10:20, 10:20] = (255, 255, 255)      # speck: below min area
    boxed[100:460, 30:610] = (70, 110, 150)    # covers ~70% of ROI: above max
    frames += [boxed] * 3
    detections, _ = _run_detector(cfg, frames)
    assert detections == []
