"""Box detection via background subtraction.

Two methods:
  - mog2:   adaptive Gaussian-mixture background model (default). Handles
            gradual lighting drift and moderately textured belts.
  - static: absolute difference against a captured empty-belt image
            (tools/capture_background.py). Best for uniform belts under
            constant lighting; boxes can never be "absorbed".

Open boxes are the hard case: the interior can resemble the belt, so the
foreground mask fragments into a rim. A large morphological close plus
merging of nearby blobs fuses the fragments back into one detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import ProcessingConfig

log = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]        # x, y, w, h in full-frame pixels
    centroid: Tuple[float, float]          # full-frame pixels
    area: float                            # blob area in pixels


def _kernel(size: int):
    if size and size > 1:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return None


def merge_close_boxes(boxes: List[Tuple[int, int, int, int]], gap: int) -> List[Tuple[int, int, int, int]]:
    """Union rectangles whose expanded versions overlap.

    Rectangles closer than `gap` pixels are merged into one. Repeats until
    stable so chains of fragments collapse into a single box.
    """
    boxes = list(boxes)
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out: List[Tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                if (x1 - gap < x2 + w2 and x2 - gap < x1 + w1 and
                        y1 - gap < y2 + h2 and y2 - gap < y1 + h1):
                    right = max(x1 + w1, x2 + w2)
                    bottom = max(y1 + h1, y2 + h2)
                    x1, y1 = min(x1, x2), min(y1, y2)
                    w1, h1 = right - x1, bottom - y1
                    used[j] = True
                    merged = True
            out.append((x1, y1, w1, h1))
            used[i] = True
        boxes = out
    return boxes


class BoxDetector:
    """Segments moving boxes from the belt and returns bounding boxes."""

    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self._initialized = False
        self._roi_px: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._subtractor = None
        self._background: Optional[np.ndarray] = None
        self._blur = None
        self._open_k = None
        self._close_k = None
        self._dilate_k = None
        self._min_area = 0.0
        self._max_area = 0.0
        self._prev_fg_fraction = 0.0
        self._prev_box_present = False  # a box-sized blob was accepted last frame
        self._frozen_frames = 0        # consecutive frames with learning frozen
        self._relearn_blank = 0        # frames to blank after a model rebuild
        # Set when the escape hatch rebuilds the model mid-run; the pipeline
        # reads and clears it to reset downstream state (tracker, packing).
        self.relearned = False

    @property
    def roi_px(self) -> Tuple[int, int, int, int]:
        return self._roi_px

    def _init_for_frame(self, frame: np.ndarray) -> None:
        cfg = self.cfg
        fh, fw = frame.shape[:2]
        rx, ry, rw, rh = cfg.roi
        x0, y0 = int(rx * fw), int(ry * fh)
        w, h = max(1, int(rw * fw)), max(1, int(rh * fh))
        w = min(w, fw - x0)
        h = min(h, fh - y0)
        self._roi_px = (x0, y0, w, h)

        roi_area = float(w * h)
        self._min_area = cfg.min_area_frac * roi_area
        self._max_area = cfg.max_area_frac * roi_area

        self._blur = cfg.blur_kernel if cfg.blur_kernel and cfg.blur_kernel > 1 else 0
        if self._blur and self._blur % 2 == 0:
            self._blur += 1
        self._open_k = _kernel(cfg.open_kernel)
        self._close_k = _kernel(cfg.close_kernel)
        self._dilate_k = _kernel(cfg.dilate_kernel)

        if cfg.method == "mog2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=cfg.mog2_history,
                varThreshold=cfg.mog2_var_threshold,
                detectShadows=cfg.detect_shadows,
            )
        else:  # static
            flag = cv2.IMREAD_COLOR if cfg.use_color else cv2.IMREAD_GRAYSCALE
            bg = cv2.imread(cfg.background_image, flag)
            if bg is None:
                raise FileNotFoundError(
                    f"Background image not found: {cfg.background_image}. "
                    "Capture one with tools/capture_background.py or use method: mog2")
            if bg.shape[:2] != (fh, fw):
                bg = cv2.resize(bg, (fw, fh))
            bg = bg[y0:y0 + h, x0:x0 + w]
            if self._blur:
                bg = cv2.GaussianBlur(bg, (self._blur, self._blur), 0)
            self._background = bg

        self._initialized = True
        log.info("Detector initialized: frame %dx%d, ROI %s, blob area %d..%d px",
                 fw, fh, self._roi_px, int(self._min_area), int(self._max_area))

    def reset(self) -> None:
        """Force re-initialization (e.g. after a camera restart)."""
        self._initialized = False
        self._prev_fg_fraction = 0.0
        self._prev_box_present = False
        self._frozen_frames = 0
        self._relearn_blank = 0
        self.relearned = False

    def process(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Return (detections in full-frame coords, binary ROI mask for debug)."""
        if not self._initialized:
            self._init_for_frame(frame)
        cfg = self.cfg
        x0, y0, w, h = self._roi_px

        roi = frame[y0:y0 + h, x0:x0 + w]
        if cfg.use_color:
            proc = roi if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            proc = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if self._blur:
            proc = cv2.GaussianBlur(proc, (self._blur, self._blur), 0)

        if cfg.method == "mog2":
            # Learn the background only while the belt is (nearly) empty, so
            # passing/stalled boxes are never absorbed into the model.
            #
            # Two independent triggers, because neither alone is enough:
            #  - foreground fraction, which catches large boxes but misses a
            #    box smaller than the threshold (it would be quietly learned
            #    away within a second of stopping at the packing station);
            #  - an accepted box-sized blob in the previous frame, which
            #    catches small boxes at any size we are willing to count,
            #    while sensor noise (always below min_area) never trips it.
            lr = cfg.learning_rate
            if (self._prev_fg_fraction > cfg.freeze_learning_fg_fraction
                    or self._prev_box_present):
                lr = 0.0
            # Count toward the relearn escape hatch ONLY when almost the whole
            # ROI is foreground. Normal box traffic covers part of the ROI with
            # gaps and never sustains this, so it stays frozen indefinitely (as
            # intended); a global lighting/exposure shift saturates the ROI and
            # triggers a rebuild.
            if self._prev_fg_fraction > cfg.relearn_fg_fraction:
                self._frozen_frames += 1
            else:
                self._frozen_frames = 0
            # Escape hatch: if learning stays frozen continuously for too long,
            # the scene has changed for good (lights switched, exposure/AWB
            # re-locked after a camera restart, belt re-taped) and a frozen
            # model can never recover on its own — the whole ROI reads as
            # foreground forever and the detector goes blind. Rebuild the model
            # from the current scene and blank detections while it re-settles.
            if (cfg.relearn_after_freeze_frames > 0
                    and self._frozen_frames >= cfg.relearn_after_freeze_frames):
                log.warning("Background learning frozen for %d frames; "
                            "rebuilding model from current scene", self._frozen_frames)
                self._subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=cfg.mog2_history,
                    varThreshold=cfg.mog2_var_threshold,
                    detectShadows=cfg.detect_shadows)
                self._frozen_frames = 0
                self._relearn_blank = max(cfg.mog2_history // 4, 30)
                self.relearned = True   # pipeline resets tracker/packing state
                lr = -1  # let the fresh model adapt quickly
            fg = self._subtractor.apply(proc, learningRate=lr)
            # MOG2 marks shadows as 127; keep only definite foreground (255).
            _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        else:
            diff = cv2.absdiff(proc, self._background)
            if diff.ndim == 3:
                # a box differing in ANY channel is foreground
                diff = diff.max(axis=2)
            _, mask = cv2.threshold(diff, cfg.static_diff_threshold, 255, cv2.THRESH_BINARY)

        self._prev_fg_fraction = cv2.countNonZero(mask) / float(w * h)

        # While a freshly rebuilt model settles, report no detections so the
        # transient garbage mask is never fed to the tracker/counter.
        if self._relearn_blank > 0:
            self._relearn_blank -= 1
            self._prev_box_present = False
            return [], mask

        if self._open_k is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)
        if self._close_k is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)
        if self._dilate_k is not None:
            mask = cv2.dilate(mask, self._dilate_k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            # External contour area includes any interior holes (open boxes).
            if cv2.contourArea(c) < self._min_area * 0.5:
                continue  # drop specks early; final size check after merging
            boxes.append((bx, by, bw, bh))

        boxes = merge_close_boxes(boxes, cfg.merge_gap_px)

        detections: List[Detection] = []
        for bx, by, bw, bh in boxes:
            area = float(bw * bh)
            if area < self._min_area or area > self._max_area:
                continue
            fx, fy = bx + x0, by + y0
            detections.append(Detection(
                bbox=(fx, fy, bw, bh),
                centroid=(fx + bw / 2.0, fy + bh / 2.0),
                area=area,
            ))
        # Remembered for the next frame's learning decision: while a real box
        # is in view the model must not adapt, however small the box is.
        self._prev_box_present = bool(detections)
        return detections, mask
