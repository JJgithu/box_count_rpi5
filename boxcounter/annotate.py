"""Drawing helpers for the live preview / debug views."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection
from .tracker import Track

_GREEN = (80, 220, 80)
_YELLOW = (60, 220, 240)
_BLUE = (240, 160, 60)
_RED = (60, 60, 240)
_WHITE = (240, 240, 240)


def draw_overlay(frame: np.ndarray,
                 tracks: List[Track],
                 roi_px: Tuple[int, int, int, int],
                 axis: str, line_px: float,
                 total: int, fps: float, rate_per_min: float) -> np.ndarray:
    """Return an annotated copy of the frame."""
    out = frame.copy()
    fh, fw = out.shape[:2]

    # ROI
    x0, y0, w, h = roi_px
    cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), _BLUE, 1)

    # Counting line
    lp = int(round(line_px))
    if axis == "y":
        cv2.line(out, (0, lp), (fw, lp), _YELLOW, 2)
    else:
        cv2.line(out, (lp, 0), (lp, fh), _YELLOW, 2)

    # Tracks
    for tr in tracks:
        x, y, bw, bh = tr.bbox
        color = _RED if tr.counted else _GREEN
        cv2.rectangle(out, (int(x), int(y)), (int(x + bw), int(y + bh)), color, 2)
        cx, cy = int(tr.centroid[0]), int(tr.centroid[1])
        cv2.circle(out, (cx, cy), 3, color, -1)
        cv2.putText(out, f"#{tr.track_id}", (int(x), max(12, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        pts = [(int(px), int(py)) for px, py in tr.history]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, color, 1)

    # Status bar
    cv2.rectangle(out, (0, 0), (fw, 22), (0, 0, 0), -1)
    cv2.putText(out, f"count: {total}   {rate_per_min:.1f}/min   {fps:.1f} fps",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)
    return out


def mask_to_bgr(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
