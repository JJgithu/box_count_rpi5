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
_TEAL = (170, 180, 70)


def draw_overlay(frame: np.ndarray,
                 tracks: List[Track],
                 roi_px: Tuple[int, int, int, int],
                 axis: str, line_px: float,
                 total: int, fps: float, rate_per_min: float,
                 packing: Optional[dict] = None) -> np.ndarray:
    """Return an annotated copy of the frame.

    packing (optional): {"zone_px": (x,y,w,h), "session_bbox": (x,y,w,h)|None,
    "pieces": int, "elapsed_s": float, "hand_in": bool}
    """
    out = frame.copy()
    fh, fw = out.shape[:2]

    # ROI
    x0, y0, w, h = roi_px
    cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), _BLUE, 1)

    # Packing zone + active session
    if packing is not None:
        zx, zy, zw, zh = packing["zone_px"]
        cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), _TEAL, 1)
        cv2.putText(out, "PACK ZONE", (zx + 4, zy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEAL, 1, cv2.LINE_AA)
        if packing.get("session_bbox"):
            bx, by, bw2, bh2 = packing["session_bbox"]
            color = _YELLOW if packing.get("hand_in") else _TEAL
            cv2.rectangle(out, (bx - 2, by - 2), (bx + bw2 + 2, by + bh2 + 2), color, 2)
            label = f"pcs:{packing.get('pieces', 0)}  {packing.get('elapsed_s', 0):.0f}s"
            if packing.get("hand_in"):
                label += "  HAND"
            cv2.putText(out, label, (bx, min(fh - 6, by + bh2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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
    if mask.ndim == 3:
        return mask
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def draw_station_overlay(frame: np.ndarray, ov: dict, total: int, fps: float,
                         rate_per_min: float, state: str) -> np.ndarray:
    """Annotated copy of the frame for station mode.

    ov is StationMonitor.overlay(): the packing zone, the box being packed
    (with its pads and time so far), and anything being checked."""
    out = frame.copy()
    fh, fw = out.shape[:2]

    rx, ry, rw, rh = ov["roi"]
    if (rx, ry, rw, rh) != (0, 0, fw, fh):
        cv2.rectangle(out, (rx, ry), (rx + rw - 1, ry + rh - 1), _BLUE, 1)
    if ov.get("outlet"):
        ox, oy, ow, oh = ov["outlet"]
        cv2.rectangle(out, (ox, oy), (ox + ow - 1, oy + oh - 1), (60, 200, 240), 1)
        cv2.putText(out, "PAD OUTLET", (ox + 3, oy + oh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 200, 240), 1, cv2.LINE_AA)
    zx, zy, zw, zh = ov["zone"]
    cv2.rectangle(out, (zx, zy), (zx + zw - 1, zy + zh - 1), _TEAL, 1)
    cv2.putText(out, "PACKING ZONE", (zx + 4, zy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEAL, 1, cv2.LINE_AA)

    for c in ov["candidates"]:
        x, y, w, h = c["bbox"]
        color = _RED if c["rejected"] else _YELLOW
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        label = "ignored" if c["rejected"] else f"checking {c['age_s']:.1f}s"
        cv2.putText(out, label, (x + 2, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for s in ov["sessions"]:
        x, y, w, h = s["bbox"]
        color = _YELLOW if s["hand_in"] else _GREEN
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        ix, iy, iw, ih = s["interior"]
        cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), color, 1)
        label = f"BOX  {s['pads']} pads  {s['elapsed_s']:.0f}s"
        if s["hand_in"]:
            label += "  HAND"
        ty = y + h + 18 if y + h + 22 < fh else max(14, y - 8)
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (fw, 22), (0, 0, 0), -1)
    cv2.putText(out, f"boxes: {total}   {rate_per_min:.1f}/min   {fps:.1f} fps",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, fh - 22), (fw, fh), (0, 0, 0), -1)
    # OpenCV's built-in fonts are plain ASCII.
    text = state.replace("\u2014", "-").encode("ascii", "replace").decode()
    cv2.putText(out, text[:90], (6, fh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                _WHITE, 1, cv2.LINE_AA)
    return out
