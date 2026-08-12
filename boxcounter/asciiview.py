"""Render the camera's view as text, for calibrating over SSH or on a TV.

Calibration normally means looking at the foreground mask, which otherwise
needs a browser (the dashboard) or copying JPEGs to another machine. On a
Pi wired to a monitor, or over a plain SSH session, neither is convenient —
so the same information is drawn with characters instead of pixels.

    ' '  belt (nothing detected)      '#'  mostly foreground
    '.'  a little foreground          '@'  solid foreground
    ':'  some foreground

Overlays: the ROI border, the counting line, the packing zone, and boxes the
detector actually accepted.
"""

from __future__ import annotations

import shutil
from typing import List, Optional, Sequence, Tuple

import numpy as np

_RAMP = " .:#@"

_CSI = "\x1b["
_RESET = f"{_CSI}0m"
_DIM = f"{_CSI}2m"
_BLUE = f"{_CSI}34m"
_AMBER = f"{_CSI}33m"
_TEAL = f"{_CSI}36m"
_GREEN = f"{_CSI}32m"


def grid_size(max_w: int = 78, max_h: int = 22) -> Tuple[int, int]:
    cols, rows = shutil.get_terminal_size((80, 24))
    return max(24, min(max_w, cols - 4)), max(8, min(max_h, rows - 12))


def render(mask: np.ndarray,
           frame_shape: Tuple[int, int],
           cols: int, rows: int,
           roi_px: Optional[Tuple[int, int, int, int]] = None,
           line: Optional[Tuple[str, float]] = None,
           zone_px: Optional[Tuple[int, int, int, int]] = None,
           boxes: Sequence[Tuple[int, int, int, int]] = (),
           colour: bool = True) -> List[str]:
    """Return the mask as `rows` lines of `cols` characters, with overlays.

    mask must be full-frame sized (same as frame_shape) so every overlay
    lands where it really is in the image.
    """
    fh, fw = frame_shape
    if mask is None or mask.size == 0:
        return ["(no image)"]

    # Average each cell of the grid -> density character.
    ys = np.linspace(0, fh, rows + 1).astype(int)
    xs = np.linspace(0, fw, cols + 1).astype(int)
    chars = [[" "] * cols for _ in range(rows)]
    for r in range(rows):
        y0, y1 = ys[r], max(ys[r] + 1, ys[r + 1])
        band = mask[y0:y1]
        if band.size == 0:
            continue
        for c in range(cols):
            x0, x1 = xs[c], max(xs[c] + 1, xs[c + 1])
            cell = band[:, x0:x1]
            if cell.size == 0:
                continue
            frac = float(np.count_nonzero(cell)) / cell.size
            idx = 0 if frac < 0.02 else min(len(_RAMP) - 1,
                                            1 + int(frac * (len(_RAMP) - 1)))
            chars[r][c] = _RAMP[idx]

    def col_of(x: float) -> int:
        return max(0, min(cols - 1, int(x / fw * cols)))

    def row_of(y: float) -> int:
        return max(0, min(rows - 1, int(y / fh * rows)))

    overlay = [[""] * cols for _ in range(rows)]

    def paint(r: int, c: int, ch: str, col: str) -> None:
        if 0 <= r < rows and 0 <= c < cols:
            chars[r][c] = ch
            overlay[r][c] = col

    # accepted detections: outline them
    for (bx, by, bw, bh) in boxes:
        r0, r1 = row_of(by), row_of(by + bh)
        c0, c1 = col_of(bx), col_of(bx + bw)
        for c in range(c0, c1 + 1):
            paint(r0, c, "-", _GREEN)
            paint(r1, c, "-", _GREEN)
        for r in range(r0, r1 + 1):
            paint(r, c0, "|", _GREEN)
            paint(r, c1, "|", _GREEN)

    # packing zone
    if zone_px is not None:
        zx, zy, zw, zh = zone_px
        r0, r1 = row_of(zy), row_of(zy + zh)
        c0, c1 = col_of(zx), col_of(zx + zw)
        for c in range(c0, c1 + 1):
            for r in (r0, r1):
                if chars[r][c] == " ":
                    paint(r, c, ".", _TEAL)
        for r in range(r0, r1 + 1):
            for c in (c0, c1):
                if chars[r][c] == " ":
                    paint(r, c, ":", _TEAL)

    # ROI border
    if roi_px is not None:
        rx, ry, rw, rh = roi_px
        r0, r1 = row_of(ry), row_of(ry + rh - 1)
        c0, c1 = col_of(rx), col_of(rx + rw - 1)
        for c in range(c0, c1 + 1):
            for r in (r0, r1):
                if chars[r][c] == " ":
                    paint(r, c, "-", _BLUE)
        for r in range(r0, r1 + 1):
            for c in (c0, c1):
                if chars[r][c] == " ":
                    paint(r, c, "|", _BLUE)

    # counting line, drawn last so it is always visible
    if line is not None:
        axis, pos = line
        if axis == "y":
            r = row_of(pos)
            for c in range(cols):
                paint(r, c, "=", _AMBER)
        else:
            c = col_of(pos)
            for r in range(rows):
                paint(r, c, "!", _AMBER)

    out = []
    for r in range(rows):
        if not colour:
            out.append("".join(chars[r]))
            continue
        parts, cur = [], ""
        for c in range(cols):
            want = overlay[r][c]
            if want != cur:
                parts.append(_RESET if not want else want)
                cur = want
            parts.append(chars[r][c])
        parts.append(_RESET)
        out.append("".join(parts))
    return out


def legend(colour: bool = True) -> str:
    if not colour:
        return ("legend:  ' '=no movement  .:#@=movement   |-|=detected box   "
                "===counting line   :.:=packing zone   ---=ROI (only inside "
                "this is used)")
    return (f"{_DIM}legend:{_RESET} ' '=no movement  .:#@=movement   "
            f"{_GREEN}|-|{_RESET}=detected box   "
            f"{_AMBER}==={_RESET}=counting line   "
            f"{_TEAL}:.:{_RESET}=packing zone   "
            f"{_BLUE}---{_RESET}=ROI (only inside this is used)")
