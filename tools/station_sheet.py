"""Turn a recorded clip into a few pictures that show what the counter saw.

    python3 -m boxcounter --record              # pack 2-3 boxes, then Ctrl-C
    python3 tools/station_sheet.py data/recordings/clip_<date>_<time>.mp4

Replays the clip through the counter exactly as it ran live (same settings
from config/config.yaml) and writes contact sheets to data/sheets/:
sheet_01.jpg, sheet_02.jpg, ... Each tile is one moment: on the left the
camera picture with what the counter drew on it (packing zone, pad outlet,
the box with its pads and time, HAND during a reach), on the right what the
counter "sees" (white = box, orange = something new, red = movement). Under
each tile: the time and what the counter was doing.

The counter's decisions (box arrived, every reach and why it did or did not
count, box counted) are printed and saved to data/sheets/decisions.txt.

The sheets are ordinary JPEG pictures: copy them to a PC/phone and send them
like a screenshot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter.annotate import draw_station_overlay     # noqa: E402
from boxcounter.config import load_config                # noqa: E402
from boxcounter.station import StationMonitor            # noqa: E402

TILE_W, TILE_H = 320, 240
COLS, ROWS = 2, 6                  # 12 moments per sheet


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: List[str] = []
        self.t = 0.0

    def emit(self, record):
        self.lines.append(f"{self.t:7.2f}s  {record.getMessage()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="a video recorded with python3 -m boxcounter --record")
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--every", type=float, default=1.0,
                    help="seconds between pictures (default 1.0)")
    ap.add_argument("--out", default=None, help="folder (default: data/sheets)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out or Path(cfg.output.data_dir) / "sheets")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"Cannot open {args.clip}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.camera.fps)

    collect = _Collect()
    st_log = logging.getLogger("boxcounter.station")
    st_log.setLevel(logging.INFO)
    st_log.addHandler(collect)
    st_log.propagate = False

    mon = None
    tiles: List[np.ndarray] = []
    sheets = 0
    total = 0
    next_t = 0.0
    i = 0

    def flush(final: bool = False) -> None:
        nonlocal sheets, tiles
        per = COLS * ROWS
        while len(tiles) >= per or (final and tiles):
            chunk, tiles = tiles[:per], tiles[per:]
            while len(chunk) < per:
                chunk.append(np.zeros_like(chunk[0]))
            rows = [np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
            sheets += 1
            cv2.imwrite(str(out_dir / f"sheet_{sheets:02d}.jpg"), np.vstack(rows),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        t = i / fps
        collect.t = t
        if mon is None:
            mon = StationMonitor(cfg.station, cfg.processing.roi, frame.shape[:2],
                                 debug=True)
        total += len(mon.update(frame, t))
        if t + 1e-9 < next_t:
            continue
        next_t = t + args.every
        state = mon.state_text()
        view = draw_station_overlay(frame, mon.overlay(), total, fps, 0.0, state)
        dbg = mon.debug_view()
        if dbg is None:
            dbg = np.zeros_like(frame)
        left = cv2.resize(view, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        right = cv2.resize(dbg, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
        tile = np.hstack([left, right])
        caption = np.zeros((22, tile.shape[1], 3), np.uint8)
        text = f"{t:6.1f}s  " + state.replace("—", "-")
        cv2.putText(caption, text.encode("ascii", "replace").decode()[:95], (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, caption]))
        flush()
    flush(final=True)
    cap.release()

    text = "\n".join(collect.lines)
    (out_dir / "decisions.txt").write_text(text + "\n")
    print(text)
    print(f"\nBoxes counted in this clip: {total}")
    print(f"{sheets} picture(s) in {out_dir}/ (sheet_01.jpg ...), "
          f"decisions in {out_dir}/decisions.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
