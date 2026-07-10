"""Generate a synthetic conveyor video for testing the counter without hardware.

Simulates a moving textured belt seen from above, with closed boxes (solid
cardboard tops) and open boxes (thin rim, interior close to belt gray — the
hard case that fragments the foreground mask). Prints the ground-truth count.

Usage:
    python3 tools/make_test_video.py --out data/test.mp4 --boxes 12
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _belt_texture(width: int, height: int, rng: random.Random) -> np.ndarray:
    """A tall tileable belt strip with subtle texture and seam lines."""
    strip = np.full((height * 3, width, 3), 108, np.uint8)
    nrng = np.random.default_rng(rng.randrange(2 ** 31))
    noise = nrng.normal(0, 4, strip.shape[:2])
    strip = np.clip(strip.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)
    for y in range(0, strip.shape[0], 160):   # faint transverse seams
        cv2.line(strip, (0, y), (width, y), (98, 98, 98), 2)
    return strip


def _draw_box(frame: np.ndarray, x: int, y: int, w: int, h: int,
              is_open: bool, rng: random.Random) -> None:
    """Draw a cardboard box (top view) with its top-left corner at (x, y)."""
    # cardboard brown in BGR, slightly varied per box
    b = 70 + rng.randint(-10, 10)
    g = 110 + rng.randint(-10, 10)
    r = 150 + rng.randint(-10, 10)
    pts = ((x, y), (x + w, y + h))
    cv2.rectangle(frame, pts[0], pts[1], (b, g, r), -1)
    if is_open:
        # Interior close to belt gray — makes the mask fragment into a rim,
        # which is exactly what morphological closing must repair.
        rim = max(6, w // 10)
        cv2.rectangle(frame, (x + rim, y + rim), (x + w - rim, y + h - rim),
                      (104, 106, 108), -1)
        # a hint of inner shadow along one flap
        cv2.rectangle(frame, (x + rim, y + rim), (x + w - rim, y + rim + 6),
                      (60, 60, 62), -1)
    else:
        # tape stripe down the middle of a closed box
        cv2.rectangle(frame, (x + w // 2 - 5, y), (x + w // 2 + 5, y + h),
                      (b + 30, g + 30, r + 30), -1)
    cv2.rectangle(frame, pts[0], pts[1], (max(0, b - 30), max(0, g - 30), max(0, r - 30)), 2)


def generate_video(path: str, n_boxes: int = 12, width: int = 640, height: int = 480,
                   fps: int = 24, speed_px: float = 6.0, lead_in_s: float = 4.0,
                   open_ratio: float = 0.5, seed: int = 42) -> int:
    """Write a synthetic conveyor video; returns the ground-truth box count."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    belt = _belt_texture(width, height, rng)
    belt_h = belt.shape[0]

    # Schedule boxes: start y (above the frame) so they enter one after
    # another with a random gap; travel is downward (+y).
    boxes = []
    next_start = -140.0
    for i in range(n_boxes):
        w = rng.randint(110, 190)
        h = rng.randint(90, 160)
        x = rng.randint(40, width - 40 - w)
        boxes.append({"x": x, "y0": next_start - h, "w": w, "h": h,
                      "open": rng.random() < open_ratio})
        next_start -= h + rng.randint(60, 200)   # gap between boxes

    # Total frames: lead-in + time for the last box to fully exit.
    travel_needed = abs(boxes[-1]["y0"]) + height + 80
    frames = int(lead_in_s * fps) + int(travel_needed / speed_px) + fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")

    lead_frames = int(lead_in_s * fps)
    scroll = 0.0
    for f in range(frames):
        # scrolling belt background
        scroll = (scroll + speed_px) % belt_h
        y_off = int(scroll)
        frame = np.empty((height, width, 3), np.uint8)
        rows = belt_h - y_off
        if rows >= height:
            frame[:] = belt[y_off:y_off + height]
        else:
            frame[:rows] = belt[y_off:]
            frame[rows:] = belt[:height - rows]

        # boxes move only after the lead-in (empty belt for model warm-up)
        if f >= lead_frames:
            dist = (f - lead_frames) * speed_px
            for bx in boxes:
                y = int(bx["y0"] + dist)
                if -bx["h"] - 10 < y < height + 10:
                    _draw_box(frame, bx["x"], y, bx["w"], bx["h"], bx["open"], rng)

        # per-frame sensor noise
        noise = nrng.normal(0, 2.0, (height, width, 1))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)

    out.release()
    return n_boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/test.mp4")
    ap.add_argument("--boxes", type=int, default=12)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--speed", type=float, default=6.0, help="belt speed px/frame")
    ap.add_argument("--open-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = generate_video(args.out, n_boxes=args.boxes, width=args.width,
                       height=args.height, fps=args.fps, speed_px=args.speed,
                       open_ratio=args.open_ratio, seed=args.seed)
    print(f"Wrote {args.out} with ground truth = {n} boxes")
    print(f"Try:  python3 -m boxcounter --config config/config.yaml --source {args.out} --no-web")
    return 0


if __name__ == "__main__":
    sys.exit(main())
