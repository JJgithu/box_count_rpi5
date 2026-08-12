"""Generate a synthetic packing-station video for testing without hardware.

Simulates an indexed conveyor seen from above: each box enters, stops in the
packing zone, a packer's arm reaches in from the bottom edge and places a pad
(one reach per pad), and the box then departs across the counting line. The
belt scrolls only while boxes move, as on a real indexed line.

Prints the ground truth (boxes, pieces per box, pack seconds per box).

Usage:
    python3 tools/make_packing_video.py --out data/packing_test.mp4 --pieces 3 2 4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _belt(width: int, height: int, rng: random.Random) -> np.ndarray:
    strip = np.full((height * 3, width, 3), 108, np.uint8)
    nrng = np.random.default_rng(rng.randrange(2 ** 31))
    noise = nrng.normal(0, 4, strip.shape[:2])
    strip = np.clip(strip.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)
    for y in range(0, strip.shape[0], 160):
        cv2.line(strip, (0, y), (width, y), (98, 98, 98), 2)
    return strip


def _draw_box(frame, x, y, w, h, pads, rng_colors):
    """Open box with its current pads (pad rects are box-relative)."""
    b, g, r = rng_colors
    cv2.rectangle(frame, (x, y), (x + w, y + h), (b, g, r), -1)
    rim = max(8, w // 10)
    cv2.rectangle(frame, (x + rim, y + rim), (x + w - rim, y + h - rim),
                  (92, 96, 100), -1)                     # dark interior
    for (px, py, pw, ph) in pads:
        cv2.rectangle(frame, (x + px, y + py), (x + px + pw, y + py + ph),
                      (208, 214, 220), -1)               # light paper pad
        cv2.rectangle(frame, (x + px, y + py), (x + px + pw, y + py + ph),
                      (170, 176, 182), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (b - 30, g - 30, r - 30), 2)


def _draw_arm(frame, height, cx, tip_y, width_px=70, wobble=0):
    """Arm reaching up from the bottom edge. Drawn with smooth, low-contrast
    fill like real skin and sleeve: frame differencing inside a box that the
    hand fully covers then sees almost no motion, which is what a real hand
    looks like to the detector."""
    """Arm reaching up from the bottom edge to tip_y, hand at the tip."""
    x0 = int(cx - width_px / 2 + wobble)
    cv2.rectangle(frame, (x0, int(tip_y) + 18, ),
                  (x0 + width_px, height), (130, 90, 50), -1)      # sleeve
    cv2.ellipse(frame, (int(cx + wobble), int(tip_y) + 14), (int(width_px * 0.55), 26),
                0, 0, 360, (125, 160, 200), -1)                    # hand


def generate_packing_video(path: str, pieces_per_box=(3, 2), width: int = 640,
                           height: int = 480, fps: int = 24, speed: float = 6.0,
                           lead_in_s: float = 4.0, stop_frac: float = 0.30,
                           seed: int = 7, box_w=(140, 180), box_h=(110, 140),
                           gap_s: float = 0.42, hold_s: float = 0.25) -> dict:
    """Write the video; returns ground truth {boxes, pieces, pack_seconds}."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    belt = _belt(width, height, rng)
    belt_h = belt.shape[0]

    # per-reach phase lengths (frames)
    EXTEND, RETRACT = 5, 5
    # How long the hand rests over the box during a reach. A packer settling
    # a pad in place holds for a second or more, which is much longer than a
    # brisk synthetic reach and exercises the static-occupant logic.
    HOLD = max(2, int(hold_s * fps))
    # Pause between reaches. A real packer takes seconds between pads; the
    # belt and box are perfectly still during that pause, which is exactly
    # when a background model can absorb a stationary box.
    GAP = max(1, int(gap_s * fps))
    SETTLE_IN, SETTLE_OUT = 8, 12
    BOX_GAP = 24              # frames between one box leaving and next entering

    # Build a global frame script: list of (box_state, arm_state, belt_moves)
    # box_state: (box_index, y) or None;  arm_state: (box_index, tip_y) or None
    boxes = []
    for i, n in enumerate(pieces_per_box):
        w = rng.randint(*box_w)
        h = rng.randint(*box_h)
        x = rng.randint(int(width * 0.25), int(width * 0.65) - w)
        color = (70 + rng.randint(-8, 8), 110 + rng.randint(-8, 8),
                 150 + rng.randint(-8, 8))
        boxes.append({"w": w, "h": h, "x": x, "color": color, "pads": [],
                      "pieces": n})

    frames = []               # (box_idx|None, box_y, arm_tip|None, belt_step)
    lead = int(lead_in_s * fps)
    for _ in range(lead):
        frames.append((None, 0.0, None, 0.0))

    truth_pack_s = []
    for i, bx in enumerate(boxes):
        stop_y = stop_frac * height - bx["h"] / 2
        # approach
        y = -float(bx["h"])
        while y < stop_y:
            y = min(stop_y, y + speed)
            frames.append((i, y, None, speed))
        # dwell
        dwell_frames = SETTLE_IN
        for _ in range(SETTLE_IN):
            frames.append((i, stop_y, None, 0.0))
        interior_cy = stop_y + bx["h"] * 0.5
        for p in range(bx["pieces"]):
            for f in range(EXTEND):
                tip = height - (height - interior_cy) * (f + 1) / EXTEND
                frames.append((i, stop_y, tip, 0.0))
            # The hand rests over the box while the pad is placed: it covers
            # the interior and barely moves, which is the realistic case and
            # the one that used to blind arm detection for later reaches.
            rest_tip = stop_y - 45
            for f in range(HOLD):
                if f == HOLD // 2:      # pad appears mid-hold
                    pw = int(bx["w"] * 0.5) + rng.randint(-8, 8)
                    ph = int(bx["h"] * 0.4) + rng.randint(-6, 6)
                    px = rng.randint(int(bx["w"] * 0.15), max(int(bx["w"] * 0.15) + 1,
                                     bx["w"] - pw - int(bx["w"] * 0.15)))
                    py = rng.randint(int(bx["h"] * 0.15), max(int(bx["h"] * 0.15) + 1,
                                     bx["h"] - ph - int(bx["h"] * 0.15)))
                    bx["pads"].append((px, py, pw, ph))
                frames.append((i, stop_y, rest_tip + (f % 2), 0.0))
            for f in range(RETRACT):
                tip = interior_cy + (height - interior_cy) * (f + 1) / RETRACT
                frames.append((i, stop_y, None if f == RETRACT - 1 else tip, 0.0))
            for _ in range(GAP):
                frames.append((i, stop_y, None, 0.0))
            dwell_frames += EXTEND + HOLD + RETRACT + GAP
        for _ in range(SETTLE_OUT):
            frames.append((i, stop_y, None, 0.0))
        dwell_frames += SETTLE_OUT
        truth_pack_s.append(round(dwell_frames / fps, 2))
        # depart
        y = stop_y
        while y < height + 10:
            y += speed
            frames.append((i, y, None, speed))
        for _ in range(BOX_GAP):
            frames.append((None, 0.0, None, 0.0))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return _render(path, frames, boxes, belt, width, height, fps, nrng,
                   truth_pack_s, pieces_per_box, speed)


def _render(path, frames, boxes, belt, width, height, fps, nrng,
            truth_pack_s, pieces_per_box, speed):
    """Render pass. Pad appearance frames are recomputed deterministically:
    pads for box i appear in order at the HOLD//2 frame of each reach."""
    EXTEND, HOLD = 5, 6
    belt_h = belt.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")

    # find pad-appearance frame indices per box by replaying reach structure
    visible_pads = [0] * len(boxes)
    in_reach_frames = 0
    prev_key = None
    scroll = 0.0
    for fi, (bi, y, tip, step) in enumerate(frames):
        key = bi
        if bi is not None and tip is not None:
            in_reach_frames += 1
            if in_reach_frames == EXTEND + HOLD // 2 + 1:
                visible_pads[bi] = min(visible_pads[bi] + 1, len(boxes[bi]["pads"]))
        elif in_reach_frames:
            in_reach_frames = 0
        if prev_key is not None and key != prev_key:
            in_reach_frames = 0
        prev_key = key

        scroll = (scroll + step) % belt_h
        y_off = int(scroll)
        frame = np.empty((height, width, 3), np.uint8)
        rows = belt_h - y_off
        if rows >= height:
            frame[:] = belt[y_off:y_off + height]
        else:
            frame[:rows] = belt[y_off:]
            frame[rows:] = belt[:height - rows]

        if bi is not None:
            bx = boxes[bi]
            pads = bx["pads"][:visible_pads[bi]]
            _draw_box(frame, bx["x"], int(y), bx["w"], bx["h"], pads, bx["color"])
            if tip is not None:
                cx = bx["x"] + bx["w"] * 0.5
                arm_w = max(70, int(bx["w"] * 1.15))
                _draw_arm(frame, height, cx, tip, width_px=arm_w,
                          wobble=(fi % 3) - 1)

        noise = nrng.normal(0, 2.0, (height, width, 1))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()
    return {
        "boxes": len(boxes),
        "pieces": list(pieces_per_box),
        "pack_seconds": truth_pack_s,
        "frames": len(frames),
        "fps": fps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/packing_test.mp4")
    ap.add_argument("--pieces", type=int, nargs="+", default=[3, 2],
                    help="pieces per box, one number per box (default: 3 2)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--box-size", type=int, nargs=2, metavar=("W", "H"),
                    help="fixed box size in pixels (default: random 140-180 x 110-140)")
    ap.add_argument("--gap", type=float, default=0.42,
                    help="seconds the packer pauses between pads (default 0.42)")
    ap.add_argument("--hold", type=float, default=0.25,
                    help="seconds the hand rests in the box per pad (default 0.25)")
    args = ap.parse_args()
    kw = {}
    if args.box_size:
        kw = {"box_w": (args.box_size[0], args.box_size[0] + 1),
              "box_h": (args.box_size[1], args.box_size[1] + 1)}
    truth = generate_packing_video(args.out, pieces_per_box=tuple(args.pieces),
                                   fps=args.fps, speed=args.speed, seed=args.seed,
                                   gap_s=args.gap, hold_s=args.hold, **kw)
    print(f"Wrote {args.out}: {truth}")
    print("Try:  python3 -m boxcounter --source", args.out, "--no-web")
    print("(enable packing in the config first — see config/config.yaml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
