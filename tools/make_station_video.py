"""Generate a synthetic packing-station video: roller conveyor + paper machine.

Seen from a camera above the paper machine, looking down at the packing spot:
the machine's front with its pad outlet along the top, the roller conveyor
across the middle, the packer standing at the bottom edge.

Each box slides in along the rollers and stops under the outlet. For every
pad the packer reaches up, tears off the pad hanging from the outlet, stuffs
it into the box and pulls the hand back, while the machine pushes out the
next pad. Then the box slides out the other side. Before the first box the
packer rests a hand on the empty rollers for a few seconds and later waves
a hand across the view — neither may be counted as a box.

Prints the ground truth (boxes, pads per box, pack seconds per box).

    python3 tools/make_station_video.py --out data/station_test.mp4 --pads 3 2 4
    python3 -m boxcounter --source data/station_test.mp4 --no-web
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H = 640, 480
OUTLET_X, OUTLET_Y = 320, 52          # where pads come out of the machine
ROLLER_TOP, ROLLER_BOTTOM = 70, 410
BOX_TOP = 118                         # top edge of the box body when parked


def _background(rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    # machine front: light grey plastic, outlet slot
    for y in range(0, 56):
        img[y, :] = 205 - 0.4 * y
    img[40:56, OUTLET_X - 70:OUTLET_X + 70] = (38, 38, 42)
    # conveyor rails (blue) and floor seen between rollers
    img[56:ROLLER_TOP] = (165, 85, 30)
    img[ROLLER_TOP:ROLLER_BOTTOM] = (56, 58, 60)
    img[ROLLER_BOTTOM:ROLLER_BOTTOM + 14] = (165, 85, 30)
    img[ROLLER_BOTTOM + 14:] = (118, 121, 123)
    # rollers: vertical metal cylinders, bright centre, darker edges
    pitch, width = 34, 26
    for x0 in range(-10, W, pitch):
        for dx in range(width):
            x = x0 + dx
            if 0 <= x < W:
                shade = 120 + 100 * np.sin(np.pi * (dx + 0.5) / width)
                img[ROLLER_TOP:ROLLER_BOTTOM, x] = (shade, shade, shade + 4)
    # the packer standing at the bottom edge (black clothes)
    img[452:, 210:430] = (30, 30, 33)
    img += rng.normal(0, 3, (H, W, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _pad_texture(w: int, h: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Crumpled kraft paper: brown, full of light and dark facets."""
    base = np.array([96, 140, 182], np.float32) + rng.normal(0, 5, 3)
    tex = np.ones((h, w, 3), np.float32) * base
    for _ in range(int(w * h / 60)):
        pts = rng.integers(0, [w, h], size=(3, 2)).astype(np.int32)
        k = float(rng.uniform(-55, 45))
        cv2.fillConvexPoly(tex, pts, (base + k).tolist())
    for _ in range(int((w + h) / 12)):
        p0 = rng.integers(0, [w, h])
        p1 = p0 + rng.integers(-25, 26, size=2)
        cv2.line(tex, tuple(int(v) for v in p0), tuple(int(v) for v in p1),
                 (base * 0.55).tolist(), 1)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    # irregular outline
    mask = np.zeros((h, w), np.uint8)
    n = 14
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = rng.uniform(0.80, 1.0, n)
    pts = np.stack([w / 2 + (w / 2 - 1) * rad * np.cos(ang),
                    h / 2 + (h / 2 - 1) * rad * np.sin(ang)], axis=1).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return np.clip(tex, 0, 255).astype(np.uint8), mask > 0


def _paste(frame: np.ndarray, tex: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    h, w = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub_m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    frame[y0:y1, x0:x1][sub_m] = tex[y0 - y:y1 - y, x0 - x:x1 - x][sub_m]


def _draw_box(frame: np.ndarray, x: int, bw: int, bh: int, color, contents) -> None:
    """Open box from above with its flaps spread, at body-left x."""
    y = BOX_TOP
    b, g, r = color
    flap = (min(255, b + 14), min(255, g + 16), min(255, r + 16))
    ft, fs = 26, 22
    polys = [
        [(x + 6, y - ft), (x + bw - 6, y - ft), (x + bw, y), (x, y)],             # far
        [(x, y + bh), (x + bw, y + bh), (x + bw - 6, y + bh + ft), (x + 6, y + bh + ft)],
        [(x - fs, y + 6), (x, y), (x, y + bh), (x - fs, y + bh - 6)],             # left
        [(x + bw, y), (x + bw + fs, y + 6), (x + bw + fs, y + bh - 6), (x + bw, y + bh)],
    ]
    for p in polys:
        cv2.fillConvexPoly(frame, np.array(p, np.int32), flap)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (b, g, r), -1)
    rim = 7
    inner = (max(0, b - 40), max(0, g - 52), max(0, r - 62))     # shaded inside
    cv2.rectangle(frame, (x + rim, y + rim), (x + bw - rim, y + bh - rim), inner, -1)
    cv2.rectangle(frame, (x + rim, y + rim), (x + bw - rim, y + rim + 14),
                  (max(0, b - 20), max(0, g - 26), max(0, r - 30)), -1)  # lit wall
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (b + 10, g + 12, r + 12), 2)
    for (tex, mask, px, py) in contents:
        _paste(frame, tex, mask, x + px, y + py)


def _draw_arm(frame: np.ndarray, hand: Tuple[float, float], base_x: int) -> None:
    hx, hy = int(hand[0]), int(hand[1])
    cv2.line(frame, (base_x, H + 40), (hx, hy + 30), (44, 40, 38), 46)     # sleeve
    cv2.ellipse(frame, (hx, hy + 6), (25, 30), 0, 0, 360, (112, 148, 196), -1)  # hand
    cv2.ellipse(frame, (hx - 20, hy + 14), (8, 14), 30, 0, 360, (108, 142, 190), -1)


def _ease(a: float, b: float, f: float) -> float:
    f = max(0.0, min(1.0, f))
    f = f * f * (3 - 2 * f)
    return a + (b - a) * f


def generate_station_video(path: str, pads_per_box=(3, 2), fps: int = 20,
                           seed: int = 11, box_size=(200, 150),
                           gap_s: float = 0.9, rest_hand: bool = True,
                           box_at_start: int = 0, pre_hung: bool = True,
                           empty_reach: bool = False, hold_s: float = 0.3,
                           nudge_px: int = 0, light_step: float = 0.0,
                           shadow: bool = False, noise_sigma: float = 2.0) -> dict:
    """Write the video; return the ground truth.

    box_at_start: if > 0, a box is already standing on the spot when the
    video starts (it becomes part of the learned background), receives this
    many pads and leaves; it is NOT part of the ground truth.
    empty_reach: after its pads, the packer reaches into each box once more
    without a pad (pats the paper down) — must not count.
    hold_s: how long the hand stays in the box after letting go of a pad.
    nudge_px: shove each box sideways by this much before its last pad.
    light_step: from the middle of the video on, the whole picture is this
    much brighter (0.1 = +10%: lights switched, a door opened).
    shadow: the packer's shadow follows the hand over the box.
    """
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    bg = _background(rng)
    noise = [rng.normal(0, noise_sigma, (H, W, 1)).astype(np.int16) for _ in range(8)]

    bw, bh = box_size
    park_x = OUTLET_X - bw // 2
    hang_w, hang_h = 96, 118

    def new_hanging():
        return _pad_texture(hang_w, hang_h, rng)

    S = lambda sec: max(1, int(round(sec * fps)))          # noqa: E731

    # Per-frame state, filled in by the script below.
    frames: List[dict] = []
    state = {
        "box": None,            # dict(x, color, contents) or None
        "hang": new_hanging() if pre_hung else None,
        "hang_len": hang_h if pre_hung else 0,
        "hand": None,           # (x, y) or None
        "carry": None,          # (tex, mask) carried pad
        "base_x": 380,
    }

    def emit(n: int, fn=None):
        for i in range(n):
            if fn is not None:
                fn(i, n)
            frames.append({
                "box": None if state["box"] is None else dict(
                    x=state["box"]["x"], color=state["box"]["color"],
                    contents=list(state["box"]["contents"])),
                "hang": state["hang"], "hang_len": state["hang_len"],
                "hand": state["hand"], "carry": state["carry"],
                "base_x": state["base_x"],
            })

    pending_emerge = {"delay": 0, "grow": 0}

    def machine_tick():
        """The machine pushes out a new pad a moment after one is torn off."""
        if state["hang"] is None and pending_emerge["delay"] > 0:
            pending_emerge["delay"] -= 1
            if pending_emerge["delay"] == 0:
                state["hang"] = new_hanging()
                state["hang_len"] = 0
        elif state["hang"] is not None and state["hang_len"] < hang_h:
            state["hang_len"] = min(hang_h, state["hang_len"] + hang_h / max(1, S(0.5)))

    def wait(sec: float):
        emit(S(sec), lambda i, n: machine_tick())

    def move_hand(p0, p1, sec: float):
        def f(i, n):
            machine_tick()
            state["hand"] = (_ease(p0[0], p1[0], (i + 1) / n),
                             _ease(p0[1], p1[1], (i + 1) / n) + (i % 2))
        emit(S(sec), f)

    def slide_box(x0, x1, sec: float):
        def f(i, n):
            machine_tick()
            state["box"]["x"] = int(_ease(x0, x1, (i + 1) / n))
        emit(S(sec), f)

    def pack_pad(box):
        rest = (state["base_x"], H + 30)
        grab = (OUTLET_X + prng.randint(-6, 6), 118)
        move_hand(rest, grab, 0.5)
        wait_hand = S(0.25)
        emit(wait_hand, lambda i, n: machine_tick())
        # tear it off: the outlet is empty for a moment, then the next pad
        state["hang"] = None
        state["hang_len"] = 0
        pending_emerge["delay"] = S(0.35)
        tex, mask = _pad_texture(prng.randint(88, 112), prng.randint(64, 84), rng)
        state["carry"] = (tex, mask)
        tx = prng.randint(14, bw - 14 - mask.shape[1])
        ty = prng.randint(14, bh - 14 - mask.shape[0])
        target = (box["x"] + tx + mask.shape[1] / 2, BOX_TOP + ty + mask.shape[0] / 2 - 10)
        move_hand(grab, target, 0.5)
        state["carry"] = None
        box["contents"].append((tex, mask, tx, ty))
        emit(S(hold_s), lambda i, n: (machine_tick(), state.__setitem__(
            "hand", (target[0] + (i % 3) - 1, target[1]))))
        move_hand(target, rest, 0.45)
        state["hand"] = None

    def run_box(n_pads: int, slide_in: bool):
        color = (70 + prng.randint(-6, 6), 108 + prng.randint(-6, 6),
                 148 + prng.randint(-6, 6))
        box = {"x": -bw - 40, "color": color, "contents": []}
        state["box"] = box
        t_in = len(frames)
        if slide_in:
            slide_box(-bw - 40, park_x, 1.0)
        else:
            box["x"] = park_x
        t_half_in = t_in + S(0.5) if slide_in else t_in
        wait(0.8)
        for k in range(n_pads):
            if nudge_px and k == n_pads - 1:
                slide_box(park_x, park_x + nudge_px, 0.3)
                wait(0.4)
            pack_pad(box)
            wait(gap_s)
        if empty_reach:
            rest = (state["base_x"], H + 30)
            spot = (box["x"] + bw * 0.5, BOX_TOP + bh * 0.55)
            move_hand(rest, spot, 0.5)
            emit(S(0.6), lambda i, n: (machine_tick(), state.__setitem__(
                "hand", (spot[0] + 3 * ((i // 2) % 3 - 1), spot[1]))))
            move_hand(spot, rest, 0.45)
            state["hand"] = None
            wait(gap_s)
        wait(0.8)
        x_now = box["x"]
        t_out = len(frames)
        slide_box(x_now, W + 40, 1.0)
        t_half_out = t_out + S(0.5)
        state["box"] = None
        return (t_half_out - t_half_in) / fps

    truth_pads: List[int] = []
    truth_secs: List[float] = []

    if box_at_start:
        run_box(box_at_start, slide_in=False)       # in the learned background
        wait(2.0)
    else:
        wait(2.5)                                   # empty spot: learning

    if rest_hand:
        # The packer rests a hand on the empty rollers — not a box.
        spot = (OUTLET_X - 20, 300)
        move_hand((state["base_x"], H + 30), spot, 0.6)
        emit(S(3.0), lambda i, n: (machine_tick(), state.__setitem__("hand", spot)))
        move_hand(spot, (state["base_x"], H + 30), 0.5)
        state["hand"] = None
        wait(1.5)

    for i, n in enumerate(pads_per_box):
        truth_pads.append(n)
        truth_secs.append(round(run_box(n, slide_in=True), 2))
        wait(1.5)
        if i == 0 and rest_hand:
            # a hand waved across the empty spot
            move_hand((40, 330), (600, 300), 0.8)
            state["hand"] = None
            wait(1.5)
    wait(4.0)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")
    for fi, st in enumerate(frames):
        frame = bg.copy()
        if st["box"] is not None:
            b = st["box"]
            _draw_box(frame, b["x"], bw, bh, b["color"], b["contents"])
        if st["hang"] is not None and st["hang_len"] > 0:
            tex, mask = st["hang"]
            L = int(st["hang_len"])
            # a pad sways briefly now and then (draughts, the packer brushing it)
            sway = ((fi // 3) % 3 - 1) if (fi % 80) < 8 else 0
            _paste(frame, tex[:L], mask[:L], OUTLET_X - hang_w // 2 + sway, OUTLET_Y)
        if st["hand"] is not None and shadow:
            hx, hy = st["hand"]
            sh = np.zeros((H, W), np.float32)
            cv2.ellipse(sh, (int(hx + 40), int(hy + 70)), (90, 120), 0, 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), 25)
            frame = (frame * (1.0 - 0.3 * sh[..., None])).astype(np.uint8)
        if st["hand"] is not None:
            _draw_arm(frame, st["hand"], st["base_x"])
            if st["carry"] is not None:
                tex, mask = st["carry"]
                hx, hy = st["hand"]
                _paste(frame, tex, mask, int(hx - mask.shape[1] / 2),
                       int(hy - mask.shape[0] / 2 + 12))
        gain = 1.0 + rng.uniform(-0.01, 0.01)
        if light_step and fi >= len(frames) // 2:
            gain *= 1.0 + light_step
        f = frame.astype(np.int16) * gain + noise[fi % len(noise)]
        out.write(np.clip(f, 0, 255).astype(np.uint8))
    out.release()
    return {"boxes": len(truth_pads), "pads": truth_pads,
            "pack_seconds": truth_secs, "frames": len(frames), "fps": fps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/station_test.mp4")
    ap.add_argument("--pads", type=int, nargs="+", default=[3, 2],
                    help="pads per box, one number per box (default: 3 2)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gap", type=float, default=0.9,
                    help="seconds between pads (default 0.9)")
    ap.add_argument("--box-size", type=int, nargs=2, metavar=("W", "H"),
                    default=[200, 150])
    ap.add_argument("--no-rest-hand", action="store_true",
                    help="leave out the resting / waving hand")
    ap.add_argument("--box-at-start", type=int, default=0, metavar="PADS",
                    help="start with a box already on the spot (not counted)")
    args = ap.parse_args()
    truth = generate_station_video(args.out, pads_per_box=tuple(args.pads),
                                   fps=args.fps, seed=args.seed, gap_s=args.gap,
                                   box_size=tuple(args.box_size),
                                   rest_hand=not args.no_rest_hand,
                                   box_at_start=args.box_at_start)
    print(f"Wrote {args.out}: {truth}")
    print(f"Try:  python3 -m boxcounter --source {args.out} --no-web")
    return 0


if __name__ == "__main__":
    sys.exit(main())
