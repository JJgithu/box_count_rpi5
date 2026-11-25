"""Synthetic packing station modelled on the real one (Storopack PAPERplus
over a roller conveyor), as the camera on the arm above the machine sees it.

What makes the real station hard, and what this scene has that the simpler
tools/make_station_video.py does not:

- The pad hanging from the machine is BIG: nearly half the box wide, and it
  reaches from the outlet over the far edge of the box deep INTO the box.
  From the camera it hides part of the box, and every new pad hangs a bit
  differently (length, sideways, crumple).
- The pad is not carried from the machine: the packer grabs the pad that
  already hangs in the box, pulls it off and stuffs it down. A moment later
  the machine pushes out the next pad, which slides down into the box by
  itself, with no hand anywhere near.
- The packer stands right against the near edge of the box, in dark clothes:
  a big dark shape that touches the box in the picture, shifts its weight
  all the time, and reaches in with an arm.
- A box sliding in or out drags the hanging pad along for a moment.
- The machine's touch screen changes when a pad is made.

    python3 tools/make_real_station_video.py --out data/real_test.mp4 --pads 3 2 4

Prints the ground truth (boxes, pads per box, pack seconds per box) and
writes the three setup pictures next to the video (empty spot, empty box,
box with the next pad hanging), as tools/station_setup.py would take them.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_station_video import _ease, _pad_texture, _paste  # noqa: E402

W, H = 640, 480
MACHINE_BOTTOM = 104                 # front edge of the machine housing
OUTLET_X, OUTLET_Y = 326, 96         # where the pad comes out
ROLLER_TOP, ROLLER_BOTTOM = 104, 424
BOX_TOP = 150                        # far edge of the box body when parked
BODY_TOP = 408                       # the packer's clothes start here


def _background(rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    # concrete floor around everything
    img[:] = (128, 130, 131)
    blotch = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 30)
    img += (blotch * 60)[..., None]
    # roller conveyor: floor seen between the rollers, blue rails
    img[ROLLER_TOP:ROLLER_BOTTOM] = (60, 62, 64)
    pitch, width = 36, 27
    for x0 in range(-12, W, pitch):
        for dx in range(width):
            x = x0 + dx
            if 0 <= x < W:
                shade = 118 + 105 * np.sin(np.pi * (dx + 0.5) / width)
                img[ROLLER_TOP:ROLLER_BOTTOM, x] = (shade, shade, shade + 4)
    img[ROLLER_BOTTOM:ROLLER_BOTTOM + 18] = (170, 88, 32)
    # the machine: silver housing over the far rail, touch screen, outlet slot
    for y in range(0, MACHINE_BOTTOM):
        img[y, 150:510] = 198 - 0.25 * y
    img[MACHINE_BOTTOM - 16:MACHINE_BOTTOM, OUTLET_X - 78:OUTLET_X + 78] = (40, 40, 44)
    img[14:66, 258:398] = (150, 70, 30)            # screen frame (blue)
    img[20:60, 266:390] = (232, 232, 228)
    img += rng.normal(0, 3, (H, W, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_screen(frame: np.ndarray, count: int) -> None:
    """The touch screen shows a number that changes with every pad."""
    for i in range(3):
        on = (count >> i) & 1
        x = 276 + 38 * i
        cv2.rectangle(frame, (x, 28), (x + 26, 52),
                      (150, 70, 30) if on else (205, 205, 200), -1)


def _hang_texture(w: int, h: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A pad as it hangs from the machine: full width where it leaves the
    outlet, ragged at its free end."""
    tex, mask = _pad_texture(w, h, rng)
    top = np.zeros_like(mask)
    top[:int(0.55 * h), int(0.06 * w):int(0.94 * w)] = True
    return tex, mask | top


def _draw_box(frame: np.ndarray, x: int, bw: int, bh: int, color, contents) -> None:
    """Open box from above, flaps spread, body's left edge at x."""
    y = BOX_TOP
    b, g, r = color
    flap = (min(255, b + 14), min(255, g + 16), min(255, r + 16))
    ft, fs = 26, 22
    polys = [
        [(x + 6, y - ft), (x + bw - 6, y - ft), (x + bw, y), (x, y)],
        [(x, y + bh), (x + bw, y + bh), (x + bw - 6, y + bh + ft), (x + 6, y + bh + ft)],
        [(x - fs, y + 6), (x, y), (x, y + bh), (x - fs, y + bh - 6)],
        [(x + bw, y), (x + bw + fs, y + 6), (x + bw + fs, y + bh - 6), (x + bw, y + bh)],
    ]
    for p in polys:
        cv2.fillConvexPoly(frame, np.array(p, np.int32), flap)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (b, g, r), -1)
    rim = 8
    inner = (max(0, b - 34), max(0, g - 44), max(0, r - 52))       # shaded inside
    cv2.rectangle(frame, (x + rim, y + rim), (x + bw - rim, y + bh - rim), inner, -1)
    cv2.rectangle(frame, (x + rim, y + rim), (x + bw - rim, y + rim + 22),
                  (max(0, b - 16), max(0, g - 20), max(0, r - 24)), -1)   # lit far wall
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (b + 10, g + 12, r + 12), 2)
    for (tex, mask, px, py) in contents:
        _paste(frame, tex, mask, x + px, y + py)


def _draw_body(frame: np.ndarray, cx: float, top: float) -> None:
    """The packer seen from above: dark shirt and trousers at the bottom edge."""
    cx, top = int(cx), int(top)
    cv2.ellipse(frame, (cx, top + 110), (128, 112), 0, 0, 360, (31, 30, 34), -1)
    cv2.ellipse(frame, (cx, top + 120), (96, 100), 0, 0, 360, (38, 36, 40), -1)


def _draw_arm(frame: np.ndarray, hand: Tuple[float, float], shoulder: Tuple[float, float]) -> None:
    hx, hy = int(hand[0]), int(hand[1])
    cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (hx, hy + 26), (42, 38, 36), 44)
    cv2.ellipse(frame, (hx, hy + 6), (24, 28), 0, 0, 360, (112, 148, 196), -1)
    cv2.ellipse(frame, (hx - 19, hy + 13), (8, 13), 30, 0, 360, (108, 142, 190), -1)


def _render(bg: np.ndarray, st: dict, bw: int, bh: int, shoulder_dx: int = 46) -> np.ndarray:
    frame = bg.copy()
    _draw_screen(frame, st["screen"])
    if st["box"] is not None:
        b = st["box"]
        _draw_box(frame, b["x"], bw, bh, b["color"], b["contents"])
    hang = st["hang"]
    if hang is not None and hang["len"] > 0:
        tex, mask = hang["tex"], hang["mask"]
        full = mask.shape[0]
        L = int(min(full, hang["len"]))
        # the paper slides out of the machine: its leading end is the bottom
        # of the texture, which moves down as more comes out
        _paste(frame, tex[full - L:], mask[full - L:],
               int(OUTLET_X - mask.shape[1] // 2 + hang["dx"]), OUTLET_Y)
    if st["body"] is not None:
        _draw_body(frame, st["body"][0], st["body"][1])
    if st["hand"] is not None:
        bx = st["body"][0] if st["body"] is not None else OUTLET_X
        _draw_arm(frame, st["hand"], (bx + shoulder_dx, BODY_TOP + 60))
        if st["carry"] is not None:
            tex, mask = st["carry"]
            hx, hy = st["hand"]
            _paste(frame, tex, mask, int(hx - mask.shape[1] / 2), int(hy - mask.shape[0] / 2 + 10))
    return frame


def generate_real_station_video(path: Optional[str], pads_per_box=(3, 2), fps: int = 20,
                                seed: int = 11, box_size=(230, 250),
                                gap_s: float = 0.9, settle_s: float = 0.8,
                                end_s: float = 0.8, hold_s: float = 0.3,
                                box_gap_s: float = 1.5, emerge_delay_s: float = 0.4,
                                emerge_s: float = 0.6, packer: bool = True,
                                packer_at_start: bool = False,
                                packer_leaves: bool = False,
                                rearrange: bool = False, grab_high: bool = False,
                                brush: bool = False, stay_hand: bool = False,
                                noise_sigma: float = 2.0,
                                lead_s: float = 2.5, setup_dir: Optional[str] = None
                                ) -> dict:
    """Write the video (path None: only the setup pictures); return the truth.

    gap_s: pause between pads (hand out of the box). settle_s / end_s: pause
    after the box stops / before it is pushed on. box_gap_s: the spot stays
    empty this long between boxes. emerge_delay_s / emerge_s: how long after
    a pad is torn off the next one starts coming, and how long it takes.
    packer: the packer's body is in view while packing. packer_at_start: it
    is already there while the counter learns the empty spot. packer_leaves:
    the packer steps out of view between boxes. rearrange: after its pads,
    the packer reaches into each box once more and pushes the last pad to
    another place — not a pad. grab_high: the pad is grabbed right below the
    outlet (the hand covers the paper there) instead of near its free end.
    brush: once per box the packer pushes the hanging pad aside without
    taking it — not a pad. stay_hand: the hand never leaves the box between
    pads (it waits there for the machine).
    """
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    bg = _background(rng)
    noise = [rng.normal(0, noise_sigma, (H, W, 1)).astype(np.int16) for _ in range(8)]
    bw, bh = box_size
    park_x = OUTLET_X - bw // 2 - 6

    def S(sec: float) -> int:
        return max(1, int(round(sec * fps)))

    def new_hang() -> dict:
        w = prng.randint(94, 114)
        full = prng.randint(int(0.50 * bh) + 54, int(0.72 * bh) + 54)   # outlet .. into the box
        tex, mask = _hang_texture(w, full, rng)
        return {"tex": tex, "mask": mask, "len": 0.0, "dx": float(prng.randint(-7, 7))}

    def full_len(h: dict) -> int:
        return h["mask"].shape[0]

    first = new_hang()
    first["len"] = float(full_len(first))
    state = {"box": None, "hang": first, "hand": None, "carry": None,
             "body": (OUTLET_X + 4.0, BODY_TOP) if (packer and packer_at_start) else None,
             "screen": 0}
    frames: List[dict] = []
    emerge = {"delay": 0}
    body = {"x": OUTLET_X + 4.0, "target": OUTLET_X + 4.0, "phase": 0.0}

    def snap() -> dict:
        h = state["hang"]
        return {
            "box": None if state["box"] is None else dict(
                x=state["box"]["x"], color=state["box"]["color"],
                contents=list(state["box"]["contents"])),
            "hang": None if h is None else dict(tex=h["tex"], mask=h["mask"],
                                                len=h["len"], dx=h["dx"]),
            "hand": state["hand"], "carry": state["carry"],
            "body": state["body"], "screen": state["screen"],
        }

    def tick() -> None:
        """Things that go on by themselves: the machine, the packer's sway."""
        h = state["hang"]
        if h is None:
            if emerge["delay"] > 0:
                emerge["delay"] -= 1
                if emerge["delay"] == 0:
                    state["hang"] = new_hang()
                    state["screen"] += 1
        elif h["len"] < full_len(h):
            h["len"] = min(float(full_len(h)), h["len"] + full_len(h) / max(1, S(emerge_s)))
        if state["body"] is not None:
            body["phase"] += 1.0 / fps
            if prng.random() < 0.25 / fps:              # shifts weight now and then
                body["target"] = OUTLET_X + 4.0 + prng.uniform(-14, 14)
            body["x"] += 0.08 * (body["target"] - body["x"])
            sway = 2.5 * np.sin(2 * np.pi * body["phase"] / 2.3)
            state["body"] = (body["x"] + sway, state["body"][1])

    def emit(n: int, fn=None) -> None:
        for i in range(n):
            tick()
            if fn is not None:
                fn(i, n)
            frames.append(snap())

    def wait(sec: float) -> None:
        emit(S(sec))

    def rest_point() -> Tuple[float, float]:
        return (body["x"] + 30.0, BODY_TOP + 40.0)

    def move_hand(p0, p1, sec: float) -> None:
        def f(i, n):
            state["hand"] = (_ease(p0[0], p1[0], (i + 1) / n),
                             _ease(p0[1], p1[1], (i + 1) / n) + (i % 2))
        emit(S(sec), f)

    def step_in() -> None:
        def f(i, n):
            state["body"] = (body["x"], _ease(H + 20, BODY_TOP, (i + 1) / n))
        state["body"] = (body["x"], H + 20.0)
        emit(S(0.6), f)

    def step_out() -> None:
        def f(i, n):
            state["body"] = (body["x"], _ease(BODY_TOP, H + 20, (i + 1) / n))
        emit(S(0.6), f)
        state["body"] = None

    def slide_box(x0: float, x1: float, sec: float, drag: float) -> None:
        """The box slides along the rollers; its wall drags the hanging pad
        along for a moment, and the pad ends up hanging a little differently."""
        h = state["hang"]
        dx0 = h["dx"] if h is not None else 0.0
        dx1 = dx0 + prng.uniform(-4, 4)

        def f(i, n):
            state["box"]["x"] = int(_ease(x0, x1, (i + 1) / n))
            hh = state["hang"]
            if hh is not None:
                t = (i + 1) / n
                bump = drag * np.sin(np.pi * min(1.0, t / 0.6)) if t < 0.6 else 0.0
                hh["dx"] = dx0 + (dx1 - dx0) * t + bump
        emit(S(sec), f)

    def pack_pad(box: dict, last: bool = True) -> None:
        h = state["hang"]
        while h is None or h["len"] < full_len(h):          # wait for the machine
            emit(1)
            h = state["hang"]
        tip_y = OUTLET_Y + full_len(h)
        grab = (OUTLET_X + h["dx"] + prng.randint(-8, 8),
                OUTLET_Y + 44.0 if grab_high else tip_y - 46.0)
        start = state["hand"] if state["hand"] is not None else rest_point()
        move_hand(start, grab, 0.45)
        emit(S(0.2), lambda i, n: state.__setitem__("hand", (grab[0] + (i % 2), grab[1])))
        # tear it off and pull it down into the box
        state["hang"] = None
        emerge["delay"] = S(emerge_delay_s)
        tex, mask = _pad_texture(prng.randint(96, 124), prng.randint(74, 96), rng)
        state["carry"] = (tex, mask)
        tx = prng.randint(14, bw - 14 - mask.shape[1])
        ty = prng.randint(int(0.30 * bh), bh - 14 - mask.shape[0])
        target = (box["x"] + tx + mask.shape[1] / 2, BOX_TOP + ty + mask.shape[0] / 2 - 10)
        move_hand(grab, target, 0.4)
        state["carry"] = None
        box["contents"].append((tex, mask, tx, ty))
        emit(S(hold_s), lambda i, n: state.__setitem__(
            "hand", (target[0] + (i % 3) - 1, target[1])))
        if stay_hand and not last:
            return                              # waits in the box for the next pad
        move_hand(target, rest_point(), 0.4)
        state["hand"] = None

    def brush_pad() -> None:
        """Push the hanging pad aside and let go: it swings back. Not a pad."""
        h = state["hang"]
        while h is None or h["len"] < full_len(h):
            emit(1)
            h = state["hang"]
        at = (OUTLET_X + h["dx"] - 30, OUTLET_Y + 0.55 * full_len(h))
        move_hand(rest_point(), at, 0.45)
        dx0 = h["dx"]

        def push(i, n):
            f = (i + 1) / n
            h["dx"] = dx0 + 22 * np.sin(np.pi * f)
            state["hand"] = (at[0] + 22 * np.sin(np.pi * min(1.0, 2 * f)) * (f < 0.5)
                             + 22 * (f >= 0.5) * (1 - (f - 0.5) * 2), at[1] + (i % 2))
        emit(S(0.7), push)
        h["dx"] = dx0 + prng.uniform(-3, 3)
        move_hand(state["hand"], rest_point(), 0.4)
        state["hand"] = None

    def rearrange_pad(box: dict) -> None:
        """Push the last pad somewhere else in the box: a reach, but no pad."""
        tex, mask, tx, ty = box["contents"][-1]
        at = (box["x"] + tx + mask.shape[1] / 2, BOX_TOP + ty + mask.shape[0] / 2 - 10)
        move_hand(rest_point(), at, 0.45)
        nx = min(bw - 14 - mask.shape[1], max(14, tx + prng.choice([-1, 1]) * 30))
        ny = min(bh - 14 - mask.shape[0], max(int(0.30 * bh), ty + prng.randint(-16, 16)))
        box["contents"][-1] = (tex, mask, nx, ny)
        to = (box["x"] + nx + mask.shape[1] / 2, BOX_TOP + ny + mask.shape[0] / 2 - 10)
        move_hand(at, to, 0.3)
        emit(S(0.3), lambda i, n: state.__setitem__("hand", (to[0] + (i % 3) - 1, to[1])))
        move_hand(to, rest_point(), 0.4)
        state["hand"] = None

    def run_box(n_pads: int) -> float:
        color = (72 + prng.randint(-6, 6), 110 + prng.randint(-6, 6),
                 150 + prng.randint(-6, 6))
        box = {"x": -bw - 40, "color": color, "contents": []}
        state["box"] = box
        t_in = len(frames)
        slide_box(-bw - 40, park_x, 1.0, drag=12.0)
        if settle_s > 0:
            wait(settle_s)
        for k in range(n_pads):
            pack_pad(box, last=(k == n_pads - 1))
            if brush and k == 0 and n_pads > 1 and not stay_hand:
                wait(0.6)
                brush_pad()
            if gap_s > 0 and k < n_pads - 1:
                wait(gap_s)
        if rearrange and n_pads:
            wait(max(gap_s, 0.9))
            rearrange_pad(box)
        if end_s > 0:
            wait(end_s)
        t_out = len(frames)
        slide_box(box["x"], W + 40, 1.0, drag=14.0)
        state["box"] = None
        return (t_out + S(0.5) - (t_in + S(0.5))) / fps

    truth_pads: List[int] = []
    truth_secs: List[float] = []
    wait(lead_s)                                    # the counter learns the spot
    if packer and state["body"] is None:
        step_in()
        wait(0.8)
    for i, n in enumerate(pads_per_box):
        truth_pads.append(n)
        truth_secs.append(round(run_box(n), 2))
        if packer and packer_leaves and i < len(pads_per_box) - 1:
            step_out()
            wait(max(0.3, box_gap_s - 1.2))
            step_in()
        else:
            wait(box_gap_s)
    if packer and state["body"] is not None:
        step_out()
    wait(4.0)

    if setup_dir:
        # The three pictures of tools/station_setup.py: empty spot with the
        # pad hanging, an empty box under that same pad, the next pad hanging.
        Path(setup_dir).mkdir(parents=True, exist_ok=True)
        srng = np.random.default_rng(seed + 1000)
        pad_a = dict(frames[0]["hang"])
        color = (72, 110, 150)
        base = {"box": None, "hang": pad_a, "hand": None, "carry": None,
                "body": None, "screen": 0}
        with_box = dict(base, box={"x": park_x, "color": color, "contents": []})
        w2 = int(pad_a["mask"].shape[1] * 0.96)
        tex_b, mask_b = _hang_texture(w2, int(pad_a["mask"].shape[0] * 0.9), srng)
        pad_b = {"tex": tex_b, "mask": mask_b, "len": float(mask_b.shape[0]), "dx": 5.0}
        after = dict(with_box, hang=pad_b, screen=1)
        for name, st in (("setup_empty", base), ("setup_box", with_box),
                         ("setup_outlet", after)):
            f = _render(bg, st, bw, bh).astype(np.int16) + noise[0]
            cv2.imwrite(str(Path(setup_dir) / f"{name}.png"), np.clip(f, 0, 255).astype(np.uint8))

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not out.isOpened():
            raise RuntimeError(f"Cannot open video writer for {path}")
        for fi, st in enumerate(frames):
            frame = _render(bg, st, bw, bh)
            gain = 1.0 + rng.uniform(-0.01, 0.01)
            f = frame.astype(np.int16) * gain + noise[fi % len(noise)]
            out.write(np.clip(f, 0, 255).astype(np.uint8))
        out.release()
    return {"boxes": len(truth_pads), "pads": truth_pads, "pack_seconds": truth_secs,
            "frames": len(frames), "fps": fps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/real_station_test.mp4")
    ap.add_argument("--pads", type=int, nargs="+", default=[3, 2, 4],
                    help="pads put into each box, e.g. --pads 3 2 4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out_dir = str(Path(args.out).parent)
    truth = generate_real_station_video(args.out, pads_per_box=tuple(args.pads),
                                        fps=args.fps, seed=args.seed, setup_dir=out_dir)
    print(f"Wrote {args.out}: {truth['frames']} frames at {truth['fps']} fps")
    print(f"Setup pictures in {out_dir}/ (setup_empty.png, setup_box.png, setup_outlet.png)")
    print(f"TRUTH: {truth['boxes']} boxes, pads per box {truth['pads']}, "
          f"pack seconds {truth['pack_seconds']}")
    print(f"\nTry it:\n  python3 -m boxcounter --source {args.out} --no-web")
    return 0


if __name__ == "__main__":
    sys.exit(main())
