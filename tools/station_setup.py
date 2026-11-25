"""Set up the packing-station counter: two or three pictures.

    python3 tools/station_setup.py

1. With the packing spot EMPTY, it takes a picture of the spot.
2. With an EMPTY BOX standing on the spot, it takes a second picture.
3. (Optional, recommended) After a pad is pulled off the machine and the
   next one has come out, a third picture: the difference shows where pads
   hang out of the machine, so that paper is never taken for part of a box
   or for a hand, and pads can be counted where they come out.

From the difference it measures where the box stands and how big it looks,
checks the camera sees all of it, and offers to save that to the config
(a backup of the config is kept). Works on the Pi's own screen or over SSH:
everything is shown as text. The pictures are also saved in data/ so they
can be looked at or sent along when asking for help.

Test it without a camera on a recorded video:

    python3 tools/station_setup.py --source data/recordings/clip.mp4 \\
        --empty-at 2.0 --box-at 12.0 --yes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter import asciiview                        # noqa: E402
from boxcounter.camera import create_source             # noqa: E402
from boxcounter.config import load_config               # noqa: E402
from boxcounter.configedit import apply_to_file         # noqa: E402
from boxcounter.station import (_WATCH_SAME, brown_mask, foreground,   # noqa: E402
                                outlet_windows, paper_look, same_pad)

_CSI = "\x1b["
_BOLD, _DIM, _RESET = f"{_CSI}1m", f"{_CSI}2m", f"{_CSI}0m"
_GREEN, _AMBER, _RED = f"{_CSI}32m", f"{_CSI}33m", f"{_CSI}31m"


def _h(text: str) -> None:
    print(f"\n{_BOLD}{text}{_RESET}")
    print(_DIM + "-" * min(72, max(20, len(text))) + _RESET)


def _ok(text: str) -> None:
    print(f"  {_GREEN}OK{_RESET}    {text}")


def _warn(text: str) -> None:
    print(f"  {_AMBER}NOTE{_RESET}  {text}")


def _bad(text: str) -> None:
    print(f"  {_RED}PROBLEM{_RESET} {text}")


class Grabber:
    """Frames from the camera, or from a video file at given times."""

    def __init__(self, cfg, source: Optional[str]):
        self.source = create_source(cfg.camera, source)
        self.is_video = source is not None
        self.fps = 30.0
        self.source.start()
        if self.is_video:
            cap = getattr(self.source, "cap", None)
            if cap is not None and cap.get(cv2.CAP_PROP_FPS) > 0:
                self.fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_no = 0

    def median(self, n: int, at_s: Optional[float] = None) -> Optional[np.ndarray]:
        """Median of n frames (a moving hand or flicker averages out)."""
        if self.is_video and at_s is not None:
            target = int(at_s * self.fps)
            while self._frame_no < target:
                if self.source.read() is None:
                    return None
                self._frame_no += 1
        frames: List[np.ndarray] = []
        while len(frames) < n:
            f = self.source.read()
            if f is None:
                break
            self._frame_no += 1
            frames.append(f)
            if not self.is_video:
                time.sleep(0.04)
        if not frames:
            return None
        return np.median(np.stack(frames), axis=0).astype(np.uint8)

    def stop(self) -> None:
        self.source.stop()


def analyse(empty: np.ndarray, box: np.ndarray, cfg) -> dict:
    """Find the box: what differs between the two pictures."""
    st = cfg.station
    fh, fw = empty.shape[:2]
    scale = min(1.0, st.work_width / float(fw))
    size = (max(32, int(round(fw * scale))), max(24, int(round(fh * scale))))
    e = cv2.GaussianBlur(cv2.resize(empty, size, interpolation=cv2.INTER_AREA), (3, 3), 0)
    b = cv2.GaussianBlur(cv2.resize(box, size, interpolation=cv2.INTER_AREA), (3, 3), 0)
    ww, wh = size
    rx, ry, rw, rh = cfg.processing.roi
    roi = np.zeros((wh, ww), bool)
    roi[int(ry * wh):int((ry + rh) * wh), int(rx * ww):int((rx + rw) * ww)] = True
    roi_area = float(np.count_nonzero(roi))

    _, fg = foreground(b, e.astype(np.float32), e, st.diff_threshold, st.suppress_shadows)
    fg &= roi
    m = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = {"found": False, "size": size, "roi_area": roi_area,
           "changed_frac": float(np.count_nonzero(fg)) / roi_area}
    if n <= 1:
        return out
    # The box: the biggest blob, preferring one near the middle of the view.
    best, best_score = 0, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        off = ((cx / ww - 0.5) ** 2 + (cy / wh - 0.5) ** 2) ** 0.5
        score = area * (1.0 - min(0.8, off))
        if score > best_score:
            best, best_score = i, score
    comp = labels == best
    x, y, w, h = (int(v) for v in stats[best, :4])
    area = int(stats[best, cv2.CC_STAT_AREA])
    brown = float(np.count_nonzero(brown_mask(b)[comp])) / max(1, area)
    edge = []
    x0, y0 = int(rx * ww), int(ry * wh)
    x1, y1 = int((rx + rw) * ww) - 1, int((ry + rh) * wh) - 1
    if x <= x0 + 1:
        edge.append("left")
    if x + w - 1 >= x1 - 1:
        edge.append("right")
    if y <= y0 + 1:
        edge.append("top")
    if y + h - 1 >= y1 - 1:
        edge.append("bottom")
    out.update({
        "found": True, "bbox": (x, y, w, h), "area_frac": area / roi_area,
        "brown_frac": brown, "edges": edge, "mask": comp,
        "box_frac_of_rect": area / float(max(1, w * h)),
    })
    return out


def find_outlet(box: np.ndarray, after: np.ndarray, cfg, res: dict) -> Optional[dict]:
    """Where the pad hangs: what differs between 'box, old pad' and
    'box, new pad'. Returns {"zone": [x, y, w, h] fractions, "frac": ...}."""
    st = cfg.station
    size = res["size"]
    a = cv2.GaussianBlur(cv2.resize(box, size, interpolation=cv2.INTER_AREA), (3, 3), 0)
    b = cv2.GaussianBlur(cv2.resize(after, size, interpolation=cv2.INTER_AREA), (3, 3), 0)
    raw, _ = foreground(b, a.astype(np.float32), a, st.diff_threshold, False)
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    ww, wh = size
    # the pad: the biggest change (a hand or the torn pad lying around would
    # be somewhere else) ...
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[best, :4])
    keep = labels == best
    # ... together with the other patches of the same strip of paper: two
    # crumpled pads look alike in places, so the change comes in patches one
    # above the other.
    # (Only paper-coloured patches: the machine's screen right above the
    # outlet changes too when a pad is made.)
    paper = brown_mask(a) | brown_mask(b)
    for i in range(1, n):
        if i == best or stats[i, cv2.CC_STAT_AREA] < 0.05 * stats[best, cv2.CC_STAT_AREA]:
            continue
        ix, iy, iw, ih = (int(v) for v in stats[i, :4])
        overlap = min(x + w, ix + iw) - max(x, ix)
        gap = max(iy - (y + h), y - (iy + ih))
        patch = labels == i
        if (overlap >= 0.5 * min(w, iw) and gap <= 0.12 * wh
                and np.count_nonzero(paper & patch) >= 0.6 * np.count_nonzero(patch)):
            keep |= patch
    ys, xs = np.nonzero(keep)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    area = int(np.count_nonzero(keep))
    frac = area / float(ww * wh)
    mx, my = max(2, int(0.12 * w)), max(2, int(0.12 * h))
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(ww, x + w + mx), min(wh, y + h + my)
    return {"zone": [round(x0 / ww, 3), round(y0 / wh, 3),
                     round((x1 - x0) / ww, 3), round((y1 - y0) / wh, 3)],
            "frac": frac, "mask": keep}


def propose(res: dict, cfg) -> dict:
    """Config values for this box."""
    ww, wh = res["size"]
    x, y, w, h = res["bbox"]
    # The zone is where boxes are looked for: the box's own outline plus room
    # for it to be put down a bit off the mark. What is outside is ignored.
    mx, my = 0.35 * w, 0.35 * h
    zx0, zy0 = max(0.0, (x - mx) / ww), max(0.0, (y - my) / wh)
    zx1, zy1 = min(1.0, (x + w + mx) / ww), min(1.0, (y + h + my) / wh)
    frac = res["area_frac"]
    min_frac = round(max(0.02, 0.45 * frac), 3)
    max_frac = round(min(0.98, max(0.5, 2.5 * frac)), 3)
    if max_frac <= min_frac:
        max_frac = min(1.0, min_frac + 0.1)
    return {
        "counting.mode": "station",
        "station.zone": [round(zx0, 3), round(zy0, 3),
                         round(zx1 - zx0, 3), round(zy1 - zy0, 3)],
        "station.min_box_frac": min_frac,
        "station.max_box_frac": max_frac,
        "station.box_color": "brown" if res["brown_frac"] >= 0.5 else "any",
    }


MAX_HIDDEN = 0.5       # an outlet zone hiding more of the box than this is not saved


def box_hidden_by(zone: List[float], res: dict) -> float:
    """How much of the box, as found in step 2, lies inside the outlet zone.
    The counter leaves the outlet zone out when it looks for boxes, so that
    part of the box is invisible to it."""
    ww, wh = res["size"]
    zx, zy, zw, zh = zone
    x0, y0 = int(zx * ww), int(zy * wh)
    x1, y1 = int((zx + zw) * ww), int((zy + zh) * wh)
    mask = res["mask"]
    total = max(1, int(np.count_nonzero(mask)))
    return float(np.count_nonzero(mask[y0:y1, x0:x1])) / total


def add_outlet(updates: dict, out: dict, res: dict) -> float:
    """Save the outlet zone unless it would hide most of the box; size the
    smallest accepted box by what stays visible. Returns the hidden part."""
    hidden = box_hidden_by(out["zone"], res)
    if hidden > MAX_HIDDEN:
        return hidden
    updates["station.outlet_zone"] = out["zone"]
    if hidden > 0.02:
        updates["station.min_box_frac"] = round(
            max(0.02, 0.45 * res["area_frac"] * (1.0 - hidden)), 3)
    return hidden


def check_outlet(box: np.ndarray, after: np.ndarray, zone: List[float], res: dict) -> dict:
    """Can pads be counted at the machine? The counter's own two tests, run
    on the pictures of step 2 (old pad) and step 3 (next pad): it must see
    paper right below the outlet in both, and it must tell the two pads
    apart."""
    ww, wh = res["size"]
    fh, fw = box.shape[:2]
    sx, sy = fw / float(ww), fh / float(wh)
    _, watch, mouth = outlet_windows(zone, ww, wh)

    def full(r):
        return (int(round(r[0] * sx)), int(round(r[1] * sy)),
                int(round(r[2] * sx)), int(round(r[3] * sy)))

    out, patches = {}, []
    for name, img in (("before", box), ("after", after)):
        small = cv2.GaussianBlur(cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA),
                                 (3, 3), 0)
        mx, my, mw, mh = mouth
        fx, fy, fw_, fh_ = full(mouth)
        strip = cv2.cvtColor(img[fy:fy + fh_, fx:fx + fw_], cv2.COLOR_BGR2GRAY)
        out["paper_" + name] = paper_look(small[my:my + mh, mx:mx + mw], strip)
        fx, fy, fw_, fh_ = full(watch)
        patches.append(cv2.GaussianBlur(
            cv2.cvtColor(img[fy:fy + fh_, fx:fx + fw_], cv2.COLOR_BGR2GRAY), (3, 3), 0))
    out["match"] = same_pad(patches[0], patches[1])
    out["ok"] = bool(out["paper_before"] and out["paper_after"]
                     and out["match"] < _WATCH_SAME)
    return out


def save_pictures(out_dir: Path, empty, box, res, updates) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "setup_empty.jpg"), empty)
    cv2.imwrite(str(out_dir / "setup_box.jpg"), box)
    img = box.copy()
    fh, fw = img.shape[:2]
    if res.get("found"):
        ww, wh = res["size"]
        sx, sy = fw / float(ww), fh / float(wh)
        mask = cv2.resize(res["mask"].astype(np.uint8) * 255, (fw, fh),
                          interpolation=cv2.INTER_NEAREST)
        tint = img.copy()
        tint[mask > 0] = (60, 200, 60)
        img = cv2.addWeighted(img, 0.6, tint, 0.4, 0)
        zx, zy, zw, zh = updates["station.zone"]
        cv2.rectangle(img, (int(zx * fw), int(zy * fh)),
                      (int((zx + zw) * fw) - 1, int((zy + zh) * fh) - 1), (170, 180, 70), 2)
        cv2.putText(img, "PACKING ZONE (boxes are looked for in here)",
                    (int(zx * fw) + 4, int(zy * fh) + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (170, 180, 70), 1, cv2.LINE_AA)
        if updates.get("station.outlet_zone"):
            ox, oy, ow, oh = updates["station.outlet_zone"]
            cv2.rectangle(img, (int(ox * fw), int(oy * fh)),
                          (int((ox + ow) * fw) - 1, int((oy + oh) * fh) - 1), (60, 200, 240), 2)
            cv2.putText(img, "PAD OUTLET", (int(ox * fw) + 4, int((oy + oh) * fh) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 240), 1, cv2.LINE_AA)
    path = out_dir / "setup_result.jpg"
    cv2.imwrite(str(path), img)
    return path


def show_ascii(res: dict, frame_shape, updates: Optional[dict]) -> None:
    fh, fw = frame_shape[:2]
    cols, rows = asciiview.grid_size()
    mask = cv2.resize(res["mask"].astype(np.uint8) * 255, (fw, fh),
                      interpolation=cv2.INTER_NEAREST)
    zone_px = None
    if updates:
        zx, zy, zw, zh = updates["station.zone"]
        zone_px = (int(zx * fw), int(zy * fh), int(zw * fw), int(zh * fh))
    ww, wh = res["size"]
    x, y, w, h = res["bbox"]
    sx, sy = fw / float(ww), fh / float(wh)
    box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
    for line in asciiview.render(mask, (fh, fw), cols, rows, zone_px=zone_px,
                                 boxes=[box]):
        print("  " + line)
    print(f"  {_DIM}@#: the box   |-|: its outline   :.: the packing zone{_RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--source", help="use a video file instead of the camera")
    ap.add_argument("--empty-at", type=float, default=None,
                    help="with --source: seconds into the video of the empty spot")
    ap.add_argument("--box-at", type=float, default=None,
                    help="with --source: seconds into the video of the box")
    ap.add_argument("--outlet-at", type=float, default=None,
                    help="with --source: seconds into the video after a new pad "
                         "has come out (step 3); omit to skip step 3")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames averaged per picture (default 20)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="do not ask; save the result")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    interactive = not args.yes and sys.stdin.isatty()

    def ask(prompt: str) -> None:
        if interactive:
            input(f"\n  {_BOLD}{prompt}{_RESET} ")

    _h("Packing-station setup")
    print("  Pictures of: the empty packing spot, an empty box on it, and\n"
          "  (optional) the machine after it has pushed out a new pad.\n"
          "  Stop the counter first if it is running (the camera can only be\n"
          "  used by one program):  sudo systemctl stop boxcounter")
    try:
        grab = Grabber(cfg, args.source)
    except Exception as exc:
        _bad(f"could not open the camera ({exc}).")
        print("  Is the counter still running? Only one program can use the\n"
              "  camera: stop it (Ctrl-C in its window, or\n"
              "  sudo systemctl stop boxcounter) and run the setup again.")
        return 1
    try:
        _h("Step 1 of 3 — the EMPTY spot")
        print("  Clear the rollers in front of the machine: no box, no hands.\n"
              "  Leave a pad hanging from the machine if one normally hangs there\n"
              "  — and keep that same pad hanging for step 2.")
        ask("Press Enter when the spot is empty...")
        print("  Taking the picture — keep your hands out of view...")
        empty = grab.median(args.frames, args.empty_at)
        if empty is None:
            _bad("no picture from the camera.")
            return 1
        _ok("got it")

        _h("Step 2 of 3 — an EMPTY BOX on the spot")
        print("  Put an empty box exactly where it stands while it is packed,\n"
              "  flaps open the way they normally are. Then step back. The pad\n"
              "  hanging from the machine may hang over the box — leave it as is.")
        ask("Press Enter when the box is in place and your hands are away...")
        print("  Taking the picture...")
        box = grab.median(args.frames, args.box_at)
        if box is None:
            _bad("no picture from the camera.")
            return 1
        _ok("got it")

        outlet_img = None
        _h("Step 3 of 3 — where the pads come out")
        print("  This is what lets the counter count pads at the machine.\n"
              "  Leave the box where it is. Pull the hanging pad off the machine\n"
              "  and put it ASIDE (not in the box, not on the rollers), and let\n"
              "  the machine push out the next one. Then step back.")
        if interactive:
            ans = input(f"\n  {_BOLD}Press Enter when the new pad hangs still "
                        f"(or type s and Enter to skip)...{_RESET} ").strip().lower()
            if ans != "s":
                print("  Taking the picture...")
                outlet_img = grab.median(args.frames)
        elif args.outlet_at is not None:
            outlet_img = grab.median(args.frames, args.outlet_at)
        if outlet_img is not None:
            _ok("got it")
        else:
            print("  skipped")
    finally:
        grab.stop()

    res = analyse(empty, box, cfg)
    out_dir = Path(cfg.output.data_dir)
    _h("What the camera sees")
    if not res["found"] or res["area_frac"] < 0.01:
        save_pictures(out_dir, empty, box, res, {"station.zone": [0, 0, 1, 1]})
        _bad("I can't see a box: the two pictures look the same.")
        print("  - Is the camera pointing at the packing spot?\n"
              "  - Was the box really on the spot for the second picture?\n"
              f"  - Pictures saved in {out_dir}/ (setup_empty.jpg, setup_box.jpg).")
        return 1

    updates = propose(res, cfg)
    frac = res["area_frac"]
    show_ascii(res, box.shape, updates)
    print()
    _ok(f"box found: it fills {frac:.0%} of the picture")
    if frac < 0.04:
        _warn("that is small — the camera may be too far away to see pads go in.\n"
              "        Lower the camera or zoom in if you can.")
    if frac > 0.85:
        _warn("the box fills almost the whole picture — raise the camera a little\n"
              "        so there is a margin of rollers around the box.")
    if res["edges"]:
        _warn(f"the box touches the {' and '.join(res['edges'])} edge of the picture.\n"
              "        It still counts, but turn or raise the camera so the whole\n"
              "        box (with its flaps) is in view if you can.")
    else:
        _ok("the whole box is in view")
    if updates["station.box_color"] == "brown":
        _ok(f"cardboard colour recognised ({res['brown_frac']:.0%} of the box)")
    else:
        _warn(f"only {res['brown_frac']:.0%} of the box looks cardboard-brown to the "
              "camera, so\n        colour will not be used to tell boxes from other "
              "things (box_color: any).")
    if res["changed_frac"] > 2.5 * frac + 0.05:
        _warn("other parts of the picture changed too between the two pictures\n"
              "        (someone walking by, lights). If the result looks wrong,\n"
              "        run the setup again.")

    # Pads: at the machine if step 3 shows the camera can do it, otherwise
    # per reach into the box. (Always written, so an outlet measured for an
    # earlier camera position does not linger.)
    updates["station.outlet_zone"] = []
    updates["station.pad_method"] = "auto"
    if outlet_img is not None:
        out = find_outlet(box, outlet_img, cfg, res)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "setup_outlet.jpg"), outlet_img)
        if out is None or out["frac"] < 0.002:
            _warn("I could not see the pad change at the machine — is the outlet in\n"
                  "        the picture? Pads will be counted per reach into the box.")
        elif out["frac"] > 0.35:
            _warn("too much of the picture changed in step 3 (someone in view?);\n"
                  "        the outlet was not saved. Run the setup again to retry.")
        else:
            hidden = add_outlet(updates, out, res)
            if not updates["station.outlet_zone"]:
                _warn(f"the pad hangs over {hidden:.0%} of the box as the camera sees it.\n"
                      "        Leaving that much out would hide the box, so the outlet was\n"
                      "        not saved. Pads will be counted per reach into the box.")
            else:
                _ok(f"outlet found: the hanging pad covers {out['frac']:.0%} of the picture")
                print("        That area is no longer looked at as part of the box"
                      + (f"\n        ({hidden:.0%} of the box; the smallest box size is"
                         " lowered to match)." if hidden > 0.02 else "."))
                chk = check_outlet(box, outlet_img, out["zone"], res)
                if chk["ok"]:
                    _ok("pads will be counted at the machine: every pad pulled off "
                        "counts once\n        (the camera sees the paper below the outlet "
                        "and tells the two pads apart)")
                else:
                    updates["station.pad_method"] = "reach"
                    if not (chk["paper_before"] and chk["paper_after"]):
                        why = ("the camera does not see paper right below the outlet "
                               "in both pictures")
                    else:
                        why = (f"the two pads look the same to the camera "
                               f"(match {chk['match']:.2f})")
                    _warn(f"pads cannot be counted at the machine: {why}.\n"
                          "        They will be counted per reach into the box instead.")
    else:
        print("  Step 3 skipped: pads will be counted per reach into the box.")

    picture = save_pictures(out_dir, empty, box, res, updates)
    print(f"\n  Pictures saved: {picture} (the box in green, the packing zone\n"
          f"  and the pad outlet as rectangles), setup_empty.jpg, setup_box.jpg")

    _h("Settings for this box")
    for key, value in updates.items():
        print(f"  {key:<24} {value}")
    print(f"  {_DIM}(min/max box size are fractions of the picture; smaller boxes\n"
          f"   down to {updates['station.min_box_frac']:.0%} of the picture still count){_RESET}")

    if interactive:
        answer = input(f"\n  {_BOLD}Save these to {cfg_path}? [Y/n]{_RESET} ").strip().lower()
        if answer not in ("", "y", "yes"):
            print("  Not saved.")
            return 0
    backup, changes = apply_to_file(cfg_path, updates, add=True)
    try:
        load_config(cfg_path)
    except Exception as exc:                      # never leave a broken config
        backup_text = backup.read_text()
        cfg_path.write_text(backup_text)
        _bad(f"the new settings did not validate ({exc}); config restored.")
        return 1
    _ok(f"saved to {cfg_path} (previous version kept as {backup.name})")
    for c in changes:
        print(f"        {c}")
    _h("Next")
    print("  Start the counter and pack a box:\n\n"
          "      python3 -m boxcounter\n\n"
          "  The line starting with 'now:' says what it sees: 'waiting for a\n"
          "  box', 'box at the station: 2 pads, 14 s', and the line 'last:' says\n"
          "  what it decided last ('a pad was taken from the machine').\n"
          "  If something is off:  python3 tools/station_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
