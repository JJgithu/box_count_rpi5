"""Set up the packing-station counter: two pictures, two key presses.

    python3 tools/station_setup.py

1. With the packing spot EMPTY, it takes a picture of the spot.
2. With an EMPTY BOX standing on the spot, it takes a second picture.

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
from boxcounter.station import brown_mask, foreground   # noqa: E402

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


def propose(res: dict, cfg) -> dict:
    """Config values for this box."""
    ww, wh = res["size"]
    x, y, w, h = res["bbox"]
    # The zone is where the box's CENTRE must be: the box's own outline plus
    # room for it to be put down a bit off the mark.
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
        cv2.putText(img, "PACKING ZONE (box centre must be inside)",
                    (int(zx * fw) + 4, int(zy * fh) + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (170, 180, 70), 1, cv2.LINE_AA)
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
    print("  Two pictures: the empty packing spot, then an empty box on it.\n"
          "  Stop the counter first if it is running (the camera can only be\n"
          "  used by one program):  sudo systemctl stop boxcounter")
    grab = Grabber(cfg, args.source)
    try:
        _h("Step 1 of 2 — the EMPTY spot")
        print("  Clear the rollers in front of the machine: no box, no hands.\n"
              "  Leave a pad hanging from the machine if one normally hangs there.")
        ask("Press Enter when the spot is empty...")
        print("  Taking the picture — keep your hands out of view...")
        empty = grab.median(args.frames, args.empty_at)
        if empty is None:
            _bad("no picture from the camera.")
            return 1
        _ok("got it")

        _h("Step 2 of 2 — an EMPTY BOX on the spot")
        print("  Put an empty box exactly where it stands while it is packed,\n"
              "  flaps open the way they normally are. Then step back.")
        ask("Press Enter when the box is in place and your hands are away...")
        print("  Taking the picture...")
        box = grab.median(args.frames, args.box_at)
        if box is None:
            _bad("no picture from the camera.")
            return 1
        _ok("got it")
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

    picture = save_pictures(out_dir, empty, box, res, updates)
    print(f"\n  Pictures saved: {picture} (the box in green,\n"
          f"  the packing zone as a rectangle), setup_empty.jpg, setup_box.jpg")

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
          "  why the last reach did or did not count as a pad.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
