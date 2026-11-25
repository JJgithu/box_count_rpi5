"""Everything needed to see why the counter decided what it did, in two
things that are easy to send: a block of text and one picture.

    python3 tools/station_report.py

Prints (and saves to data/report/report.txt):
  - the version and the station settings in use,
  - the counter's last decisions from data/boxcounter.log: every box, every
    pad, and every time something at the spot was NOT taken as a box, with
    the reason.

Writes data/report/setup.jpg: the pictures the setup took (empty spot, box,
next pad, result) side by side — what the camera actually sees.

Lists the recorded clips (python3 -m boxcounter --record), newest first.

To get the files off the Pi: while the counter runs, open
http://<pi-address>:8080/files in a browser on the same network and tap
them; or copy them with scp (see docs/STATION.md).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boxcounter.config import load_config                # noqa: E402

DECISION = re.compile(
    r"Box |Outlet|Ignoring|not taken as a box|Learned|re-learn|abandon|gave up|"
    r"Repaired|Not counted|Locked exposure|Manual exposure|Picamera2 started|"
    r"Flicker|ERROR|WARNING|Traceback")


def version() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd %s",
                              "--date=short"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def decisions(log_path: Path, limit: int) -> list:
    if not log_path.exists():
        return []
    lines = []
    for name in (str(log_path) + ".1", str(log_path)):       # older part first
        p = Path(name)
        if p.exists():
            with p.open(errors="replace") as fh:
                lines += [ln.rstrip("\n") for ln in fh if DECISION.search(ln)]
    return lines[-limit:]


def montage(data_dir: Path, out: Path) -> bool:
    tiles = []
    for name in ("setup_empty.jpg", "setup_box.jpg", "setup_outlet.jpg", "setup_result.jpg"):
        if not (data_dir / name).exists():
            continue
        img = cv2.imread(str(data_dir / name))
        if img is None:
            continue
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (639, 22), (0, 0, 0), -1)
        cv2.putText(img, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return False
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    out.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--lines", type=int, default=80,
                    help="how many of the last decisions to show (default 80)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = Path(cfg.output.data_dir)
    st, cam = cfg.station, cfg.camera
    text = [
        "=== box counter report ===",
        f"version : {version()}",
        f"made    : {time.strftime('%Y-%m-%d %H:%M')}",
        f"mode    : {cfg.counting.mode}",
        f"camera  : {cam.source} {cam.width}x{cam.height} @ {cam.fps} fps, "
        f"flicker_hz {cam.flicker_hz}, lock_exposure {cam.lock_exposure}",
        f"roi     : {cfg.processing.roi}",
        f"zone    : {st.zone}",
        f"box     : {st.min_box_frac} .. {st.max_box_frac} of the picture, colour {st.box_color}",
        f"outlet  : {st.outlet_zone if st.outlet_zone else 'not measured'}",
        f"pads    : {'off' if not st.count_pads else st.pad_method}"
        f" (quiet {st.quiet_seconds}, pad_min_change {st.pad_min_change},"
        f" outlet_quiet {st.outlet_quiet_seconds})",
        f"timing  : confirm {st.confirm_seconds}/{st.confirm_busy_seconds}, "
        f"depart {st.depart_seconds}, min_pack {st.min_pack_seconds}",
        "",
        f"--- last decisions ({data / 'boxcounter.log'}) ---",
    ]
    found = decisions(data / "boxcounter.log", args.lines)
    text += found if found else ["(no log yet: run the counter first)"]

    clips = sorted((data / "recordings").glob("*.mp4"), key=lambda p: -p.stat().st_mtime)
    text += ["", "--- recorded clips (newest first) ---"]
    text += [f"{p}  {p.stat().st_size / 1e6:.1f} MB  "
             f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}"
             for p in clips[:8]] or ["(none: python3 -m boxcounter --record)"]

    out_dir = data / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print()
    if montage(data, out_dir / "setup.jpg"):
        print(f"Setup pictures in one image: {out_dir / 'setup.jpg'}")
    else:
        print("No setup pictures found (run python3 tools/station_setup.py).")
    print(f"This text is saved as:       {out_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
