"""Capture an empty-belt background image for processing.method: static.

Averages (median) several seconds of frames of the EMPTY, RUNNING belt so
transient noise and belt seams blend away.

Usage (on the Pi, belt running, no boxes):
    python3 tools/capture_background.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter.camera import create_source           # noqa: E402
from boxcounter.config import load_config             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--source", help="use a video file instead of the camera")
    ap.add_argument("--frames", type=int, default=60,
                    help="number of frames to sample (default 60)")
    ap.add_argument("--out", default=None,
                    help="output path (default: processing.background_image)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = Path(args.out or cfg.processing.background_image)

    source = create_source(cfg.camera, args.source)
    source.start()
    print(f"Sampling {args.frames} frames of the (empty!) belt ...")

    samples = []
    grabbed = 0
    while grabbed < args.frames:
        frame = source.read()
        if frame is None:
            break
        samples.append(frame)   # keep color: detection runs in color by default
        grabbed += 1
        time.sleep(0.05)   # spread samples over ~3 s of belt travel
    source.stop()

    if not samples:
        print("ERROR: no frames captured", file=sys.stderr)
        return 1

    background = np.median(np.stack(samples), axis=0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), background)
    print(f"Saved background ({grabbed} frames median) to {out_path}")
    print("Set processing.method: static in the config to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
