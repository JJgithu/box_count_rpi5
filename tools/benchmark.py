"""Measure per-stage processing time on the target hardware.

    python3 tools/benchmark.py --config config/config.yaml            # camera
    python3 tools/benchmark.py --source data/test.mp4 --frames 500    # video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter.camera import create_source           # noqa: E402
from boxcounter.config import load_config             # noqa: E402
from boxcounter.detector import BoxDetector           # noqa: E402
from boxcounter.tracker import CentroidTracker        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--source", help="video file instead of the camera")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = create_source(cfg.camera, args.source)
    detector = BoxDetector(cfg.processing)
    tracker = None

    source.start()
    t_read = t_detect = t_track = 0.0
    n = 0
    wall0 = time.monotonic()
    while n < args.frames:
        t0 = time.monotonic()
        frame = source.read()
        if frame is None:
            break
        t1 = time.monotonic()
        detections, _ = detector.process(frame)
        t2 = time.monotonic()
        if tracker is None:
            fh, fw = frame.shape[:2]
            diag = (fw ** 2 + fh ** 2) ** 0.5
            tracker = CentroidTracker(cfg.tracking.max_distance_frac * diag,
                                      cfg.tracking.max_disappeared)
        tracker.update(detections)
        t3 = time.monotonic()
        t_read += t1 - t0
        t_detect += t2 - t1
        t_track += t3 - t2
        n += 1
    wall = time.monotonic() - wall0
    source.stop()

    if n == 0:
        print("No frames processed", file=sys.stderr)
        return 1
    print(f"frames:      {n}")
    print(f"capture:     {1000 * t_read / n:6.2f} ms/frame")
    print(f"detection:   {1000 * t_detect / n:6.2f} ms/frame")
    print(f"tracking:    {1000 * t_track / n:6.2f} ms/frame")
    print(f"end-to-end:  {n / wall:6.1f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
