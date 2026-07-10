"""Headless calibration helper.

1) Snapshot mode (default): captures one frame and saves an annotated copy
   with a 10%-grid, the configured ROI and the counting line, so you can read
   fraction coordinates off the image and edit config.yaml accordingly.

       python3 tools/calibrate.py --config config/config.yaml

2) Check mode: runs the detector for a while and reports blob sizes as
   fractions of the ROI, then suggests min/max area settings. Run it with
   boxes passing on the belt.

       python3 tools/calibrate.py --check --seconds 30

For live visual tuning, start the counter and open the web dashboard
(http://<pi>:8080) — the "show mask" toggle displays the foreground mask.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter.camera import create_source           # noqa: E402
from boxcounter.config import load_config             # noqa: E402
from boxcounter.detector import BoxDetector           # noqa: E402


def snapshot(cfg, source_path, out_path: Path) -> int:
    source = create_source(cfg.camera, source_path)
    source.start()
    frame = source.read()
    source.stop()
    if frame is None:
        print("ERROR: no frame captured", file=sys.stderr)
        return 1

    fh, fw = frame.shape[:2]
    img = frame.copy()

    # 10% grid with fraction labels
    for i in range(1, 10):
        x = int(fw * i / 10)
        y = int(fh * i / 10)
        cv2.line(img, (x, 0), (x, fh), (90, 90, 90), 1)
        cv2.line(img, (0, y), (fw, y), (90, 90, 90), 1)
        cv2.putText(img, f"{i/10:.1f}", (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"{i/10:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # current ROI
    rx, ry, rw, rh = cfg.processing.roi
    cv2.rectangle(img, (int(rx * fw), int(ry * fh)),
                  (int((rx + rw) * fw), int((ry + rh) * fh)), (240, 160, 60), 2)
    # current counting line
    if cfg.counting.axis == "y":
        y = int(cfg.counting.line_position * fh)
        cv2.line(img, (0, y), (fw, y), (60, 220, 240), 2)
    else:
        x = int(cfg.counting.line_position * fw)
        cv2.line(img, (x, 0), (x, fh), (60, 220, 240), 2)

    cv2.imwrite(str(out_path), img)
    print(f"Snapshot saved to {out_path}")
    print("Blue = processing.roi, yellow = counting line, grid = fractions of frame.")
    print("Adjust config.yaml until the ROI covers only belt and the line sits")
    print("across the middle of the box path.")
    return 0


def check(cfg, source_path, seconds: float) -> int:
    source = create_source(cfg.camera, source_path)
    detector = BoxDetector(cfg.processing)
    source.start()
    print(f"Measuring detections for {seconds:.0f}s — run boxes on the belt now ...")

    areas = []
    deadline = time.monotonic() + seconds
    frames = 0
    t0 = time.monotonic()
    while time.monotonic() < deadline:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        detections, _ = detector.process(frame)
        if frames <= cfg.processing.warmup_frames:
            continue
        rx, ry, rw, rh = detector.roi_px
        roi_area = float(rw * rh)
        for det in detections:
            areas.append(det.area / roi_area)
    source.stop()
    elapsed = time.monotonic() - t0

    print(f"{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-6):.1f} fps)")
    if not areas:
        print("No blobs detected. Lower mog2_var_threshold or check lighting/ROI.")
        return 1
    areas.sort()
    p10 = areas[int(len(areas) * 0.10)]
    p50 = areas[len(areas) // 2]
    p90 = areas[int(len(areas) * 0.90)]
    print(f"Blob area (fraction of ROI): p10={p10:.3f}  median={p50:.3f}  p90={p90:.3f}")
    print(f"Suggested config:  min_area_frac: {max(0.005, p10 * 0.5):.3f}"
          f"   max_area_frac: {min(0.9, p90 * 2.5):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--source", help="use a video file instead of the camera")
    ap.add_argument("--check", action="store_true", help="measure blob sizes")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="data/calibration_snapshot.jpg")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        return check(cfg, args.source, args.seconds)
    return snapshot(cfg, args.source, out)


if __name__ == "__main__":
    sys.exit(main())
