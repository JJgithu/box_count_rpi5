"""Headless calibration helper — three modes.

1) Snapshot (default): saves an annotated frame with a 10% grid, the ROI,
   the counting line and the packing zone, so you can read fraction
   coordinates straight off the image and edit config.yaml.

       python3 tools/calibrate.py

2) Blob check: measures detected box sizes and suggests min/max area.
   Run it with boxes moving on the belt.

       python3 tools/calibrate.py --check --seconds 30

3) Packing check: live readout of what the arm detector actually sees —
   ring signal vs threshold, interior motion, session state, pieces. Use it
   to tune arm_enter_frac / interior_motion_frac by watching your own hand
   reach into a box. Saves an annotated frame of the parked box + its watch
   band the moment a session starts.

       python3 tools/calibrate.py --packing --seconds 60

For continuous visual tuning, run the counter and open http://<pi>:8080 —
the "show mask" toggle displays the foreground mask.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter.camera import create_source           # noqa: E402
from boxcounter.config import load_config             # noqa: E402
from boxcounter.detector import BoxDetector           # noqa: E402
from boxcounter.packing import PackingMonitor         # noqa: E402
from boxcounter.tracker import CentroidTracker        # noqa: E402

_BLUE = (240, 160, 60)
_YELLOW = (60, 220, 240)
_TEAL = (170, 180, 70)
_GREY = (90, 90, 90)
_WHITE = (200, 200, 200)


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
        x, y = int(fw * i / 10), int(fh * i / 10)
        cv2.line(img, (x, 0), (x, fh), _GREY, 1)
        cv2.line(img, (0, y), (fw, y), _GREY, 1)
        cv2.putText(img, f"{i/10:.1f}", (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{i/10:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, _WHITE, 1, cv2.LINE_AA)

    # ROI
    rx, ry, rw, rh = cfg.processing.roi
    cv2.rectangle(img, (int(rx * fw), int(ry * fh)),
                  (int((rx + rw) * fw), int((ry + rh) * fh)), _BLUE, 2)
    cv2.putText(img, "ROI", (int(rx * fw) + 4, int(ry * fh) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BLUE, 1, cv2.LINE_AA)

    # packing zone
    if cfg.packing.enabled:
        zx, zy, zw, zh = cfg.packing.zone
        p1 = (int(zx * fw), int(zy * fh))
        p2 = (int((zx + zw) * fw), int((zy + zh) * fh))
        cv2.rectangle(img, p1, p2, _TEAL, 2)
        cv2.putText(img, "PACK ZONE", (p1[0] + 4, p1[1] + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEAL, 1, cv2.LINE_AA)

    # counting line
    if cfg.counting.axis == "y":
        y = int(cfg.counting.line_position * fh)
        cv2.line(img, (0, y), (fw, y), _YELLOW, 2)
        cv2.putText(img, "COUNT LINE", (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, _YELLOW, 1, cv2.LINE_AA)
    else:
        x = int(cfg.counting.line_position * fw)
        cv2.line(img, (x, 0), (x, fh), _YELLOW, 2)
        cv2.putText(img, "COUNT LINE", (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, _YELLOW, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"Snapshot saved to {out_path}  ({fw}x{fh})")
    print("  blue  = processing.roi        -> must cover BELT ONLY")
    print("  teal  = packing.zone          -> where boxes stop to be packed")
    print("  yellow= counting.line_position-> after the zone; boxes counted here")
    print("  grid  = fractions of the frame, for reading off new values")
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
        _, _, rw, rh = detector.roi_px
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


def _bar(value: float, threshold: float, width: int = 12) -> str:
    """ASCII meter; '|' marks the threshold."""
    scale = max(threshold * 3.0, 1e-6)
    filled = min(width, int(round(width * value / scale)))
    mark = min(width - 1, int(round(width * threshold / scale)))
    cells = ["#" if i < filled else "-" for i in range(width)]
    cells[mark] = "|" if cells[mark] == "-" else "|"
    return "".join(cells)


def packing_check(cfg, source_path, seconds: float, out_dir: Path) -> int:
    """Live readout of the packing monitor's internal signals."""
    if not cfg.packing.enabled:
        print("NOTE: packing.enabled is false in the config; running the monitor\n"
              "      anyway so you can tune before switching it on.\n")

    source = create_source(cfg.camera, source_path)
    detector = BoxDetector(cfg.processing)
    source.start()
    first = source.read()
    if first is None:
        print("ERROR: no frame captured", file=sys.stderr)
        source.stop()
        return 1
    fh, fw = first.shape[:2]
    diag = (fw ** 2 + fh ** 2) ** 0.5
    tracker = CentroidTracker(cfg.tracking.max_distance_frac * diag,
                              cfg.tracking.max_disappeared)
    monitor = PackingMonitor(cfg.packing, (fh, fw))

    print(f"Watching for {seconds:.0f}s. Park a box in the zone and reach into it\n"
          f"as the packer would. Columns:\n"
          f"  signal  = arm signal (ring foreground above baseline), "
          f"threshold={cfg.packing.arm_enter_frac}\n"
          f"  motion  = movement inside the box, "
          f"threshold={cfg.packing.interior_motion_frac}\n")
    print(f"{'time':>6}  {'state':<8} {'signal':>7} {'bar':<12} {'motion':>7} "
          f"{'bar':<12}  pieces")

    signals_idle, signals_hand, motions = [], [], []
    window: deque = deque(maxlen=7)     # +/-3 frame guard around transitions
    saved_shot = False
    frames = 0
    t0 = time.monotonic()
    last_print = 0.0
    deadline = t0 + seconds
    while time.monotonic() < deadline:
        frame = source.read()
        if frame is None:
            break
        frames += 1
        detections, mask = detector.process(frame)
        if frames <= cfg.processing.warmup_frames:
            continue
        tracks = tracker.update(detections)

        full_mask = np.zeros((fh, fw), np.uint8)
        x0, y0, rw, rh = detector.roi_px
        full_mask[y0:y0 + rh, x0:x0 + rw] = mask
        media_t = time.monotonic() - t0
        monitor.update(media_t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       full_mask, tracks)

        ses = monitor.session
        if ses is not None:
            ring = monitor._ring_frac(full_mask)             # noqa: SLF001
            signal = max(0.0, ring - monitor._ring_baseline)  # noqa: SLF001
            motion = monitor._visit_max_motion if monitor.hand_in else 0.0  # noqa: SLF001
            # Classify only frames that are well away from an enter/exit
            # transition ON BOTH SIDES: the arm's approach ramp is still
            # "hand out" (the enter debounce has not latched yet) and the
            # exit tail is still "hand in", so sampling right at a boundary
            # would pollute both distributions and make a perfectly
            # separable setup look inseparable. A short sliding window with
            # a unanimous verdict gives the needed guard band.
            window.append((signal, motion, monitor.hand_in))
            if len(window) == window.maxlen:
                mid_sig, mid_motion, mid_hand = window[len(window) // 2]
                if all(h == mid_hand for _, _, h in window):
                    if mid_hand:
                        signals_hand.append(mid_sig)
                        motions.append(mid_motion)
                    else:
                        signals_idle.append(mid_sig)

            if not saved_shot:
                saved_shot = True
                shot = frame.copy()
                bx, by, bw, bh = ses.bbox
                r = cfg.packing.ring_px
                cv2.rectangle(shot, (bx, by), (bx + bw, by + bh), _TEAL, 2)
                cv2.rectangle(shot, (bx - r, by - r), (bx + bw + r, by + bh + r),
                              _YELLOW, 1)
                ix = int(bw * cfg.packing.interior_inset_frac)
                iy = int(bh * cfg.packing.interior_inset_frac)
                cv2.rectangle(shot, (bx + ix, by + iy),
                              (bx + bw - ix, by + bh - iy), _BLUE, 1)
                cv2.putText(shot, "teal=box  yellow=watch band  blue=interior",
                            (6, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            _WHITE, 1, cv2.LINE_AA)
                path = out_dir / "packing_snapshot.jpg"
                cv2.imwrite(str(path), shot)
                print(f"  -> session started; geometry saved to {path}")

            now = time.monotonic()
            if now - last_print >= 0.4:
                last_print = now
                state = "HAND IN" if monitor.hand_in else "idle"
                print(f"{media_t:6.1f}  {state:<8} {signal:7.3f} "
                      f"{_bar(signal, cfg.packing.arm_enter_frac):<12} "
                      f"{motion:7.3f} "
                      f"{_bar(motion, cfg.packing.interior_motion_frac):<12}  "
                      f"{ses.pieces}")
        else:
            now = time.monotonic()
            if now - last_print >= 2.0:
                last_print = now
                print(f"{media_t:6.1f}  {'no box':<8} "
                      f"(waiting for a box to stop in the zone; {len(tracks)} tracks)")

    source.stop()
    print()
    completed = list(monitor.completed) + ([monitor.session] if monitor.session else [])
    total_pieces = sum(s.pieces for s in completed)
    print(f"Sessions seen: {len(completed)}, pieces counted: {total_pieces}")
    if not signals_idle and not signals_hand:
        print("No packing session started. Check: is the box fully inside the frame,")
        print("inside packing.zone, and does it STOP? (see min_arrival_px, dwell_*)")
        return 1

    def pct(vals, p):
        if not vals:
            return float("nan")
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    print(f"arm signal while idle : p50={pct(signals_idle, .5):.3f} "
          f"p95={pct(signals_idle, .95):.3f}   (must stay BELOW arm_enter_frac)")
    print(f"arm signal while HAND : p50={pct(signals_hand, .5):.3f} "
          f"p95={pct(signals_hand, .95):.3f}   (must stay ABOVE arm_exit_frac)")
    if motions:
        print(f"interior motion, hand : p50={pct(motions, .5):.3f} "
              f"p95={pct(motions, .95):.3f}")
    if signals_idle and signals_hand:
        idle_hi, hand_lo = pct(signals_idle, .95), pct(signals_hand, .05)
        if hand_lo > idle_hi:
            enter = idle_hi + 0.6 * (hand_lo - idle_hi)
            exit_ = idle_hi + 0.25 * (hand_lo - idle_hi)
            print(f"\nSuggested:  arm_enter_frac: {enter:.3f}   "
                  f"arm_exit_frac: {exit_:.3f}")
        else:
            print("\nWARNING: idle and hand signals overlap — the arm is not clearly")
            print("separable. Widen ring_px, improve lighting, or move the queued box")
            print("further from the packing spot.")
    if motions:
        print(f"Suggested:  interior_motion_frac: {max(0.01, pct(motions, .1) * 0.5):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--source", help="use a video file instead of the camera")
    ap.add_argument("--check", action="store_true", help="measure blob sizes")
    ap.add_argument("--packing", action="store_true",
                    help="live readout of the packing/arm detector")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="data/calibration_snapshot.jpg")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.packing:
        return packing_check(cfg, args.source, args.seconds, out.parent)
    if args.check:
        return check(cfg, args.source, args.seconds)
    return snapshot(cfg, args.source, out)


if __name__ == "__main__":
    sys.exit(main())
