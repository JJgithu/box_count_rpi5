"""Guided calibration — run this at the machine, follow what it says.

    python3 tools/wizard.py                 # full guided calibration
    python3 tools/wizard.py --diagnose      # "why is nothing counting?"
    python3 tools/wizard.py --live          # live view of what the camera sees
    python3 tools/wizard.py --geometry      # watch boxes, set ROI/line/zone

No browser and no copying files to another machine: the camera's view is
drawn in the terminal, so this works over SSH or on a monitor plugged into
the Pi.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxcounter import asciiview                       # noqa: E402
from boxcounter.camera import create_source            # noqa: E402
from boxcounter.config import load_config              # noqa: E402
from boxcounter.configedit import (apply_to_file,      # noqa: E402
                                   format_value, set_values)
from boxcounter.counter import LineCounter             # noqa: E402
from boxcounter.detector import BoxDetector            # noqa: E402
from boxcounter.tracker import CentroidTracker         # noqa: E402

# Anything smaller than this fraction of the region of interest is treated as
# noise or debris rather than a box. Also the floor for min_area_frac: below
# it the detector starts accepting specks.
_NOISE_FRAC = 0.002
# Fewest box sightings worth calibrating from.
_MIN_SAMPLES = 20

_CSI = "\x1b["
_RESET = f"{_CSI}0m"
_BOLD = f"{_CSI}1m"
_DIM = f"{_CSI}2m"
_RED = f"{_CSI}31m"
_GREEN = f"{_CSI}32m"
_AMBER = f"{_CSI}33m"


def _h(text: str) -> None:
    print(f"\n{_BOLD}{text}{_RESET}")
    print(_DIM + "-" * min(72, max(20, len(text))) + _RESET)


def _ok(text: str) -> None:
    print(f"  {_GREEN}OK{_RESET}    {text}")


def _warn(text: str) -> None:
    print(f"  {_AMBER}WARN{_RESET}  {text}")


def _bad(text: str) -> None:
    print(f"  {_RED}PROBLEM{_RESET} {text}")


def _pause(prompt: str = "Press Enter to continue") -> None:
    try:
        input(f"\n{_DIM}{prompt}...{_RESET}")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(0)


class Observer:
    """Runs the real detection stages and records what happens at each one."""

    def __init__(self, cfg, source_path: Optional[str]):
        self.cfg = cfg
        self.source = create_source(cfg.camera, source_path)
        self.detector = BoxDetector(cfg.processing)
        self.tracker: Optional[CentroidTracker] = None
        self.counter: Optional[LineCounter] = None
        self.frame_shape: Tuple[int, int] = (0, 0)

        # funnel counters
        self.frames = 0
        self.frames_with_fg = 0
        self.frames_with_contours = 0
        self.frames_with_detections = 0
        self.raw_blob_fracs: List[float] = []      # before size filtering
        self.kept_blob_fracs: List[float] = []     # after size filtering
        # Raw blobs as (area_frac, cx_frac, cy_frac). Needed so geometry can
        # still be inferred when the current size limits reject everything —
        # otherwise a bad min_area_frac would make the wizard unable to fix
        # the very setting that is broken.
        self.raw_blobs: List[Tuple[float, float, float]] = []
        # Largest blob per frame. A transient (an operator leaning over the
        # belt) dominates a pooled percentile but only affects a handful of
        # frames, so per-frame maxima are far more robust for sizing advice.
        self.frame_max_blob: List[float] = []
        # Absolute travel per axis, for detecting which way the belt runs.
        self.abs_dx = 0.0
        self.abs_dy = 0.0
        self.fg_fracs: List[float] = []
        self.track_ids: set = set()
        self.max_hits = 0
        self.crossed_ids: set = set()
        self.counted = 0
        self.centroids: List[Tuple[float, float]] = []
        self.slow_points: List[Tuple[float, float]] = []
        self.travel_dx = 0.0
        self.travel_dy = 0.0
        self.last_mask: Optional[np.ndarray] = None
        self.last_boxes: List[Tuple[int, int, int, int]] = []

    def start(self):
        self.source.start()

    def stop(self):
        self.source.stop()

    def _full_mask(self, mask) -> np.ndarray:
        fh, fw = self.frame_shape
        full = np.zeros((fh, fw), np.uint8)
        x0, y0, rw, rh = self.detector.roi_px
        full[y0:y0 + rh, x0:x0 + rw] = mask
        return full

    def step(self) -> bool:
        frame = self.source.read()
        if frame is None:
            return False
        self.frames += 1
        self.frame_shape = frame.shape[:2]
        detections, mask = self.detector.process(frame)

        roi_area = float(mask.shape[0] * mask.shape[1]) or 1.0
        fg = cv2.countNonZero(mask)
        self.fg_fracs.append(fg / roi_area)
        if fg > 0:
            self.frames_with_fg += 1

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            self.frames_with_contours += 1
            rx0, ry0 = self.detector.roi_px[0], self.detector.roi_px[1]
            fh_, fw_ = self.frame_shape
            biggest = 0.0
            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                frac = (bw * bh) / roi_area
                biggest = max(biggest, frac)
                self.raw_blob_fracs.append(frac)
                self.raw_blobs.append((
                    frac,
                    (rx0 + bx + bw / 2) / fw_,
                    (ry0 + by + bh / 2) / fh_))
            self.frame_max_blob.append(biggest)

        warm = self.frames <= self.cfg.processing.warmup_frames
        if warm:
            detections = []
        if detections:
            self.frames_with_detections += 1
            for d in detections:
                self.kept_blob_fracs.append(d.area / roi_area)

        if self.tracker is None:
            fh, fw = self.frame_shape
            diag = (fw ** 2 + fh ** 2) ** 0.5
            self.tracker = CentroidTracker(
                self.cfg.tracking.max_distance_frac * diag,
                self.cfg.tracking.max_disappeared)
            ccfg = self.cfg.counting
            span = fw if ccfg.axis == "x" else fh
            rx0, ry0, rw, rh = self.detector.roi_px
            bounds = (rx0, rx0 + rw) if ccfg.axis == "x" else (ry0, ry0 + rh)
            self.counter = LineCounter(
                axis=ccfg.axis, line_px=ccfg.line_position * span,
                hysteresis_px=ccfg.hysteresis_frac * span,
                direction=ccfg.direction,
                min_travel_px=ccfg.min_travel_frac * span,
                min_hits=self.cfg.tracking.min_hits, bounds_px=bounds)

        tracks = self.tracker.update(detections)
        fh, fw = self.frame_shape
        for tr in tracks:
            self.track_ids.add(tr.track_id)
            self.max_hits = max(self.max_hits, tr.hits)
            if tr.misses == 0:
                cx, cy = tr.centroid
                self.centroids.append((cx / fw, cy / fh))
                speed = (tr.velocity[0] ** 2 + tr.velocity[1] ** 2) ** 0.5
                # A brand-new track starts at velocity (0, 0), so it would
                # look "stopped" wherever it first appears. Only trust the
                # speed once it has been measured over several frames.
                if tr.hits >= 4 and speed < self.cfg.packing.dwell_speed_px:
                    self.slow_points.append((cx / fw, cy / fh))
                self.travel_dx += tr.velocity[0]
                self.travel_dy += tr.velocity[1]
                self.abs_dx += abs(tr.velocity[0])
                self.abs_dy += abs(tr.velocity[1])
        if not warm:
            self.counted += len(self.counter.update(tracks))
            # A track that has been confidently on BOTH sides has crossed.
            for tr in tracks:
                if (tr.track_id in self.counter._seen_minus  # noqa: SLF001
                        and tr.track_id in self.counter._seen_plus):  # noqa: SLF001
                    self.crossed_ids.add(tr.track_id)

        self.last_mask = self._full_mask(mask)
        self.last_boxes = [d.bbox for d in detections]
        return True


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _validate_proposal(cfg_path: Path, updates: dict) -> List[str]:
    """Try the proposed values on a scratch copy; return any load errors."""
    import tempfile
    try:
        text, _ = set_values(cfg_path.read_text(), updates)
    except Exception as exc:                       # malformed update
        return [str(exc)]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        load_config(tmp)
        return []
    except Exception as exc:
        return [line.strip(" -") for line in str(exc).splitlines() if line.strip()]
    finally:
        tmp.unlink(missing_ok=True)


def _observe(cfg, source_path, seconds: float, label: str) -> Observer:
    obs = Observer(cfg, source_path)
    obs.start()
    print(f"{label} for {seconds:.0f}s ...")
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            if not obs.step():
                break
    finally:
        obs.stop()
    return obs


# ---------------------------------------------------------------- diagnose

def diagnose(cfg, source_path, seconds: float) -> int:
    _h("DIAGNOSIS — where does the chain break?")
    print("Run boxes down the belt (and pack one) while this measures.\n")
    obs = _observe(cfg, source_path, seconds, "Watching")

    if obs.frames == 0:
        _bad("No frames from the camera at all.")
        print("     Check:  rpicam-hello --list-cameras")
        return 1

    fh, fw = obs.frame_shape
    roi = obs.detector.roi_px
    print(f"\nFrames: {obs.frames}  ({obs.frames / max(seconds,1e-9):.1f} fps), "
          f"image {fw}x{fh}, ROI {roi}")

    _h("Stage-by-stage funnel")
    stages = [
        ("frames captured", obs.frames),
        ("frames with ANY foreground", obs.frames_with_fg),
        ("frames with blobs (any size)", obs.frames_with_contours),
        ("frames with ACCEPTED boxes", obs.frames_with_detections),
        ("tracks created", len(obs.track_ids)),
        ("tracks that crossed the line", len(obs.crossed_ids)),
        ("BOXES COUNTED", obs.counted),
    ]
    width = max(len(n) for n, _ in stages)
    for name, value in stages:
        bar = "#" * min(30, value) if value else ""
        print(f"  {name:<{width}}  {value:>6}  {_DIM}{bar}{_RESET}")

    _h("Verdict")
    rc = 0
    fg_p95 = _pct(obs.fg_fracs, 0.95) or 0.0

    if obs.frames_with_fg == 0:
        _bad("The detector sees NOTHING moving — every frame is empty.")
        print("     Most likely: nothing actually passed during the test, or the")
        print("     ROI is looking at the wrong part of the image, or the")
        print("     background model absorbed everything.")
        print(f"     Try:  lower processing.mog2_var_threshold "
              f"(now {cfg.processing.mog2_var_threshold}) to 16")
        print("     Then: python3 tools/wizard.py --live   (see what it sees)")
        rc = 1
    elif obs.frames_with_contours == 0:
        _bad("Foreground exists but forms no blobs — it is being erased.")
        print(f"     processing.open_kernel ({cfg.processing.open_kernel}) is "
              "probably too aggressive; try 3.")
        rc = 1
    elif obs.frames_with_detections == 0:
        # Per-frame maxima, so one transient (an operator leaning over the
        # belt) cannot masquerade as "your boxes".
        big = _pct(obs.frame_max_blob, 0.50) or 0.0
        peak = max(obs.frame_max_blob) if obs.frame_max_blob else 0.0
        if big < _NOISE_FRAC and peak < _NOISE_FRAC:
            # Everything moving was speck-sized: no box came past at all.
            _bad("Nothing box-sized passed while this was measuring.")
            print(f"     The biggest moving thing was {peak*100:.2f}% of the "
                  "region of interest — that is noise or debris.")
            print("     Either no box went by during the test (send boxes past")
            print("     WHILE it measures), or the region of interest is not")
            print("     looking at the belt. Check what the camera sees:")
            print("        python3 tools/wizard.py --live")
        else:
            _bad("Blobs are found but ALL are rejected by the size filter.")
            print(f"     Typical biggest blob per frame: {big:.4f} "
                  f"(largest seen {peak:.4f}, fraction of ROI)")
            print(f"     Configured accept range: "
                  f"{cfg.processing.min_area_frac} .. {cfg.processing.max_area_frac}")
            if big < cfg.processing.min_area_frac:
                suggestion = max(_NOISE_FRAC, big * 0.4)
                print("     -> your boxes are SMALLER than min_area_frac.")
                print(f"        Set processing.min_area_frac: {suggestion:.4f}")
            elif big > cfg.processing.max_area_frac:
                print("     -> your boxes are LARGER than max_area_frac.")
                print(f"        Set processing.max_area_frac: {min(0.9, big*1.6):.3f}")
            else:
                print("     -> blobs fall between the two limits; widen both.")
            print("     Or let the wizard set it:  python3 tools/wizard.py --geometry")
        rc = 1
    elif len(obs.track_ids) == 0:
        _bad("Boxes are detected but no tracks form (should be impossible).")
        rc = 1
    elif not obs.crossed_ids:
        _bad("Boxes are tracked but none ever crossed the counting line.")
        # Check the travel axis FIRST: with the wrong axis, everything else
        # said about the line would be confidently wrong.
        observed_axis = None
        if obs.abs_dx + obs.abs_dy > 1e-6:
            observed_axis = "x" if obs.abs_dx > obs.abs_dy else "y"
        if observed_axis and observed_axis != cfg.counting.axis:
            print(f"     Boxes travel along the {observed_axis.upper()} axis of "
                  f"the image, but counting.axis is '{cfg.counting.axis}'.")
            print(f"     -> set counting.axis: {observed_axis}")
            print("        (or let the wizard do it: python3 tools/wizard.py "
                  "--geometry)")
        elif obs.centroids:
            axis = cfg.counting.axis
            vals = [c[1] if axis == "y" else c[0] for c in obs.centroids]
            # Robust range: the extremes are single frames of a box half out
            # of view, which would make almost any line look "in the path".
            lo, hi = _pct(vals, 0.05), _pct(vals, 0.95)
            line = cfg.counting.line_position
            print(f"     Boxes travelled mostly between {axis}={lo:.2f} and "
                  f"{axis}={hi:.2f} (fractions of the frame).")
            print(f"     The counting line sits at {axis}={line}.")
            if line < lo or line > hi:
                print("     -> the line is OUTSIDE the path the boxes take.")
                print(f"        Set counting.line_position: {(lo+hi)/2:.2f}")
            elif line - lo < 0.08 or hi - line < 0.08:
                print("     -> the line is too close to where boxes enter or leave")
                print("        view. A box touching the edge of the view cannot be")
                print("        counted (that rule is what stops the packer's arm")
                print("        being counted as a box).")
                print(f"        Set counting.line_position: {(lo+hi)/2:.2f}")
            else:
                print("     -> the line is in the path, so boxes may not travel far")
                print("        enough, or min_hits/min_travel_frac reject them.")
                print(f"        Try counting.min_travel_frac: "
                      f"{max(0.02, cfg.counting.min_travel_frac/2):.2f}")
        rc = 1
    elif obs.counted == 0:
        _bad("Boxes crossed the line but were not counted.")
        drift = obs.travel_dy if cfg.counting.axis == "y" else obs.travel_dx
        moving = "positive" if drift > 0 else "negative"
        print(f"     Boxes are travelling in the {_BOLD}{moving}{_RESET} direction; "
              f"counting.direction is '{cfg.counting.direction}'.")
        if moving != cfg.counting.direction and cfg.counting.direction != "any":
            print(f"     -> set counting.direction: {moving}")
        else:
            print(f"     -> try lowering counting.min_travel_frac "
                  f"(now {cfg.counting.min_travel_frac}) or tracking.min_hits "
                  f"(now {cfg.tracking.min_hits})")
        rc = 1
    else:
        _ok(f"Counting works: {obs.counted} boxes counted in this test.")

    # secondary warnings that do not stop counting
    if fg_p95 > 0.5:
        _warn(f"{fg_p95*100:.0f}% of the ROI is foreground at times — the model "
              "may be mis-adapted, or the ROI includes moving machinery.")
    if obs.max_hits and obs.max_hits < cfg.tracking.min_hits:
        _warn(f"Best track had only {obs.max_hits} detections but "
              f"tracking.min_hits is {cfg.tracking.min_hits}; boxes may pass "
              "too quickly. Raise camera.fps or lower min_hits.")

    if cfg.packing.enabled:
        _h("Packing station")
        if not obs.slow_points:
            _warn("No box was ever seen stopped — no packing session can start.")
            print("     Pieces and pack times will stay empty. If boxes do stop,")
            print("     raise packing.dwell_speed_px (now "
                  f"{cfg.packing.dwell_speed_px}).")
        else:
            xs = [p[0] for p in obs.slow_points]
            ys = [p[1] for p in obs.slow_points]
            print(f"  Boxes stopped around x={sum(xs)/len(xs):.2f} "
                  f"y={sum(ys)/len(ys):.2f} (fractions of the frame).")
            zx, zy, zw, zh = cfg.packing.zone
            inside = sum(1 for x, y in obs.slow_points
                         if zx <= x <= zx + zw and zy <= y <= zy + zh)
            if inside == 0:
                _bad("...which is OUTSIDE packing.zone — sessions never start.")
                print("     Fix with:  python3 tools/wizard.py --geometry")
            else:
                _ok(f"{inside}/{len(obs.slow_points)} stopped samples are inside "
                    "packing.zone.")
            print("\n  For arm/pad tuning run:")
            print("     python3 tools/calibrate.py --packing --seconds 60")

    print(f"\n{_DIM}Re-run this after every change: "
          f"python3 tools/wizard.py --diagnose{_RESET}")
    return rc


# ------------------------------------------------------------------- live

def live(cfg, source_path, seconds: float) -> int:
    _h("LIVE VIEW — what the camera sees")
    rx, ry, rw, rh = cfg.processing.roi
    print("Characters show movement ANYWHERE in the picture. Boxes should")
    print("appear as solid '@' shapes; those the detector accepts are outlined")
    print("in green. Only movement INSIDE the blue border is actually used —")
    print(f"that border is processing.roi, currently [{rx}, {ry}, {rw}, {rh}].")
    print("If your boxes travel outside it, that is what to fix. Ctrl-C stops.\n")
    cols, rows = asciiview.grid_size()
    # Observe the WHOLE frame regardless of the configured ROI, and draw the
    # real ROI only as an overlay. Otherwise a badly-placed ROI blanks the very
    # display the user needs in order to correct it — a dead end.
    import dataclasses
    full_cfg = dataclasses.replace(
        cfg, processing=dataclasses.replace(cfg.processing,
                                            roi=[0.0, 0.0, 1.0, 1.0]))
    obs = Observer(full_cfg, source_path)
    obs.start()
    deadline = time.monotonic() + seconds
    fh = fw = 0
    try:
        sys.stdout.write(f"{_CSI}?25l")
        first = True
        while time.monotonic() < deadline:
            if not obs.step():
                break
            if obs.last_mask is None:
                continue
            fh, fw = obs.frame_shape
            axis = cfg.counting.axis
            span = fw if axis == "x" else fh
            zone = None
            if cfg.packing.enabled:
                zx, zy, zw, zh = cfg.packing.zone
                zone = (int(zx * fw), int(zy * fh), int(zw * fw), int(zh * fh))
            art = asciiview.render(
                obs.last_mask, (fh, fw), cols, rows,
                roi_px=(int(rx * fw), int(ry * fh), int(rw * fw), int(rh * fh)),
                line=(axis, cfg.counting.line_position * span),
                zone_px=zone, boxes=obs.last_boxes)
            if not first:
                sys.stdout.write(f"{_CSI}{len(art) + 3}A")
            first = False
            sys.stdout.write("\r" + "\n".join(a + f"{_CSI}K" for a in art) + "\n")
            sys.stdout.write(asciiview.legend() + f"{_CSI}K\n")
            sys.stdout.write(
                f"{_DIM}frame {obs.frames}   accepted boxes now: "
                f"{len(obs.last_boxes)}   tracks: {len(obs.track_ids)}   "
                f"counted: {obs.counted}{_RESET}{_CSI}K\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(f"{_CSI}?25h")
        sys.stdout.flush()
        obs.stop()
    print(f"\nSeen: {obs.frames} frames, {len(obs.track_ids)} tracks, "
          f"{obs.counted} counted.")
    return 0


# --------------------------------------------------------------- geometry

def geometry(cfg, cfg_path: Path, source_path, seconds: float,
             assume_yes: bool = False, _pass: int = 1) -> int:
    _h("GEOMETRY — watching your line to set the counting line and zone"
       + (f"  (pass {_pass})" if _pass > 1 else ""))
    print("Run several boxes through the FULL cycle: arriving, being packed,")
    print("and leaving. The more complete cycles, the better the result.\n")
    obs = _observe(cfg, source_path, seconds, "Observing")

    centroids = obs.centroids
    sizes = obs.kept_blob_fracs
    bootstrapped = False
    if not centroids:
        # The current size limits reject everything, so fall back to the raw
        # blobs. Keep only the big ones: real boxes are the largest things
        # moving on the belt, the rest is sensor noise.
        typical = _pct(obs.frame_max_blob, 0.50) or 0.0
        if typical < _NOISE_FRAC:
            _bad(f"The biggest moving thing was only {typical*100:.2f}% of the "
                 "region of interest — that is noise or debris, not a box.")
            print("  Nothing box-sized passed while this was measuring. Send")
            print("  boxes down the belt DURING the test, and check the region")
            print("  of interest covers the belt:")
            print("     python3 tools/wizard.py --live")
            return 1
        usable = [b for b in obs.raw_blobs if b[0] >= typical * 0.4]
        if usable:
            bootstrapped = True
            centroids = [(b[1], b[2]) for b in usable]
            sizes = [b[0] for b in usable]
            _warn("The current size limits reject every blob, so the geometry "
                  "was inferred from raw detections instead.")

    if not centroids:
        _bad("No boxes were seen at all, so geometry cannot be inferred.")
        print("Nothing moved, or the detector sees nothing. Check with:")
        print("  python3 tools/wizard.py --live")
        return 1

    if len(centroids) < _MIN_SAMPLES:
        _bad(f"Only {len(centroids)} box sightings in {seconds:.0f}s — too "
             "little to calibrate from.")
        print("  Run more boxes through while this is measuring, or raise")
        print("  --seconds. Nothing has been changed.")
        return 1

    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    updates = {}

    _h("What was observed")
    print(f"  {len(centroids)} box sightings")
    print(f"  boxes travelled x={min(xs):.2f}..{max(xs):.2f}  "
          f"y={min(ys):.2f}..{max(ys):.2f}")

    # Which way does the belt run in the image? Use the spread of positions
    # (works even in bootstrap mode, where no tracks exist).
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    if obs.abs_dx + obs.abs_dy > 1e-6:
        observed_axis = "x" if obs.abs_dx > obs.abs_dy else "y"
    else:
        observed_axis = "x" if span_x > span_y else "y"
    if observed_axis != cfg.counting.axis:
        _warn(f"Boxes travel along the {observed_axis.upper()} axis of the "
              f"image, but counting.axis is '{cfg.counting.axis}'.")
        updates["counting.axis"] = observed_axis
    axis = observed_axis
    print(f"  travel axis: {axis}")

    drift = obs.travel_dy if axis == "y" else obs.travel_dx
    # Direction needs tracking; in bootstrap mode there are no tracks yet.
    direction = None
    if abs(drift) > 1e-6:
        direction = "positive" if drift > 0 else "negative"
        print(f"  travel direction along {axis}: {direction}")
    else:
        print(f"  travel direction along {axis}: not determined "
              "(no tracks formed yet)")

    # blob sizes -> area limits. The two limits must be ordered by
    # construction: the floor on the minimum and the rounding of the maximum
    # are independent, and a config with min >= max fails validation, so the
    # counter would refuse to start at all.
    if sizes:
        lo, hi = _pct(sizes, 0.10), _pct(sizes, 0.90)
        lo_v = round(max(_NOISE_FRAC, lo * 0.5), 4)
        hi_v = round(min(0.9, max(hi * 2.5, lo_v * 4)), 3)
        updates["processing.min_area_frac"] = lo_v
        updates["processing.max_area_frac"] = hi_v
        print(f"  box area p10={lo:.4f} p90={hi:.4f} (fraction of ROI)")

    # counting direction
    if direction and cfg.counting.direction not in (direction, "any"):
        updates["counting.direction"] = direction

    # packing zone from where boxes actually stopped (robust percentiles, so
    # one stray sample cannot stretch the zone across the frame)
    dwell = None
    zone = None
    if obs.slow_points:
        sx = [p[0] for p in obs.slow_points]
        sy = [p[1] for p in obs.slow_points]
        dwell = (sum(sx) / len(sx), sum(sy) / len(sy))
        print(f"  boxes stopped around x={dwell[0]:.2f} y={dwell[1]:.2f} "
              f"({len(obs.slow_points)} samples)")
        if cfg.packing.enabled:
            pad = 0.10
            x_lo, x_hi = _pct(sx, 0.02), _pct(sx, 0.98)
            y_lo, y_hi = _pct(sy, 0.02), _pct(sy, 0.98)
            zx = max(0.0, x_lo - pad)
            zy = max(0.0, y_lo - pad)
            zone = [zx, zy,
                    min(1.0 - zx, (x_hi - x_lo) + 2 * pad),
                    min(1.0 - zy, (y_hi - y_lo) + 2 * pad)]
    else:
        print("  no stopping point seen (no packing session would start)")

    # counting line: downstream of the dwell point, inside the observed path
    travelled = ys if axis == "y" else xs
    lo_t, hi_t = _pct(travelled, 0.02), _pct(travelled, 0.98)
    if dwell is not None and cfg.packing.enabled:
        stop_at = dwell[1] if axis == "y" else dwell[0]
        line = (stop_at + hi_t) / 2 if direction != "negative" else (stop_at + lo_t) / 2
    else:
        line = (lo_t + hi_t) / 2
    line = round(min(0.92, max(0.08, line)), 2)
    updates["counting.line_position"] = line

    # Keep the zone clear of the counting line: a box must finish packing
    # before it is counted, and the two regions must not overlap. Every branch
    # must still leave the zone inside the frame, or the written config fails
    # validation and the counter will not start.
    if zone is not None:
        zx, zy, zw, zh = zone
        gap = 0.04
        if axis == "y":
            if direction != "negative":
                zh = min(zh, max(0.05, line - gap - zy))
            else:
                new_zy = min(max(zy, line + gap), 0.94)
                zh = max(0.05, zy + zh - new_zy)
                zy = new_zy
        else:
            if direction != "negative":
                zw = min(zw, max(0.05, line - gap - zx))
            else:
                new_zx = min(max(zx, line + gap), 0.94)
                zw = max(0.05, zx + zw - new_zx)
                zx = new_zx
        zx = min(max(0.0, zx), 0.94)
        zy = min(max(0.0, zy), 0.94)
        zw = min(max(0.05, zw), 1.0 - zx)
        zh = min(max(0.05, zh), 1.0 - zy)
        updates["packing.zone"] = [round(v, 2) for v in (zx, zy, zw, zh)]

    _h("Proposed configuration")
    if not updates:
        _ok("Nothing needs changing — the current values already fit.")
        return 0
    for key, value in updates.items():
        print(f"  {key}: {format_value(value)}")

    # Never write something the counter cannot load: check the proposal on a
    # scratch copy first. Turning "counts nothing" into "will not start" would
    # be far worse than doing nothing.
    problems = _validate_proposal(cfg_path, updates)
    if problems:
        _bad("The values worked out here do not form a valid configuration:")
        for p in problems:
            print(f"     - {p}")
        print("\n  Nothing has been changed. This usually means too little was")
        print("  seen; run more boxes past and try again with --seconds 90.")
        return 1

    print(f"\n{_DIM}The ROI is deliberately left alone — only you can see what is")
    print(f"belt and what is machinery. Set processing.roi by eye using:")
    print(f"  python3 tools/wizard.py --live{_RESET}")

    if not assume_yes:
        try:
            answer = input(f"\nWrite these into {cfg_path}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes"):
            print("Nothing written.")
            return 0

    bak, changes = apply_to_file(cfg_path, updates)
    print(f"\nBacked up to {bak}")
    print(f"{_DIM}(to undo everything:  cp {bak} {cfg_path}){_RESET}")
    for c in changes:
        print(f"  {c}")

    try:
        new_cfg = load_config(cfg_path)
    except Exception as exc:                       # should be unreachable
        _bad(f"The values just written do not load: {exc}")
        print(f"\n  Undo with:  cp {bak} {cfg_path}")
        return 1

    if bootstrapped and _pass < 2:
        # The first pass only repaired detection, so no tracks existed and
        # the travel direction and packing zone could not be learned. Now
        # that boxes are detectable, look again.
        print(f"\n{_DIM}Detection was repaired; observing again to learn the "
              f"travel direction and packing zone...{_RESET}")
        return geometry(new_cfg, cfg_path, source_path, seconds,
                        assume_yes=assume_yes, _pass=_pass + 1)

    if bootstrapped:
        # Pass 2 still had to bootstrap, so detection is NOT actually fixed.
        # Saying "done" here would be a lie the user cannot check.
        _warn("Detection is still not accepting boxes after this change.")
        print("  The numbers above are the best guess from raw motion, but")
        print("  something more basic is wrong — most likely the region of")
        print("  interest is not on the belt. Look at what the camera sees:")
        print("     python3 tools/wizard.py --live")
        return 1

    print(f"\n{_GREEN}Config updated.{_RESET} Verify with:")
    print("  python3 tools/wizard.py --diagnose")
    return 0


# ------------------------------------------------------------------ guided

def guided(cfg, cfg_path: Path, source_path, seconds: float) -> int:
    print(f"{_BOLD}Box counter — guided calibration{_RESET}")
    print("Four steps. Have the belt running and boxes ready.\n")
    print("  1. check the camera")
    print("  2. look at what the detector sees, and set the ROI")
    print("  3. learn the geometry from real boxes")
    print("  4. verify counting end to end")

    _h("STEP 1 of 4 — camera")
    obs = Observer(cfg, source_path)
    obs.start()
    ok = obs.step()
    obs.stop()
    if not ok:
        _bad("No frame from the camera. Check: rpicam-hello --list-cameras")
        return 1
    fh, fw = obs.frame_shape
    _ok(f"Camera delivers {fw}x{fh} frames.")

    _pause("Press Enter to watch the belt")
    _h("STEP 2 of 4 — what the detector sees")
    print("Send a few boxes past. Everything moving shows as '.:#@'.")
    print("Note roughly where the BELT is, in fractions of the frame:")
    print("the ROI should cover belt only — no rollers, rails or floor.\n")
    live(cfg, source_path, min(seconds, 25))
    print("\nIf the ROI (blue border) is not on belt only, edit config.yaml:")
    print(f"  {_BOLD}processing.roi: [x, y, width, height]{_RESET}   (fractions)")
    print("then re-run this wizard.")

    _pause("Press Enter to learn the geometry")
    rc = geometry(cfg, cfg_path, source_path, seconds)
    if rc:
        return rc

    _pause("Press Enter to verify")
    cfg = load_config(cfg_path)          # reload what we just wrote
    _h("STEP 4 of 4 — verification")
    return diagnose(cfg, source_path, seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/config.yaml")
    ap.add_argument("--source", help="video file instead of the camera")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="apply suggested config changes without asking")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(cfg_path)

    if args.diagnose:
        return diagnose(cfg, args.source, args.seconds)
    if args.live:
        return live(cfg, args.source, args.seconds)
    if args.geometry:
        return geometry(cfg, cfg_path, args.source, args.seconds, args.yes)
    return guided(cfg, cfg_path, args.source, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
