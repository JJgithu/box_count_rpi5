"""Main processing loop: capture -> detect -> track -> count -> record."""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .annotate import draw_overlay, mask_to_bgr
from .camera import FrameSource, create_source
from .config import AppConfig
from .counter import LineCounter
from .detector import BoxDetector
from .gpio_out import GpioPulse
from .packing import PackingMonitor
from .storage import CountStore
from .tracker import CentroidTracker
from .webui import SharedState, WebUI

log = logging.getLogger(__name__)

_CAMERA_MAX_FAILURES = 30       # consecutive read failures before restart


def _sd_notify(state: str) -> None:
    """Send a message to the systemd notify socket, or do nothing off-systemd.

    Lets systemd's WatchdogSec kill+restart the process if the main loop ever
    hangs (e.g. a CSI camera stall where capture blocks with no exception) —
    the one failure mode a same-process watchdog cannot recover from.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        if addr.startswith("@"):            # Linux abstract namespace socket
            addr = "\0" + addr[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(state.encode())
    except OSError:
        pass


class Pipeline:
    def __init__(self, cfg: AppConfig,
                 source: Optional[FrameSource] = None,
                 video_override: Optional[str] = None,
                 display: bool = False,
                 enable_web: Optional[bool] = None,
                 max_frames: Optional[int] = None):
        self.cfg = cfg
        self.source = source or create_source(cfg.camera, video_override)
        self.display = display
        self.max_frames = max_frames

        self.detector = BoxDetector(cfg.processing)
        self.tracker: Optional[CentroidTracker] = None    # built on first frame
        self.counter: Optional[LineCounter] = None        # (needs frame size)
        self.packing: Optional[PackingMonitor] = None     # (needs frame size)
        self._pack_history: deque = deque(maxlen=50)      # (pieces, pack_s)

        self.store = CountStore(cfg.output.data_dir,
                                use_sqlite=cfg.output.sqlite,
                                use_csv=cfg.output.csv)
        self.gpio = GpioPulse(cfg.gpio)

        self.total = self.store.total()      # resume running totals on restart
        self.pieces_total = self.store.pieces_total()
        self.session_count = 0
        self._event_times: deque = deque(maxlen=600)   # for boxes/min
        self._reset_requested = False
        self._stop = False
        self._start_ts = time.monotonic()
        self._fps = 0.0

        self.state = SharedState()
        self.web: Optional[WebUI] = None
        if enable_web if enable_web is not None else cfg.web.enabled:
            self.web = WebUI(cfg.web, self.state,
                             reset_cb=self.request_reset,
                             recent_cb=self.store.recent)

    # -- control ---------------------------------------------------------

    def request_reset(self) -> None:
        """Thread-safe: the web UI calls this; the loop applies it."""
        self._reset_requested = True

    def request_stop(self, *_args) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    # -- geometry --------------------------------------------------------

    def _init_geometry(self, frame) -> None:
        fh, fw = frame.shape[:2]
        diag = (fw ** 2 + fh ** 2) ** 0.5
        tcfg = self.cfg.tracking
        self.tracker = CentroidTracker(
            max_distance=tcfg.max_distance_frac * diag,
            max_disappeared=tcfg.max_disappeared)

        ccfg = self.cfg.counting
        span = fw if ccfg.axis == "x" else fh
        # Detections are clipped to the detector's ROI, so the arm-exclusion
        # bounds must be the ROI's extent along the travel axis (the detector
        # is initialized before this runs, so roi_px is valid).
        rx0, ry0, rw, rh = self.detector.roi_px
        bounds = (rx0, rx0 + rw) if ccfg.axis == "x" else (ry0, ry0 + rh)
        self.counter = LineCounter(
            axis=ccfg.axis,
            line_px=ccfg.line_position * span,
            hysteresis_px=ccfg.hysteresis_frac * span,
            direction=ccfg.direction,
            min_travel_px=ccfg.min_travel_frac * span,
            min_hits=tcfg.min_hits,
            bounds_px=bounds)
        if self.cfg.packing.enabled:
            self.packing = PackingMonitor(self.cfg.packing, (fh, fw))
            log.info("Packing monitor on: zone %s px", self.packing.zone_px)
        log.info("Geometry: frame %dx%d, line at %s=%.0f px, hysteresis %.0f px",
                 fw, fh, ccfg.axis, self.counter.line, self.counter.hyst)

    def _media_time(self) -> float:
        t = self.source.media_time()
        return t if t is not None else time.monotonic()

    def _reset_scene_state(self) -> None:
        """Drop state built on a scene that no longer exists (camera restart
        or background relearn). Counts already taken are unaffected."""
        if self.tracker is not None:
            self.tracker.tracks.clear()
        if self.packing is not None:
            self.packing.reset()

    # -- stats -----------------------------------------------------------

    def _rate_per_min(self) -> float:
        now = time.time()
        cutoff = now - 60.0
        while self._event_times and self._event_times[0] < cutoff:
            self._event_times.popleft()
        return float(len(self._event_times))

    def _stats(self) -> dict:
        s = {
            "total": self.total,
            "session": self.session_count,
            "rate_per_min": self._rate_per_min(),
            "fps": round(self._fps, 1),
            "uptime_s": int(time.monotonic() - self._start_ts),
        }
        if self.packing is not None:
            s["packing"] = True
            s["pieces_total"] = self.pieces_total
            s["expected_pieces"] = self.cfg.packing.expected_pieces or None
            hist = [p for p in self._pack_history if p[0] is not None]
            if hist:
                last = hist[-1]
                s["last_pieces"] = last[0]
                s["last_pack_s"] = last[1]
                s["avg_pieces"] = round(sum(p[0] for p in hist) / len(hist), 1)
                times = [p[1] for p in hist if p[1] is not None]
                if times:
                    s["avg_pack_s"] = round(sum(times) / len(times), 1)
            s.update(self.packing.live_stats(self._media_time()))
        return s

    # -- main loop -------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        self.source.start()
        if self.web is not None:
            self.web.start()

        frame_idx = 0
        failures = 0
        last_heartbeat = time.monotonic()
        fps_clock = time.monotonic()
        wd_clock = 0.0
        _sd_notify("READY=1")           # tell systemd startup is complete
        log.info("Pipeline running (resumed total: %d)", self.total)

        try:
            while not self._stop:
                frame = self.source.read()
                if frame is None:
                    if self.source.is_live:
                        # A live camera returning None is a transient failure
                        # (USB glitch, dropped CSI frame). Coast, and after a
                        # run of failures fully restart the camera.
                        failures += 1
                        if failures >= _CAMERA_MAX_FAILURES:
                            log.error("Camera unresponsive, restarting it")
                            self.source.stop()
                            time.sleep(2.0)
                            self.source.start()
                            # The restart re-runs auto-exposure and may lock to
                            # a different image; rebuild the background model
                            # and re-warm so stale-model garbage is never
                            # counted, and drop tracker/packing state built on
                            # the old scene.
                            self.detector.reset()
                            self._reset_scene_state()
                            frame_idx = 0
                            failures = 0
                        continue
                    break  # video file ended
                failures = 0
                frame_idx += 1

                detections, mask = self.detector.process(frame)
                if self.detector.relearned:
                    # The background model was rebuilt mid-run (permanent
                    # scene change): tracks and packing sessions built on the
                    # old scene are no longer meaningful.
                    self.detector.relearned = False
                    self._reset_scene_state()
                if frame_idx <= cfg.processing.warmup_frames:
                    detections = []   # let the background model settle

                if self.tracker is None:
                    self._init_geometry(frame)
                tracks = self.tracker.update(detections)

                media_t = self._media_time()
                if self.packing is not None and frame_idx > cfg.processing.warmup_frames:
                    # Piece counting must never take down box counting.
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        fh, fw = frame.shape[:2]
                        x0, y0, rw, rh = self.detector.roi_px
                        full_mask = np.zeros((fh, fw), np.uint8)
                        full_mask[y0:y0 + rh, x0:x0 + rw] = mask
                        self.packing.update(media_t, gray, full_mask, tracks)
                    except Exception:
                        log.exception("Packing monitor failed this frame; continuing")

                events = []
                if frame_idx > cfg.processing.warmup_frames:
                    events = self.counter.update(tracks)
                for ev in events:
                    self.total += 1
                    self.session_count += 1
                    self._event_times.append(ev.ts)
                    if self.packing is not None:
                        try:
                            session = self.packing.claim_for_track(ev.track_id, media_t)
                        except Exception:
                            log.exception("Packing claim failed; box counted without pieces")
                            session = None
                        if session is not None:
                            ev.pieces = session.pieces
                            ev.pack_seconds = session.pack_seconds
                            self.pieces_total += session.pieces
                        self._pack_history.append((ev.pieces, ev.pack_seconds))
                    self.store.record(ev)
                    self.gpio.pulse()
                    if ev.pieces is not None:
                        log.info("BOX #%d (track %d, %dx%d px) — %d pieces, "
                                 "packed in %.1f s",
                                 self.total, ev.track_id, ev.bbox[2], ev.bbox[3],
                                 ev.pieces, ev.pack_seconds or 0.0)
                    else:
                        log.info("BOX #%d (track %d, %dx%d px)",
                                 self.total, ev.track_id, ev.bbox[2], ev.bbox[3])

                if self._reset_requested:
                    self._reset_requested = False
                    self.store.reset()
                    self.total = 0
                    self.pieces_total = 0
                    self._pack_history.clear()

                # fps (exponential moving average)
                now = time.monotonic()
                dt = now - fps_clock
                fps_clock = now
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst

                pack_overlay = None
                if self.packing is not None:
                    ses = self.packing.session
                    pack_overlay = {
                        "zone_px": self.packing.zone_px,
                        "session_bbox": ses.bbox if ses else None,
                        "pieces": ses.pieces if ses else 0,
                        "elapsed_s": (media_t - ses.start_t) if ses else 0.0,
                        "hand_in": self.packing.hand_in if ses else False,
                    }
                annotated = draw_overlay(frame, tracks, self.detector.roi_px,
                                         cfg.counting.axis,
                                         self.counter.line if self.counter else 0,
                                         self.total, self._fps, self._rate_per_min(),
                                         packing=pack_overlay)
                self.state.update(annotated, mask_to_bgr(mask), self._stats())

                if self.display:
                    cv2.imshow("boxcounter", annotated)
                    cv2.imshow("mask", mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if now - last_heartbeat >= cfg.output.heartbeat_seconds:
                    last_heartbeat = now
                    log.info("heartbeat: total=%d rate=%.1f/min fps=%.1f tracks=%d",
                             self.total, self._rate_per_min(), self._fps, len(tracks))

                # Pet the systemd watchdog ~1x/s; if the loop ever hangs, the
                # pings stop and systemd restarts us.
                if now - wd_clock >= 1.0:
                    wd_clock = now
                    _sd_notify("WATCHDOG=1")

                if self.max_frames and frame_idx >= self.max_frames:
                    break
        finally:
            _sd_notify("STOPPING=1")
            self.source.stop()
            self.store.close()
            self.gpio.close()
            if self.display:
                cv2.destroyAllWindows()

        summary = {"frames": frame_idx, "total": self.total,
                   "session": self.session_count, "fps": round(self._fps, 1)}
        log.info("Pipeline stopped: %s", summary)
        return summary
