"""Packing-station monitor: counts pieces placed into each box and times the pack.

How it works, per frame, with no ML:

1. A box track that stops inside the packing zone (fully inside the frame,
   nearly zero velocity for a few frames) becomes the ACTIVE box. Its bounding
   box is frozen for the whole session — later frames may see the box blob
   merged with the packer's arm, and the frozen bbox keeps the geometry stable.

2. Arm detection: a ring-shaped band around the frozen bbox is watched in the
   foreground mask. An arm reaching into the box must cross this band (arms
   connect to the frame edge), so a sustained rise of foreground in the band
   above its baseline means "hand present". The baseline absorbs static
   neighbors such as the next box queuing behind.

3. Insertion confirmation: while the box is stationary, ANY motion inside its
   interior is the packer's hand or a dropped pad (frame differencing).
   A completed arm visit counts as an insertion only if enough interior motion
   happened during it — an arm passing near the box without touching its
   contents does not count. Optionally the interior must also look different
   after the visit (appearance check).

4. When the box departs the zone, the session ends: pieces = insertions x
   pieces_per_visit, pack time = arrival -> departure. The finished session is
   attached to the box's count event when it crosses the counting line
   (matched by track id, falling back to first-in-first-out order, which the
   conveyor preserves).

Known limitation (inherent to camera-only counting): one reach that places
two pads counts once. If your process packs fixed bundles per reach, set
pieces_per_visit accordingly.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PackingConfig
from .tracker import Track

log = logging.getLogger(__name__)

_EDGE_MARGIN = 2          # bbox closer than this to the frame edge = not a box
_REBIND_IOU = 0.3         # overlap needed to re-attach a session to a new track
_BASELINE_ALPHA = 0.02    # slow EMA for the ring foreground baseline


@dataclass
class PackSession:
    track_id: int
    bbox: Tuple[int, int, int, int]           # frozen at activation
    start_t: float                            # media time, seconds
    end_t: Optional[float] = None
    pieces: int = 0
    visits: int = 0                            # completed arm visits (incl. non-counting)
    claimed: bool = False
    reason: str = ""                           # how the session ended

    @property
    def pack_seconds(self) -> Optional[float]:
        return None if self.end_t is None else max(0.0, self.end_t - self.start_t)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _coverage(a: Tuple[int, int, int, int], ref: Tuple[int, int, int, int]) -> float:
    """Fraction of `ref`'s area covered by `a` (containment, not IoU)."""
    ax, ay, aw, ah = a
    rx, ry, rw, rh = ref
    ix = max(0, min(ax + aw, rx + rw) - max(ax, rx))
    iy = max(0, min(ay + ah, ry + rh) - max(ay, ry))
    area = rw * rh
    return (ix * iy) / area if area > 0 else 0.0


class PackingMonitor:
    def __init__(self, cfg: PackingConfig, frame_shape: Tuple[int, int]):
        self.cfg = cfg
        fh, fw = frame_shape
        self.fw, self.fh = fw, fh
        zx, zy, zw, zh = cfg.zone
        self.zone_px = (int(zx * fw), int(zy * fh),
                        max(1, int(zw * fw)), max(1, int(zh * fh)))

        self.session: Optional[PackSession] = None
        self.completed: Deque[PackSession] = deque(maxlen=20)

        # candidate tracking (before activation)
        self._slow: Dict[int, Tuple[int, float]] = {}   # tid -> (count, first_t)

        # per-session working state
        self._hand_in = False
        self._enter_streak = 0
        self._exit_streak = 0
        self._visit_frames = 0
        self._visit_max_motion = 0.0
        self._ring_baseline = 0.0
        self._depart_streak = 0
        self._depart_start_t = 0.0
        self._missing_frames = 0
        self._last_seen_t = 0.0
        self._session_clock_t = 0.0
        self._still_frames = 0
        self._prev_interior: Optional[np.ndarray] = None
        self._idle_interior: Optional[np.ndarray] = None

    # -- geometry ---------------------------------------------------------

    def _in_zone(self, cx: float, cy: float) -> bool:
        zx, zy, zw, zh = self.zone_px
        return zx <= cx <= zx + zw and zy <= cy <= zy + zh

    def _interior_rect(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.session.bbox
        dx = int(w * self.cfg.interior_inset_frac)
        dy = int(h * self.cfg.interior_inset_frac)
        return (x + dx, y + dy, max(1, w - 2 * dx), max(1, h - 2 * dy))

    def _ring_frac(self, mask: np.ndarray) -> float:
        """Foreground fraction of the band around the frozen bbox."""
        x, y, w, h = self.session.bbox
        r = self.cfg.ring_px
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(self.fw, x + w + r), min(self.fh, y + h + r)
        outer = mask[y0:y1, x0:x1]
        outer_area = outer.size
        inner = mask[max(0, y):min(self.fh, y + h), max(0, x):min(self.fw, x + w)]
        ring_area = outer_area - inner.size
        if ring_area <= 0:
            return 0.0
        fg = cv2.countNonZero(outer) - cv2.countNonZero(inner)
        return max(0, fg) / float(ring_area)

    def _interior_crop(self, gray: np.ndarray) -> np.ndarray:
        x, y, w, h = self._interior_rect()
        x0, y0 = max(0, x), max(0, y)
        return gray[y0:y0 + h, x0:x0 + w]

    def _interior_motion(self, gray: np.ndarray) -> float:
        cur = self._interior_crop(gray)
        prev = self._prev_interior
        self._prev_interior = cur.copy()
        if prev is None or prev.shape != cur.shape:
            return 0.0
        diff = cv2.absdiff(cur, prev)
        moving = cv2.countNonZero(
            cv2.threshold(diff, self.cfg.motion_threshold, 255, cv2.THRESH_BINARY)[1])
        return moving / float(cur.size)

    # -- session lifecycle ------------------------------------------------

    def _try_activate(self, t: float, tracks: List[Track],
                      fg_mask: np.ndarray) -> None:
        cfg = self.cfg
        seen = set()
        for tr in tracks:
            if tr.misses > 0:
                continue
            x, y, w, h = tr.bbox
            cx, cy = tr.centroid
            speed = (tr.velocity[0] ** 2 + tr.velocity[1] ** 2) ** 0.5
            fully_inside = (x > _EDGE_MARGIN and y > _EDGE_MARGIN
                            and x + w < self.fw - _EDGE_MARGIN
                            and y + h < self.fh - _EDGE_MARGIN)
            # A real box was seen ARRIVING; a stationary ghost blob (e.g.
            # left behind by a background-model rebuild) is born parked and
            # never moved, so it can never capture a session.
            arrived = ((cx - tr.start_centroid[0]) ** 2
                       + (cy - tr.start_centroid[1]) ** 2) ** 0.5 >= cfg.min_arrival_px
            if not (self._in_zone(cx, cy) and fully_inside and arrived
                    and speed < cfg.dwell_speed_px):
                continue
            seen.add(tr.track_id)
            count, first_t = self._slow.get(tr.track_id, (0, t))
            count += 1
            self._slow[tr.track_id] = (count, first_t)
            if count >= cfg.dwell_frames:
                self._start_session(tr, first_t, fg_mask)
                return
        # drop candidates that stopped qualifying
        for tid in [k for k in self._slow if k not in seen]:
            del self._slow[tid]

    def _start_session(self, tr: Track, start_t: float,
                       fg_mask: np.ndarray) -> None:
        # If this same box already timed out earlier (packer's long break),
        # resume that session instead of fragmenting into duplicates.
        resumed = None
        for s in reversed(self.completed):
            if (not s.claimed and s.track_id == tr.track_id
                    and s.reason == "timeout"):
                resumed = s
                break
        if resumed is not None:
            self.completed.remove(resumed)
            resumed.end_t = None
            resumed.reason = ""
            self.session = resumed
            log.info("Packing session resumed (track %d, %d pieces so far)",
                     tr.track_id, resumed.pieces)
        else:
            self.session = PackSession(track_id=tr.track_id,
                                       bbox=tuple(int(v) for v in tr.bbox),
                                       start_t=start_t)
            log.info("Packing session started (track %d, bbox %s)",
                     tr.track_id, self.session.bbox)
        # The inactivity timeout runs from (re)activation, not from the
        # original arrival — otherwise a resumed session would instantly time
        # out again. pack_seconds still spans arrival -> departure.
        self._session_clock_t = start_t
        self._hand_in = False
        self._enter_streak = self._exit_streak = 0
        self._visit_frames = 0
        self._visit_max_motion = 0.0
        self._still_frames = 0
        self._depart_streak = 0
        self._missing_frames = 0
        self._last_seen_t = start_t
        self._prev_interior = None
        self._idle_interior = None
        self._slow.clear()
        # Seed the arm baseline from the CURRENT scene, so anything already
        # sitting in the watch band (e.g. the next box queued close behind)
        # is ignored from the first frame instead of reading as an arm.
        self._ring_baseline = self._ring_frac(fg_mask)

    def _end_session(self, t: float, reason: str) -> None:
        if self.session is None:
            return
        if self._hand_in:
            # A visit still open when the box leaves is the box's own motion
            # sliding through the ring (or a push) — never a real insertion.
            self._hand_in = False
            self._visit_frames = 0
            self._visit_max_motion = 0.0
        s = self.session
        self.session = None
        s.end_t = t
        s.reason = reason
        exp = self.cfg.expected_pieces
        # Only judge cleanly-finished packs; a timeout/abort may resume later.
        if exp > 0 and s.pieces != exp and reason in ("departed", "crossed line"):
            log.warning("Box (track %d) left with %d pieces, expected %d",
                        s.track_id, s.pieces, exp)
        log.info("Packing session ended (%s): track %d, %d pieces, %.1f s",
                 reason, s.track_id, s.pieces, s.pack_seconds or 0.0)
        self.completed.append(s)

    def _finish_visit(self, gray: Optional[np.ndarray],
                      tracks: Optional[List[Track]] = None) -> None:
        """The arm has withdrawn: decide whether the visit was a real insertion."""
        cfg = self.cfg
        s = self.session
        self._hand_in = False
        long_enough = self._visit_frames >= cfg.min_visit_frames
        moved_inside = self._visit_max_motion >= cfg.interior_motion_frac
        # The box must still be there. If the "visit" was really the box being
        # pushed/dragged out while occluded, no track overlaps the frozen bbox
        # at visit end and no piece may be awarded.
        box_present = tracks is None or any(
            tr.misses == 0 and _iou(tr.bbox, s.bbox) > _REBIND_IOU for tr in tracks)
        changed = True
        if cfg.appearance_check and gray is not None and self._idle_interior is not None:
            cur = cv2.GaussianBlur(self._interior_crop(gray), (5, 5), 0)
            ref = self._idle_interior
            if cur.shape == ref.shape:
                changed = float(cv2.absdiff(cur, ref).mean()) >= cfg.appearance_delta
        s.visits += 1
        if long_enough and moved_inside and changed and box_present:
            s.pieces += cfg.pieces_per_visit
            log.info("Insertion #%d into box (track %d) — visit %d frames, "
                     "motion %.2f", s.pieces, s.track_id,
                     self._visit_frames, self._visit_max_motion)
        else:
            log.debug("Arm visit ignored (frames=%d motion=%.3f changed=%s box=%s)",
                      self._visit_frames, self._visit_max_motion, changed, box_present)
        if gray is not None:
            self._idle_interior = cv2.GaussianBlur(self._interior_crop(gray), (5, 5), 0)
        self._visit_frames = 0
        self._visit_max_motion = 0.0

    # -- main per-frame update --------------------------------------------

    def update(self, t: float, gray: np.ndarray, fg_mask: np.ndarray,
               tracks: List[Track]) -> None:
        """Advance the monitor by one frame.

        t is media time in seconds (video position or wall clock), gray is the
        full-frame grayscale image, fg_mask the full-frame foreground mask.
        """
        cfg = self.cfg
        if self.session is None:
            self._try_activate(t, tracks, fg_mask)
            if self.session is None:
                return

        s = self.session

        if t - self._session_clock_t > cfg.max_session_s:
            log.warning("Packing session timed out after %.0f s", cfg.max_session_s)
            self._end_session(t, "timeout")
            return

        # -- arm state machine -------------------------------------------
        ring = self._ring_frac(fg_mask)
        signal = max(0.0, ring - self._ring_baseline)
        motion = self._interior_motion(gray)

        if not self._hand_in:
            # Track the empty-scene baseline only while the arm is out.
            self._ring_baseline += _BASELINE_ALPHA * (ring - self._ring_baseline)
            if self._idle_interior is None:
                self._idle_interior = cv2.GaussianBlur(self._interior_crop(gray), (5, 5), 0)
            if signal > cfg.arm_enter_frac:
                self._enter_streak += 1
                if self._enter_streak >= cfg.enter_frames:
                    self._hand_in = True
                    self._visit_frames = self._enter_streak
                    self._visit_max_motion = motion
                    self._exit_streak = 0
            else:
                self._enter_streak = 0
        else:
            self._visit_frames += 1
            self._visit_max_motion = max(self._visit_max_motion, motion)
            # A hand at work moves; a box newly queued into the watch band
            # does not. Sustained stillness means the "hand" is really a new
            # static occupant: close the visit (it still counts if the hand
            # moved inside earlier) and adopt the new scene as the baseline.
            if motion < cfg.interior_motion_frac * 0.5:
                self._still_frames += 1
            else:
                self._still_frames = 0
            if signal < cfg.arm_exit_frac:
                self._exit_streak += 1
                if self._exit_streak >= cfg.exit_frames:
                    self._finish_visit(gray, tracks)
                    self._enter_streak = 0
            else:
                self._exit_streak = 0
            if self._hand_in and (self._still_frames >= cfg.static_exit_frames
                                  or self._visit_frames > cfg.max_visit_frames):
                log.info("Static occupant in the watch band (%d frames); "
                         "closing visit and re-baselining", self._visit_frames)
                self._finish_visit(gray, tracks)
                self._still_frames = 0
                self._enter_streak = 0
                self._ring_baseline = max(self._ring_baseline, ring)

        # -- departure / track-loss --------------------------------------
        # Departure is geometric: how much of the FROZEN bbox the bound track
        # still covers. An arm merging with the box yields a blob that
        # CONTAINS the box (coverage stays ~1 no matter how large the blob),
        # while a departing box slides off its spot (coverage falls toward 0).
        # Safe to run even while the arm FSM believes a hand is in (during a
        # departure that is really the box moving through its own ring).
        ours = next((tr for tr in tracks if tr.track_id == s.track_id), None)
        if ours is not None and ours.misses == 0:
            self._missing_frames = 0
            if _coverage(ours.bbox, s.bbox) < 0.3:
                if self._depart_streak == 0:
                    self._depart_start_t = t
                self._depart_streak += 1
                if self._depart_streak >= cfg.depart_frames:
                    self._end_session(self._depart_start_t, "departed")
            else:
                self._depart_streak = 0
                self._last_seen_t = t
            return

        # Track missing/coasting. During an arm visit the box is often
        # occluded or merged into the arm blob — never end the session then.
        if self._hand_in:
            return
        self._missing_frames += 1
        # Try to re-bind: tracking may have re-spawned the same box. Only a
        # track that plausibly IS the parked box qualifies: overlapping the
        # frozen spot, essentially stationary, and BORN at the spot — a
        # different box that travelled in from upstream is born elsewhere and
        # must not inherit this session. Best overlap wins.
        best, best_iou = None, _REBIND_IOU
        for tr in tracks:
            if tr.misses != 0:
                continue
            speed = (tr.velocity[0] ** 2 + tr.velocity[1] ** 2) ** 0.5
            if speed >= cfg.dwell_speed_px:
                continue
            bx, by, bw, bh = s.bbox
            r = cfg.ring_px
            sx, sy = tr.start_centroid
            born_here = (bx - r <= sx <= bx + bw + r and by - r <= sy <= by + bh + r)
            if not born_here:
                continue
            iou = _iou(tr.bbox, s.bbox)
            if iou > best_iou:
                best, best_iou = tr, iou
        if best is not None:
            log.debug("Packing session re-bound: track %d -> %d",
                      s.track_id, best.track_id)
            s.track_id = best.track_id
            self._missing_frames = 0
            return
        if self._missing_frames > cfg.track_grace_frames:
            self._end_session(self._last_seen_t, "track lost")

    # -- results ----------------------------------------------------------

    def claim_for_track(self, track_id: int, t: float) -> Optional[PackSession]:
        """Hand the finished session for this box to the counter's event.

        Called when a box is counted at the line. Prefers an exact track-id
        match; falls back to the oldest unclaimed session (conveyor order is
        first-in-first-out). If the box somehow reaches the line while its
        session is still open, the session is closed here.
        """
        if self.session is not None and self.session.track_id == track_id:
            self._end_session(t, "crossed line")
        # Exact-track match, newest first: if an older timeout fragment for
        # the same track exists, the recent cleanly-ended session wins.
        for s in reversed(self.completed):
            if not s.claimed and s.track_id == track_id:
                s.claimed = True
                return s
        # FIFO fallback is only trusted for sessions that ended cleanly and
        # recently — a stale "track lost"/timeout leftover from an earlier
        # anomaly must not contaminate an unrelated box.
        for s in self.completed:
            if (not s.claimed and s.reason in ("departed", "crossed line")
                    and s.end_t is not None and t - s.end_t < 120.0):
                s.claimed = True
                return s
        return None

    @property
    def hand_in(self) -> bool:
        return self._hand_in

    def reset(self) -> None:
        """Called when upstream state was rebuilt (camera restart, background
        relearn): the current scene can no longer be trusted, so abort any
        open session. 'aborted' sessions are never matched by the FIFO
        fallback, so they cannot contaminate later boxes."""
        if self.session is not None:
            self._end_session(self._last_seen_t, "aborted")
        self._slow.clear()

    def live_stats(self, t: float) -> dict:
        s = self.session
        if s is None:
            return {"packing_now": False}
        return {
            "packing_now": True,
            "current_pieces": s.pieces,
            "current_elapsed_s": round(max(0.0, t - s.start_t), 1),
            "hand_in": self._hand_in,
        }
