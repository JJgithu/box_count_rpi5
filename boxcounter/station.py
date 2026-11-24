"""Packing-station counter (counting.mode: station).

Built for a box that STOPS in front of the paper machine, gets filled with
pads, and leaves — slid along the rollers in either direction, or lifted off.
There is no counting line and no travel direction to configure: the camera
only has to see the packing spot.

Per frame, at a reduced resolution (station.work_width), with no ML:

1. Background. A picture of the EMPTY spot is learned when the counter starts
   and kept up to date: slow lighting drift is followed, things left lying
   still for a long time are absorbed, and the spot behind a box that was
   already standing there at start-up (a "ghost" once that box is taken away)
   is recognised and repaired.

2. Foreground = pixels whose colour differs from the empty spot (shadows on
   the rollers excluded). Motion = pixels that changed over the last couple
   of frames. "Still" = no motion for still_seconds.

3. A box is foreground that is still, big enough, mostly cardboard-coloured,
   centred inside the zone, not touching the picture edge the way an arm
   does, and that stays like that for confirm_seconds. Hands and arms are
   never counted as boxes: they do not stay still, they are smaller than a
   box, and they reach in from the edge.

4. While the box is there, every REACH — movement over the box, then the box
   going quiet again — is checked: if enough of the inside of the box looks
   different than just before the reach, a pad went in. The pad hanging from
   the machine is part of the empty-spot picture, not of the box, so a new
   pad appearing at the outlet is not counted by itself.

5. When most of the box's spot is empty again for depart_seconds, the box has
   left and is counted, with its pads and its pack time (box set down -> box
   gone). A box nudged during packing is followed to its new position as long
   as it still overlaps where it was.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .config import StationConfig
from .counter import CountEvent

log = logging.getLogger(__name__)

Rect = Tuple[int, int, int, int]

_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

_CAND_GAP_S = 1.0          # a candidate may vanish this long (hand passing over)
_GEOMETRY_EVERY_S = 0.5    # how often a quiet box's outline is re-checked
_GEOMETRY_KEEP = 0.9       # outline only re-measured once coverage falls below
_BACKDATE_MAX_S = 10.0     # pack time may start at most this long before
                           # the box was confirmed
_MOVED_KEEP = 0.75         # after a reach, less of the box's spot covered than
                           # this = the box itself was moved, not a pad added
_STILL_COVER = 0.985       # box covers this much of its spot = not moved yet
_HURRIED_MIN_S = 0.5       # hand busy this long before the box moved = a reach
_SHIFT_MIN_RESPONSE = 0.25 # phase-correlation peak needed to call it a shove
                           # (the silhouette must also fit better there)


def rect_iou(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def expand_rect(r: Rect, m: int, width: int, height: int) -> Rect:
    x, y, w, h = r
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(width, x + w + m), min(height, y + h + m)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def foreground(img: np.ndarray, bg_f: np.ndarray, bg8: np.ndarray,
               threshold: int, suppress_shadows: bool
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(raw, shadow-free) boolean foreground masks of img vs the background.

    A shadow darkens all three colour channels by about the same factor; a
    brown box on grey rollers changes them very unequally. A pixel darker
    than the background by a near-uniform factor is therefore a shadow."""
    # (cv2.split + cv2.max/min: numpy reductions over the 3-channel axis are
    # several times slower, and this runs on every frame.)
    c0, c1, c2 = cv2.split(cv2.absdiff(img, bg8))
    raw = cv2.max(cv2.max(c0, c1), c2) > threshold
    if not suppress_shadows:
        return raw, raw
    r0, r1, r2 = cv2.split(cv2.divide(img.astype(np.float32) + 1.0, bg_f + 1.0))
    rmin = cv2.min(cv2.min(r0, r1), r2)
    rmax = cv2.max(cv2.max(r0, r1), r2)
    shadow = (rmax < 0.97) & (rmin > 0.30) & ((rmax - rmin) < 0.15 * rmax)
    return raw, raw & ~shadow


def brown_mask(img: np.ndarray) -> np.ndarray:
    """Cardboard / kraft-paper coloured pixels. Deliberately broad (including
    red-shifted browns) so it survives warehouse lighting and white balance."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 35) | (h >= 170)) & (s >= 35) & (v >= 35)


def snapshot(gray_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Brightness plus local structure (brightness / its neighbourhood).

    The neighbourhood is about a fifth of the box across. A soft shadow (the
    packer leaning over the box) dims a whole neighbourhood evenly, so its
    structure is unchanged and it is not mistaken for a pad; a pad is smaller
    than that and stands out from its surroundings."""
    g = gray_crop.astype(np.float32)
    sigma = max(6.0, 0.2 * min(g.shape[:2]))
    base = cv2.GaussianBlur(g, (0, 0), sigma)
    return g, g / (base + 10.0)


def changed_mask(a: Tuple[np.ndarray, np.ndarray],
                 b: Tuple[np.ndarray, np.ndarray], threshold: int) -> np.ndarray:
    ga, sa = a
    gb, sb = b
    d = np.abs(ga - gb)
    return (d > threshold) & ((np.abs(sa - sb) > 0.12) | (d > 2 * threshold))


def _pads(n: int) -> str:
    return f"{n} pad" if n == 1 else f"{n} pads"


def _as_u8(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask, dtype=np.uint8)


@dataclass
class _Candidate:
    """Something still, box-sized, in the zone — not yet confirmed."""
    bbox: Rect
    area: int
    first_t: float
    last_t: float
    cx0: float
    cy0: float
    label: int = -1                 # component index this frame; -1 = unseen
    rejected_until: float = -1.0
    reason: str = ""
    early: Optional[np.ndarray] = None   # work-frame grey image of the box
                                         # before anyone reached into it
    busy: bool = False              # a hand moved over it while unconfirmed

    def age(self, t: float) -> float:
        return t - self.first_t


@dataclass
class _Session:
    """One box at the station."""
    sid: int
    start_t: float                  # relative seconds: box set down
    last_present_t: float
    bbox: Rect = (0, 0, 1, 1)
    fp: Optional[np.ndarray] = None     # footprint, bbox-local bool
    fp_area: int = 1
    outer: Rect = (0, 0, 1, 1)
    act_mask: Optional[np.ndarray] = None   # outer-local bool
    act_area: int = 1
    interior: Rect = (0, 0, 1, 1)
    fp_int: Optional[np.ndarray] = None     # interior-local bool
    fp_int_area: int = 1
    low_since: Optional[float] = None
    pads: int = 0
    reaches: int = 0
    in_reach: bool = False
    on_frames: int = 0
    reach_start_t: float = 0.0
    quiet_since: Optional[float] = None
    ref: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ref_outer: Optional[np.ndarray] = None   # grey crop of outer, same moment
    ref_t: float = 0.0
    geo_t: float = 0.0
    cov: float = 1.0
    act: float = 0.0
    # (t, coverage, motion, interior crop, outer crop, outer foreground) for
    # each frame of a reach: the reach is judged on its quietest frame, not on
    # whatever frame the quiet period happened to end on (which may already
    # show the next reach's hand coming in).
    hist: Deque = field(default_factory=lambda: deque(maxlen=90))
    recent_act: Deque = field(default_factory=lambda: deque(maxlen=3))


class StationMonitor:
    """Counts boxes at a packing station, the pads put in each, and the time."""

    def __init__(self, cfg: StationConfig, roi: List[float],
                 frame_shape: Tuple[int, int], debug: bool = False):
        self.cfg = cfg
        fh, fw = frame_shape[:2]
        self.fw, self.fh = fw, fh
        scale = min(1.0, cfg.work_width / float(fw))
        self.ww = max(32, int(round(fw * scale)))
        self.wh = max(24, int(round(fh * scale)))
        self.sx = fw / float(self.ww)        # work -> full-frame factors
        self.sy = fh / float(self.wh)

        rx, ry, rw, rh = roi
        x0, y0 = int(round(rx * self.ww)), int(round(ry * self.wh))
        x1 = min(self.ww, int(round((rx + rw) * self.ww)))
        y1 = min(self.wh, int(round((ry + rh) * self.wh)))
        self.roi: Rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        self.roi_mask = np.zeros((self.wh, self.ww), bool)
        self.roi_mask[y0:y1, x0:x1] = True
        self.roi_area = float(self.roi[2] * self.roi[3])

        zx, zy, zw, zh = cfg.zone
        self.zone = (zx * self.ww, zy * self.wh,
                     (zx + zw) * self.ww, (zy + zh) * self.wh)
        self.min_area = cfg.min_box_frac * self.roi_area
        self.max_area = cfg.max_box_frac * self.roi_area
        self.debug = debug
        self._next_sid = 1
        self._clear()

    # -- state -------------------------------------------------------------

    def _clear(self) -> None:
        self.bg_f: Optional[np.ndarray] = None
        self.bg8: Optional[np.ndarray] = None
        self._learn: List[np.ndarray] = []
        self._grays: Deque[np.ndarray] = deque(maxlen=3)
        self.last_motion: Optional[np.ndarray] = None
        self.fg_since: Optional[np.ndarray] = None
        self.sessions: List[_Session] = []
        self.candidates: List[_Candidate] = []
        self._t0: Optional[float] = None
        self._tr = 0.0
        self._global_since: Optional[float] = None
        self._frame_no = 0
        self.note = ""                       # last decision, for the panel
        self._stable: Optional[np.ndarray] = None
        self.last_reach: Optional[dict] = None
        self._dbg = None

    def reset(self) -> None:
        """Camera restarted: the old picture of the spot can't be trusted."""
        for s in self.sessions:
            log.warning("Box %d abandoned (camera restart), not counted", s.sid)
        self._clear()

    @property
    def learned(self) -> bool:
        return self.bg_f is not None

    def _set_background(self, bg: np.ndarray) -> None:
        self.bg_f = bg.astype(np.float32)
        self.bg8 = np.clip(self.bg_f + 0.5, 0, 255).astype(np.uint8)

    def _crop(self, img: np.ndarray, r: Rect) -> np.ndarray:
        x, y, w, h = r
        return img[y:y + h, x:x + w]

    def to_full(self, r: Rect) -> Rect:
        x, y, w, h = r
        return (int(round(x * self.sx)), int(round(y * self.sy)),
                int(round(w * self.sx)), int(round(h * self.sy)))

    # -- per frame ---------------------------------------------------------

    def update(self, frame: np.ndarray, t: float) -> List[CountEvent]:
        """Process one full-resolution BGR frame taken at time t (seconds).
        Returns the boxes that finished (left the station) this frame."""
        cfg = self.cfg
        if self._t0 is None:
            self._t0 = t
        tr = t - self._t0
        self._tr = tr
        self._frame_no += 1

        if frame.shape[1] != self.ww or frame.shape[0] != self.wh:
            small = cv2.resize(frame, (self.ww, self.wh), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        small = cv2.GaussianBlur(small, (3, 3), 0)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self._grays.append(gray)

        if self.bg_f is None:
            self._learn.append(small)
            if len(self._learn) >= cfg.learn_frames:
                self._set_background(np.median(np.stack(self._learn), axis=0))
                self._learn = []
                self.last_motion = np.full((self.wh, self.ww), tr, np.float64)
                self.fg_since = np.full((self.wh, self.ww), tr, np.float64)
                log.info("Learned the empty packing spot from %d frames",
                         cfg.learn_frames)
            return []

        motion = cv2.absdiff(gray, self._grays[0]) > cfg.motion_threshold
        motion = cv2.dilate(_as_u8(motion), _K3).astype(bool) & self.roi_mask
        self.last_motion[motion] = tr
        still = (tr - self.last_motion) >= cfg.still_seconds

        fg_raw, fg = foreground(small, self.bg_f, self.bg8,
                                cfg.diff_threshold, cfg.suppress_shadows)
        fg_raw &= self.roi_mask
        fg = cv2.morphologyEx(_as_u8(fg), cv2.MORPH_OPEN, _K3).astype(bool)
        fg &= self.roi_mask
        self.fg_since[~fg] = tr
        stable = fg & still
        self._stable = stable

        events: List[CountEvent] = []
        for s in list(self.sessions):
            ev = self._update_session(s, tr, small, gray, fg, motion, stable)
            if ev is not None:
                events.append(ev)
        self._update_candidates(tr, small, gray, fg, stable, motion)
        self._maintain_background(tr, small, fg_raw, still)
        if self.debug:
            self._dbg = (fg, stable, motion)
        return events

    # -- sessions ----------------------------------------------------------

    def _set_geometry(self, s: _Session, bbox: Rect, fp: np.ndarray) -> None:
        x, y, w, h = (int(v) for v in bbox)
        s.bbox = (x, y, w, h)
        s.fp = np.ascontiguousarray(fp, dtype=bool)
        s.fp_area = max(1, int(np.count_nonzero(s.fp)))
        m = max(3, int(0.08 * max(w, h)))
        s.outer = expand_rect(s.bbox, m, self.ww, self.wh)
        ox, oy, ow, oh = s.outer
        act = np.zeros((oh, ow), np.uint8)
        act[y - oy:y - oy + h, x - ox:x - ox + w] = s.fp
        # Reaches are measured on the box itself, a little inside its edge:
        # the paper hanging from the machine right next to the box sways and
        # grows, and that motion must not hold a reach open.
        s.act_mask = cv2.erode(act, _K5).astype(bool)
        if np.count_nonzero(s.act_mask) < 0.3 * s.fp_area:
            s.act_mask = act.astype(bool)        # thin outline: keep it whole
        s.act_area = max(1, int(np.count_nonzero(s.act_mask)))
        # The inside of the box, where pads land: its outline shrunk past the
        # spread flaps and the rim. Deliberately the whole rectangle, not just
        # the box-coloured part: the pad hanging from the machine covers part
        # of the box from the camera's point of view, and a pad stuffed under
        # it is only visible as the hanging pad being torn off and replaced.
        ins = int(round(self.cfg.interior_inset_frac * min(w, h)))
        iw, ih = max(4, w - 2 * ins), max(4, h - 2 * ins)
        s.interior = (x + (w - iw) // 2, y + (h - ih) // 2, iw, ih)
        s.fp_int = np.ones((ih, iw), bool)
        s.fp_int_area = iw * ih
        s.hist.clear()

    def _update_session(self, s: _Session, tr: float, small: np.ndarray,
                        gray: np.ndarray, fg: np.ndarray, motion: np.ndarray,
                        stable: np.ndarray) -> Optional[CountEvent]:
        cfg = self.cfg
        x, y, w, h = s.bbox
        s.cov = float(np.count_nonzero(fg[y:y + h, x:x + w] & s.fp)) / s.fp_area
        ox, oy, ow, oh = s.outer
        s.act = float(np.count_nonzero(
            motion[oy:oy + oh, ox:ox + ow] & s.act_mask)) / s.act_area
        s.recent_act.append(s.act)

        if tr - s.start_t > cfg.max_session_seconds:
            log.warning("Box %d has been at the station for over %.0f s; "
                        "treating it as part of the background, not counted",
                        s.sid, cfg.max_session_seconds)
            self.note = "gave up on an object that never left the spot"
            self.sessions.remove(s)
            region = np.zeros((self.wh, self.ww), bool)
            region[y:y + h, x:x + w] = s.fp
            self.bg_f[region] = small[region]
            self.bg8 = np.clip(self.bg_f + 0.5, 0, 255).astype(np.uint8)
            return None

        if s.cov >= cfg.present_cover:
            s.last_present_t = tr
        if s.cov < cfg.depart_cover:
            if s.low_since is None:
                s.low_since = tr
            if tr - s.low_since >= cfg.depart_seconds:
                # Gone. Counted right away: on a conveyor the next box can be
                # on the spot a second later and must start its own session.
                if s.in_reach and cfg.count_pads:
                    self._settle_hurried_reach(s)
                log.info("Box %d left the spot", s.sid)
                return self._finish(s)
        else:
            s.low_since = None

        if cfg.count_pads:
            self._reach_fsm(s, tr, gray, fg)
        if not s.in_reach:
            self._refresh_geometry(s, tr, gray, stable)
        return None

    def _finish(self, s: _Session) -> Optional[CountEvent]:
        cfg = self.cfg
        self.sessions.remove(s)
        dur = max(0.0, s.last_present_t - s.start_t)
        if dur < cfg.min_pack_seconds:
            log.info("Not counted: something stood at the spot only %.1f s "
                     "(station.min_pack_seconds is %.1f)", dur, cfg.min_pack_seconds)
            self.note = f"not counted: at the spot only {dur:.1f} s"
            return None
        if (cfg.count_pads and cfg.expected_pieces > 0
                and s.pads != cfg.expected_pieces):
            log.warning("Box left with %d pads, expected %d",
                        s.pads, cfg.expected_pieces)
        ev = CountEvent(ts=time.time(), track_id=s.sid, bbox=self.to_full(s.bbox),
                        area=float(s.fp_area * self.sx * self.sy), direction=0,
                        pieces=s.pads if cfg.count_pads else None,
                        pack_seconds=dur)
        log.info("Box counted: %d pads, packed in %.1f s (%d reaches)",
                 s.pads, dur, s.reaches)
        self.note = (f"box counted: {_pads(s.pads)}, packed in {dur:.1f} s"
                     if cfg.count_pads else f"box counted: packed in {dur:.1f} s")
        return ev

    # -- pads ----------------------------------------------------------------

    def _take_ref(self, s: _Session, gray: np.ndarray, tr: float) -> None:
        s.ref = snapshot(self._crop(gray, s.interior))
        s.ref_outer = self._crop(gray, s.outer).astype(np.float32)
        s.ref_t = tr

    def _fp_cover_in(self, fg_outer: np.ndarray, at: Rect, s: _Session,
                     dx: int = 0, dy: int = 0) -> float:
        """Share of the footprint, moved by (dx, dy), that is foreground in
        fg_outer (a crop of the frame taken at rect `at`)."""
        x, y, w, h = s.bbox
        ox, oy = x + dx - at[0], y + dy - at[1]
        H, W = fg_outer.shape
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(W, ox + w), min(H, oy + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        part = s.fp[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return float(np.count_nonzero(fg_outer[y0:y1, x0:x1] & part)) / s.fp_area

    def _box_shift(self, s: _Session, out_img: np.ndarray,
                   fg_outer: np.ndarray) -> Optional[Tuple[int, int]]:
        """Did the whole box slide since the before-picture? (The packer
        shoving it a little.) Returns the shift in pixels, or None.

        Two independent tests must agree: the picture of the box must match
        best at an offset, AND the box's silhouette must actually fit better
        there. A reach that merely changed a lot of the picture can fool the
        first, never the second."""
        if s.ref_outer is None or out_img.shape != s.ref_outer.shape or min(out_img.shape) < 16:
            return None
        cur = out_img.astype(np.float32)
        win = cv2.createHanningWindow((cur.shape[1], cur.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(s.ref_outer, cur, win)
        mag = (dx * dx + dy * dy) ** 0.5
        if resp < _SHIFT_MIN_RESPONSE or mag < 2.0 or mag > 0.5 * min(s.bbox[2:]):
            return None
        idx, idy = int(round(dx)), int(round(dy))
        here = self._fp_cover_in(fg_outer, s.outer, s)
        there = self._fp_cover_in(fg_outer, s.outer, s, idx, idy)
        if there < here + 0.02:
            return None
        return idx, idy

    def _translate(self, s: _Session, dx: int, dy: int) -> bool:
        """Move the session's geometry with the box. False if it would leave
        the picture."""
        x, y, w, h = s.bbox
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx + w > self.ww or ny + h > self.wh:
            return False
        ref, ref_outer = s.ref, s.ref_outer
        self._set_geometry(s, (nx, ny, w, h), s.fp)
        s.ref, s.ref_outer = ref, None        # contents moved with the box
        return True

    def _change_frac(self, s: _Session, cur: Tuple[np.ndarray, np.ndarray],
                     still_ref: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
        if s.ref is None or s.ref[0].shape != cur[0].shape:
            return 0.0
        ch = changed_mask(s.ref, cur, self.cfg.change_threshold) & s.fp_int
        if still_ref is not None:
            img, valid = still_ref
            filled = np.where(valid, img, cur[0])
            same_as_before = ~changed_mask(snapshot(filled), cur, self.cfg.change_threshold)
            ch &= ~(same_as_before & valid)
        return float(np.count_nonzero(ch)) / s.fp_int_area

    def _reach_fsm(self, s: _Session, tr: float, gray: np.ndarray,
                   fg: np.ndarray) -> None:
        cfg = self.cfg
        if not s.in_reach:
            if s.act >= cfg.reach_start_motion:
                s.on_frames += 1
                if s.on_frames >= 2:
                    s.in_reach = True
                    s.reach_start_t = tr
                    s.quiet_since = None
                    s.hist.clear()
            else:
                s.on_frames = 0
                # While the box is quiet, keep the "before" picture fresh so
                # paper slowly settling between reaches is not mistaken for a
                # new pad at the next reach.
                if s.act < cfg.reach_end_motion and tr - s.ref_t >= 0.5:
                    self._take_ref(s, gray, tr)
            return

        s.hist.append((tr, s.cov, s.act, self._crop(gray, s.interior).copy(),
                       self._crop(gray, s.outer).copy(), self._crop(fg, s.outer).copy()))
        if float(np.median(s.recent_act)) < cfg.reach_end_motion:
            if s.quiet_since is None:
                s.quiet_since = tr
            if tr - s.quiet_since >= cfg.quiet_seconds:
                self._end_reach(s, tr, gray, fg)
                return
        else:
            s.quiet_since = None
        if tr - s.reach_start_t > cfg.max_reach_seconds:
            self._end_reach(s, tr, gray, fg)

    def _judge_reach(self, s: _Session, frac: float, moved: bool, tr: float,
                     how: str) -> None:
        cfg = self.cfg
        s.reaches += 1
        if moved:
            verdict = "box moved, not a pad"
        elif frac >= cfg.pad_min_change:
            s.pads += cfg.pieces_per_visit
            verdict = f"+{cfg.pieces_per_visit} pad" + ("s" if cfg.pieces_per_visit > 1 else "")
        else:
            verdict = "no pad"
        self.last_reach = {"box": s.sid, "change": round(frac, 3),
                           "verdict": verdict, "t": tr}
        self.note = f"reach: {frac:.0%} of the box changed -> {verdict}"
        log.info("Box %d reach %d%s: %.0f%% of the box changed "
                 "(pad needs %.0f%%) -> %s; %d pads so far",
                 s.sid, s.reaches, how, 100 * frac, 100 * cfg.pad_min_change,
                 verdict, s.pads)

    def _end_reach(self, s: _Session, tr: float, gray: np.ndarray,
                   fg: np.ndarray) -> None:
        s.in_reach = False
        s.on_frames = 0
        # Judge on the quietest frame of the quiet period (latest on ties).
        quiet = [h for h in s.hist
                 if s.quiet_since is not None and h[0] >= s.quiet_since]
        s.quiet_since = None
        if quiet:
            q = min(reversed(quiet), key=lambda h: h[2])
            _, cov, _, int_img, out_img, fg_out = q
        else:
            cov = s.cov
            int_img = self._crop(gray, s.interior)
            out_img = self._crop(gray, s.outer)
            fg_out = self._crop(fg, s.outer)
        old_outer, old_ref_outer = s.outer, s.ref_outer
        how = ""
        still_ref = None
        far = False
        shift = self._box_shift(s, out_img, fg_out)
        if shift is not None and self._translate(s, *shift):
            # The box was shoved: compare its contents where they are now.
            how = f" (box shoved {shift[0]:+d},{shift[1]:+d} px)"
            cov = self._fp_cover_in(fg_out, old_outer, s)
            got = self._crop_from(out_img.astype(np.float32), old_outer, s.interior)
            still_ref = self._crop_from(old_ref_outer, old_outer, s.interior)
            if got is not None and got[1].all():
                int_img = got[0]
            else:
                # Shoved further than the margin kept around the box: its
                # contents cannot be lined up reliably. Follow it, but do
                # not count this reach as a pad.
                far = True
        frac = self._change_frac(s, snapshot(int_img), still_ref) if not far else 0.0
        moved = far or cov < _MOVED_KEEP
        self._judge_reach(s, frac, moved, tr, how)
        # The next reach is compared with this same quiet moment — not with
        # the current frame, which may already show the next hand coming in.
        if quiet and not how:
            s.ref = snapshot(int_img)
            s.ref_outer = out_img.astype(np.float32)
            s.ref_t = tr
        elif quiet and not far:
            s.ref = snapshot(int_img)            # contents, lined up
            s.ref_outer = self._crop(gray, s.outer).astype(np.float32)
            s.ref_t = tr
        else:
            self._take_ref(s, gray, tr)
        s.hist.clear()
        if moved and self._stable is not None:
            # Follow the box to where it stands now, straight away: the next
            # reach is usually only a moment off and must be judged there.
            s.cov = cov
            self._refresh_geometry(s, tr, gray, self._stable, force=True)

    def _crop_from(self, img: np.ndarray, at: Rect, r: Rect
                   ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Crop rect r (frame coords) out of img, which was cut from the frame
        at rect `at`. Returns (crop, valid) — valid False where r reaches
        outside img — or None if they do not overlap at all."""
        if img is None:
            return None
        ax, ay = at[0], at[1]
        x, y, w, h = r
        out = np.zeros((h, w), np.float32)
        valid = np.zeros((h, w), bool)
        sx0, sy0 = max(0, x - ax), max(0, y - ay)
        sx1, sy1 = min(img.shape[1], x - ax + w), min(img.shape[0], y - ay + h)
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        dx0, dy0 = sx0 - (x - ax), sy0 - (y - ay)
        out[dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] = img[sy0:sy1, sx0:sx1]
        valid[dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] = True
        return out, valid

    def _settle_hurried_reach(self, s: _Session) -> None:
        """The box started leaving while a reach was still open (the packer
        put the last pad in and pushed the box straight away). Judge the
        reach from the last quiet-ish frame before the box began to move —
        but only if a hand was at work well before that: a "reach" that is
        really just the box starting to slide has nothing to judge."""
        hist = list(s.hist)
        moved = next((i for i, h in enumerate(hist) if h[1] < _STILL_COVER), None)
        if moved is None or s.ref is None:
            return
        t_moved = hist[moved][0]
        if t_moved - s.reach_start_t < _HURRIED_MIN_S:
            return
        before = [h for h in hist[:moved] if h[0] <= t_moved - 0.1][-15:]
        if not before:
            return
        best = min(before, key=lambda h: h[2])
        frac = self._change_frac(s, snapshot(best[3].astype(np.float32)))
        self._judge_reach(s, frac, False, best[0], " (as the box left)")
        s.hist.clear()

    def _refresh_geometry(self, s: _Session, tr: float, gray: np.ndarray,
                          stable: np.ndarray, force: bool = False) -> None:
        """Follow a box that was nudged or whose flaps were folded: once it
        no longer covers its recorded outline and has gone still, re-measure.
        A box that still covers its outline keeps it, so paper hanging from
        the machine next to it is never absorbed into the box."""
        if not force and (s.cov >= _GEOMETRY_KEEP or tr - s.geo_t < _GEOMETRY_EVERY_S
                          or s.act >= self.cfg.reach_end_motion):
            return
        s.geo_t = tr
        x, y, w, h = s.bbox
        m = max(4, int(0.5 * max(w, h)))
        wx, wy, ww, wh = expand_rect(s.bbox, m, self.ww, self.wh)
        win = cv2.morphologyEx(_as_u8(stable[wy:wy + wh, wx:wx + ww]),
                               cv2.MORPH_CLOSE, _K7)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
        if n <= 1:
            return
        fpw = np.zeros((wh, ww), bool)
        fpw[y - wy:y - wy + h, x - wx:x - wx + w] = s.fp
        overlap = np.bincount(labels[fpw], minlength=n)
        overlap[0] = 0
        best = int(np.argmax(overlap))
        if best == 0 or overlap[best] < 0.15 * s.fp_area:
            return
        area = int(stats[best, cv2.CC_STAT_AREA])
        if not (self.min_area <= area <= self.max_area):
            return
        bx, by, bw, bh = (int(v) for v in stats[best, :4])
        nb = (bx + wx, by + wy, bw, bh)
        self._set_geometry(s, nb, labels[by:by + bh, bx:bx + bw] == best)
        self._take_ref(s, gray, tr)
        log.info("Box %d outline re-measured (moved or flaps changed)", s.sid)

    # -- candidates --------------------------------------------------------

    def _update_candidates(self, tr: float, small: np.ndarray, gray: np.ndarray,
                           fg: np.ndarray, stable: np.ndarray,
                           motion: np.ndarray) -> None:
        cfg = self.cfg
        protect = np.zeros((self.wh, self.ww), bool)
        for s in self.sessions:
            ox, oy, ow, oh = s.outer
            protect[oy:oy + oh, ox:ox + ow] = True
        m = cv2.morphologyEx(_as_u8(stable & ~protect), cv2.MORPH_CLOSE, _K7)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)

        found = []
        zx0, zy0, zx1, zy1 = self.zone
        # Half size is enough to KEEP following a box a hand is partly
        # covering; a new candidate (and any confirmation) needs full size.
        keep_area = 0.5 * self.min_area
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < keep_area or area > self.max_area:
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            if not (zx0 <= cx <= zx1 and zy0 <= cy <= zy1):
                continue
            found.append((i, tuple(int(v) for v in stats[i, :4]), area, cx, cy))

        for c in self.candidates:
            c.label = -1
        for i, bbox, area, cx, cy in found:
            best, best_score = None, 0.0
            for c in self.candidates:
                if c.label != -1:
                    continue
                # A hand over the box hides part of it, so the visible blob
                # shrinks or shifts: accept a weak overlap, or a centre that
                # stays inside the old outline.
                iou = rect_iou(bbox, c.bbox)
                bx, by, bw, bh = c.bbox
                inside = bx <= cx <= bx + bw and by <= cy <= by + bh
                ratio = area / float(max(1, c.area))
                score = iou if iou > 0.3 else (0.31 if inside and 0.3 <= ratio <= 3.0 else 0.0)
                if score > best_score:
                    best, best_score = c, score
            if best is None:
                if area < self.min_area:
                    continue
                self.candidates.append(_Candidate(
                    bbox=bbox, area=area, first_t=tr, last_t=tr, cx0=cx, cy0=cy,
                    label=i))
                continue
            best.bbox, best.area, best.last_t, best.label = bbox, area, tr, i
        for c in self.candidates:
            if c.label == -1:
                # Not seen as a still blob this frame. If its spot is still
                # mostly covered by something (the box under a moving hand),
                # it is hidden, not gone.
                x, y, w, h = c.bbox
                if np.count_nonzero(fg[y:y + h, x:x + w]) >= 0.6 * w * h:
                    c.last_t = tr
        self.candidates = [c for c in self.candidates
                           if c.label != -1 or tr - c.last_t <= _CAND_GAP_S]

        for c in list(self.candidates):
            if c.label == -1:
                continue
            x, y, w, h = c.bbox
            quiet = (np.count_nonzero(motion[y:y + h, x:x + w])
                     <= cfg.reach_end_motion * w * h)
            if not quiet:
                c.busy = c.busy or c.early is not None
            elif c.early is None and not c.busy and c.area >= self.min_area:
                # The box as it looked before anyone reached into it: if a pad
                # goes in before the box is confirmed, it is still counted.
                c.early = gray.copy()
            # Confirm only at a quiet moment: a hand over the box is then not
            # mistaken for part of it, and a box still sliding along the
            # rollers (always moving at its edges) is never confirmed.
            if (c.age(tr) < cfg.confirm_seconds or tr < c.rejected_until
                    or not quiet or c.area < self.min_area):
                continue
            comp = labels == c.label
            ok, reason = self._accept(comp, c.area, small)
            if not ok:
                if reason:
                    if reason != c.reason:
                        log.info("Ignoring an object at the spot: %s", reason)
                    c.rejected_until = tr + 0.5
                    c.reason = reason
                    self.note = f"ignored an object: {reason}"
                else:                       # ghost repaired: forget it
                    self.candidates.remove(c)
                continue
            self.candidates.remove(c)
            fp = self._clean_footprint(comp, small)
            self._absorb_fragments(c, fp)
            self._start_session(tr, fp, c, gray)

    def _absorb_fragments(self, c: _Candidate, comp: np.ndarray) -> None:
        """A hand across a small box splits it into pieces that were followed
        as separate candidates. Whatever lies on the confirmed box was the box
        all along: take over the earliest clock and "before" picture."""
        for o in list(self.candidates):
            x, y, w, h = o.bbox
            cx, cy = min(x + w // 2, self.ww - 1), min(y + h // 2, self.wh - 1)
            if not comp[cy, cx]:
                continue
            if o.first_t < c.first_t:
                c.first_t = o.first_t
                if o.early is not None:
                    c.early = o.early
            elif c.early is None:
                c.early = o.early
            c.busy = c.busy or o.busy
            self.candidates.remove(o)

    def _clean_footprint(self, comp: np.ndarray, small: np.ndarray) -> np.ndarray:
        """In brown mode keep only the cardboard (and paper) part of the blob:
        a dark sleeve resting against the box looks "still" in its uniform
        middle and would otherwise become part of the box's outline."""
        if self.cfg.box_color != "brown":
            return comp
        brown = cv2.dilate(_as_u8(brown_mask(small)), _K5).astype(bool)
        keep = _as_u8(comp & brown)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
        if n <= 1:
            return comp
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        piece = labels == best
        if np.count_nonzero(piece) < 0.5 * np.count_nonzero(comp):
            return comp
        return cv2.morphologyEx(_as_u8(piece), cv2.MORPH_CLOSE, _K7).astype(bool) & comp

    def _accept(self, comp: np.ndarray, area: int, small: np.ndarray) -> Tuple[bool, str]:
        cfg = self.cfg
        # 1. An arm meets the picture edge with a narrow wrist or forearm.
        x0, y0, rw, rh = self.roi
        x1, y1 = x0 + rw - 1, y0 + rh - 1
        contact = (np.count_nonzero(comp[y0, x0:x1 + 1])
                   + np.count_nonzero(comp[y1, x0:x1 + 1])
                   + np.count_nonzero(comp[y0:y1 + 1, x0])
                   + np.count_nonzero(comp[y0:y1 + 1, x1]))
        if contact > 0 and contact / max(1.0, area ** 0.5) < cfg.arm_contact_ratio:
            return False, "it reaches in from the picture edge like an arm"
        # 2. The bare spot behind a box that stood here when we started.
        if self._is_ghost(comp, small):
            log.info("Repaired the empty-spot picture where a box stood at start-up")
            self._heal(comp, small)
            return False, ""
        # 3. Cardboard colour.
        if cfg.box_color == "brown":
            px = max(1, int(np.count_nonzero(comp)))
            frac = float(np.count_nonzero(brown_mask(small)[comp])) / px
            if frac < cfg.min_brown_frac:
                return False, (f"not cardboard-coloured ({frac:.0%} brown, "
                               f"needs {cfg.min_brown_frac:.0%})")
        return True, ""

    def _is_ghost(self, comp: np.ndarray, small: np.ndarray) -> bool:
        """True if this "object" is really the empty spot: it now looks like
        its surroundings, while the background picture there does not."""
        area = int(np.count_nonzero(comp))
        r = max(3, int(0.15 * area ** 0.5))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = cv2.dilate(_as_u8(comp), k).astype(bool) & ~comp & self.roi_mask
        if np.count_nonzero(ring) < 30:
            return False

        def hist(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
            h = cv2.calcHist([img], [0, 1, 2], _as_u8(mask), [8, 8, 8],
                             [0, 256, 0, 256, 0, 256])
            cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
            return h

        ring_now = hist(small, ring)
        d_now = cv2.compareHist(hist(small, comp), ring_now, cv2.HISTCMP_BHATTACHARYYA)
        d_bg = cv2.compareHist(hist(self.bg8, comp), ring_now, cv2.HISTCMP_BHATTACHARYYA)
        return d_now < 0.35 and d_now + 0.15 < d_bg

    def _heal(self, comp: np.ndarray, small: np.ndarray) -> None:
        region = cv2.dilate(_as_u8(comp), _K7).astype(bool)
        self.bg_f[region] = small[region]
        self.bg8 = np.clip(self.bg_f + 0.5, 0, 255).astype(np.uint8)

    def _start_session(self, tr: float, comp: np.ndarray, c: _Candidate,
                       gray: np.ndarray) -> None:
        ys, xs = np.nonzero(comp)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
        # Pack time starts when the box was set down, not when we finished
        # making sure it is a box: the moment its pixels stopped matching the
        # empty spot.
        since = self.fg_since[comp]
        start = float(np.median(since)) if since.size else c.first_t
        start = min(c.first_t, max(start, c.first_t - _BACKDATE_MAX_S))
        s = _Session(sid=self._next_sid, start_t=start, last_present_t=tr)
        self._next_sid += 1
        self._set_geometry(s, (x, y, w, h), comp[y:y + h, x:x + w])
        self._take_ref(s, gray, tr)
        s.geo_t = tr
        self.sessions.append(s)
        log.info("Box at the station (%.0f%% of the view)",
                 100.0 * s.fp_area / self.roi_area)
        self.note = "box arrived"
        if self.cfg.count_pads and c.busy and c.early is not None:
            # Someone reached into the box while we were still making sure it
            # is a box: judge that reach against how the box looked before.
            s.ref = snapshot(self._crop(c.early, s.interior))
            frac = self._change_frac(s, snapshot(self._crop(gray, s.interior)))
            self._judge_reach(s, frac, False, tr, " (before the box was confirmed)")
            self._take_ref(s, gray, tr)

    # -- background upkeep ---------------------------------------------------

    def _maintain_background(self, tr: float, small: np.ndarray,
                             fg_raw: np.ndarray, still: np.ndarray) -> None:
        cfg = self.cfg
        protect = np.zeros((self.wh, self.ww), bool)
        for s in self.sessions:
            ox, oy, ow, oh = expand_rect(s.outer, 4, self.ww, self.wh)
            protect[oy:oy + oh, ox:ox + ow] = True
        keep_present = protect.copy()
        for c in self.candidates:
            if tr >= c.rejected_until:          # still being checked
                x, y, w, h = c.bbox
                protect[y:y + h, x:x + w] = True

        changed = False
        if self._frame_no % 3 == 0:
            # Follow slow lighting drift on the bare spot.
            upd = ~fg_raw & still & ~protect & self.roi_mask
            cv2.accumulateWeighted(small, self.bg_f, 0.05, mask=_as_u8(upd))
            changed = True
        if self._frame_no % 5 == 0:
            # Anything lying still for a long time without being a box (a
            # dropped pad, a tape gun) is furniture now.
            absorb = fg_raw & ((tr - self.last_motion) >= cfg.absorb_seconds) & ~protect
            if absorb.any():
                self.bg_f[absorb] = small[absorb]
                changed = True
        # Most of the view changed for a long time: lights or camera changed.
        # (Measured outside the box being packed and anything being checked:
        # a big box plus the packer is not a lighting change.)
        free = self.roi_mask & ~protect
        if np.count_nonzero(fg_raw & free) > 0.6 * max(1, np.count_nonzero(free)):
            if self._global_since is None:
                self._global_since = tr
            elif tr - self._global_since >= cfg.relearn_seconds:
                log.warning("Most of the picture changed for %.0f s (lights? camera "
                            "moved?) — re-learning the empty spot", cfg.relearn_seconds)
                self.note = "re-learned the empty spot (lighting or camera changed)"
                self.bg_f[~keep_present] = small[~keep_present]
                self._global_since = None
                changed = True
        else:
            self._global_since = None
        if changed:
            self.bg8 = np.clip(self.bg_f + 0.5, 0, 255).astype(np.uint8)

    # -- reporting -----------------------------------------------------------

    def present_session(self) -> Optional[_Session]:
        return self.sessions[0] if self.sessions else None

    def state_text(self) -> str:
        tr = self._tr
        if self.bg_f is None:
            return "learning the empty packing spot — keep it clear"
        s = self.present_session()
        if s is not None:
            text = f"box at the station: {_pads(s.pads)}, {tr - s.start_t:.0f} s"
            return text + ("  (hand in box)" if s.in_reach else "")
        live = [c for c in self.candidates if c.label != -1]
        checking = [c for c in live if tr >= c.rejected_until]
        if checking:
            c = max(checking, key=lambda c: c.area)
            return f"something stopped at the spot — checking ({c.age(tr):.1f} s)"
        if live:
            c = max(live, key=lambda c: c.area)
            return f"ignoring an object at the spot: {c.reason}"
        return "waiting for a box"

    def live_stats(self) -> dict:
        s = self.present_session()
        out = {"packing_now": s is not None,
               "station_state": self.state_text(),
               "station_note": self.note}
        if s is not None:
            out.update({"current_pieces": s.pads,
                        "current_elapsed_s": round(max(0.0, self._tr - s.start_t), 1),
                        "hand_in": s.in_reach})
        return out

    def overlay(self) -> dict:
        """Geometry for drawing, in full-frame pixels."""
        zx0, zy0, zx1, zy1 = self.zone
        sessions = []
        for s in self.sessions:
            sessions.append({
                "bbox": self.to_full(s.bbox),
                "interior": self.to_full(s.interior),
                "pads": s.pads,
                "elapsed_s": max(0.0, self._tr - s.start_t),
                "hand_in": s.in_reach,
            })
        cands = []
        for c in self.candidates:
            if c.label == -1:
                continue
            cands.append({"bbox": self.to_full(c.bbox), "age_s": c.age(self._tr),
                          "rejected": self._tr < c.rejected_until})
        return {
            "roi": self.to_full(self.roi),
            "zone": self.to_full((int(zx0), int(zy0), int(zx1 - zx0), int(zy1 - zy0))),
            "sessions": sessions,
            "candidates": cands,
            "learning": self.bg_f is None,
        }

    def debug_view(self) -> Optional[np.ndarray]:
        """What the detector sees: dim background, orange = differs from the
        empty spot, white = differs AND still (box material), red = moving."""
        if self._dbg is None or self.bg8 is None:
            return None
        fg, stable, motion = self._dbg
        img = (self.bg8 // 3).copy()
        img[fg] = (0, 120, 230)
        img[stable] = (235, 235, 235)
        img[motion] = (40, 40, 230)
        for s in self.sessions:
            x, y, w, h = s.bbox
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (80, 220, 80), 1)
        if (self.ww, self.wh) != (self.fw, self.fh):
            img = cv2.resize(img, (self.fw, self.fh), interpolation=cv2.INTER_NEAREST)
        return img
