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

3. A box is looked for only inside the packing zone, among foreground that
   is still and (box_color: brown) cardboard-coloured. All such pieces at
   the spot together are ONE box — an arm across it or the paper hanging
   into it often cuts it in pieces — if they are big enough and stay for
   confirm_seconds. What was at the spot before the box arrived (the packer
   standing against it) is left out, thin strips are dropped, and what
   reaches in from the picture edge like an arm is cut off, not held
   against the box. Hands and arms alone are never boxes: they do not stay
   still and they are smaller than a box.

4. Pads. Where the machine's outlet has been measured (setup step 3), pads
   are counted there: the paper just below the outlet is watched, and the
   pad hanging there gone — or a DIFFERENT crumpled pad hanging there — is
   one pad taken, credited to the box at the station. The same pad hanging
   a little differently (dragged along by a box, brushed by a hand) is
   recognised by its creases and not counted. Without the outlet, every
   REACH — movement over the box, then the box going quiet again — is
   checked: if enough of the inside of the box looks different than just
   before the reach, a pad went in.

5. When most of the box's spot is empty again for depart_seconds — or more
   than half of it for a full second — the box has left and is counted,
   with its pads and its pack time (box set down -> box gone). A box nudged
   during packing is followed to its new position as long as it still
   overlaps where it was.
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
_EARLY_FILL_S = 1.0        # a box's "before" picture is completed this long
_BACKDATE_MAX_S = 10.0     # pack time may start at most this long before
                           # the box was confirmed
_MOVED_KEEP = 0.75         # after a reach, less of the box's spot covered than
                           # this = the box itself was moved, not a pad added
_LEAVE_DROP = 0.05         # coverage this far below its best = leaving
_HURRIED_MIN_S = 0.5       # hand busy this long before the box moved = a reach
_OUTLET_PRE_S = 6.0        # a pad pulled off this long before the box was set
                           # down still belongs to it (pulled first, then the
                           # box slid under it) — unless it was already
                           # counted for the box before
_OUTLET_MERGE_S = 0.3      # outlet changes this close together are one pad
                           # (torn off, then the machine pushes out the next)
_OUTLET_WAIT_S = 3.0       # a box that leaves while a pad is still coming out
                           # waits at most this long for it to be counted
_SHIFT_MIN_RESPONSE = 0.25 # phase-correlation peak needed to call it a shove
                           # (the silhouette must also fit better there)
_INSIDE_HOLD = 3.0         # movement over the inside of the box (as a multiple
                           # of reach_end_motion) that keeps an open reach open
_ARRIVE_S = 1.5            # a box slides in and stops within this; what was
                           # at the spot longer before, or turns up later, is
                           # not part of it
_HALF_GONE = 0.5           # less of the box's spot covered than this for
_HALF_GONE_S = 1.0         # this long = gone too (something else stayed
                           # behind where it stood: the packer, the next pad)
_WATCH_SAME = 0.5          # outlet window: match above this = the same pad
_PAPER_GONE_S = 0.2        # no paper at the outlet this long = the pad was taken
_PAPER_BACK_S = 0.3        # paper at the outlet this long = a pad hangs there


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


def _creases(img: np.ndarray) -> np.ndarray:
    """The fine structure of the paper (creases, folds) without its broad
    light and shade: what makes one crumpled pad different from the next."""
    g = img.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 4.0)


def same_pad(a: np.ndarray, b: np.ndarray) -> float:
    """How well picture b of the paper at the outlet matches picture a,
    allowing for the paper having slipped a few pixels: 1 = the same crumpled
    pad, around 0 = a different one."""
    h, w = a.shape[:2]
    my, mx = max(2, min(6, h // 5)), max(2, min(6, w // 5))
    if (h - 2 * my) * (w - 2 * mx) < 16 or b.shape[:2] != (h, w):
        return 1.0
    if float(a.std()) < 4.0 or float(b.std()) < 4.0:
        # nothing to match on (a bare, even surface): compare brightness
        return 1.0 if float(np.mean(cv2.absdiff(a, b))) < 8.0 else 0.0
    ca, cb = _creases(a), _creases(b)
    return float(cv2.matchTemplate(cb, ca[my:h - my, mx:w - mx], cv2.TM_CCOEFF_NORMED).max())


def outlet_windows(zone: List[float], ww: int, wh: int) -> Tuple[Rect, Rect, Rect]:
    """From the outlet zone (fractions of the picture, as the setup measures
    it: the hanging pad plus a margin) to pixel rectangles at size ww x wh:

    - the zone itself (left out of everything about the BOX),
    - the watch window: the middle of the paper a little below the outlet.
      Paper always covers it, whatever stands underneath, and it swings less
      than the free end. It does not reach up to the outlet itself: the
      machine's edge is the same in every picture and would make any two
      pads "match";
    - the mouth strip: right below the outlet, above anything that can lie
      in a box — whether paper hangs there at all (a pad already stuffed
      into the box shows through the window when the outlet is bare and
      must not be taken for the next pad)."""
    ox, oy, ow, oh = zone
    x0, y0 = int(ox * ww), int(oy * wh)
    x1 = min(ww, int(round((ox + ow) * ww)))
    y1 = min(wh, int(round((oy + oh) * wh)))
    w, h = max(4, x1 - x0), max(4, y1 - y0)
    ww_, wh_ = max(8, int(round(0.55 * w))), max(8, int(round(0.40 * h)))
    wx = max(0, min(ww - ww_, x0 + (w - ww_) // 2))
    wy = max(0, min(wh - wh_, y0 + int(round(0.20 * h))))
    mh = max(4, int(round(0.12 * h)))
    my = max(0, min(wh - mh, y0 + int(round(0.10 * h))))
    return (x0, y0, w, h), (wx, wy, ww_, wh_), (wx, my, ww_, mh)


def paper_look(col: np.ndarray, gray: np.ndarray,
               stats: Optional[Tuple[float, float]] = None) -> bool:
    """Does this strip below the outlet show crumpled paper? col: the strip
    in colour, gray: the same strip in grey at full resolution. Paper is
    kraft-coloured and full of creases; the machine, the rollers and a
    cardboard flap behind a bare outlet are grey, or as brown but smooth.
    stats: brightness and creasing of the pads seen so far — the next ones
    are about as bright, and not much smoother."""
    if float(np.count_nonzero(brown_mask(col))) < 0.5 * col.shape[0] * col.shape[1]:
        return False
    rough = float(_creases(gray).std())
    if stats is None:
        return rough >= 2.5
    # (pads differ a lot in how creased they are right at the outlet: only
    # clearly smoother than paper is "not paper")
    mean, ref_rough = stats
    return (rough >= max(2.5, 0.25 * ref_rough)
            and 0.6 * mean <= float(gray.mean()) <= 1.5 * mean)


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
class _PadOut:
    """One pad taken from the machine (outlet counting)."""
    start: float                    # the hand reached the outlet
    end: float                      # the next pad hung still again
    credited: Optional[int] = None  # session id it was counted for


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
    # How the box looked before anyone reached into it, built pixel by pixel
    # from the first moment each part was seen still: parts a hand covered
    # at first fill in as soon as the hand moves on.
    early: Optional[np.ndarray] = None       # work-frame grey
    early_known: Optional[np.ndarray] = None  # which pixels are filled in
    busy: bool = False              # a hand moved over it while unconfirmed
    seen: Optional[np.ndarray] = None    # everywhere it has been seen still
                                         # (a hand hides different parts of
                                         # the box at different moments)

    def age(self, t: float) -> float:
        # (+1 µs: frame times are sums of 1/fps, and 9 frames at 15 fps must
        # count as the full 0.6 s, not 0.5999999)
        return t - self.first_t + 1e-6


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
    interior_stale: bool = False            # outline grew; re-measure when quiet
    core: Optional[np.ndarray] = None       # bbox-local: the box minus its rim
    core_area: int = 1
    ref_exclude: Optional[np.ndarray] = None  # interior-local: ignore in the
                                              # next comparison (a hand that
                                              # was in the before-picture)
    low_since: Optional[float] = None
    half_since: Optional[float] = None
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
    inside: float = 0.0      # movement over the whole inside rectangle
    busy: float = 0.0        # act, or the inside still clearly moving
    # (t, coverage, busy, interior crop, outer crop, outer foreground) for
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
        self.zone_mask = np.zeros((self.wh, self.ww), bool)
        self.zone_mask[max(0, int(zy * self.wh)):int(round((zy + zh) * self.wh)),
                       max(0, int(zx * self.ww)):int(round((zx + zw) * self.ww))] = True
        self.zone_mask &= self.roi_mask
        self.min_area = cfg.min_box_frac * self.roi_area
        self.max_area = cfg.max_box_frac * self.roi_area
        # Where pads come out of the machine. Excluded from everything about
        # the BOX (the paper hanging there is not the box), and watched on
        # its own when pads are counted at the outlet.
        self.outlet: Optional[Rect] = None
        self.outlet_mask: Optional[np.ndarray] = None
        if cfg.outlet_zone:
            self.outlet, self.watch, self.mouth = outlet_windows(
                cfg.outlet_zone, self.ww, self.wh)
            x0, y0, w_, h_ = self.outlet
            self.outlet_mask = np.zeros((self.wh, self.ww), bool)
            self.outlet_mask[y0:y0 + h_, x0:x0 + w_] = True
        # Where the outlet is known, pads are counted there: every pad the
        # machine hands over is one pad, however it is stuffed into the box
        # and however much of the box the hanging paper hides. Without it
        # (or with pad_method: reach) pads are counted per reach into the box.
        self.pad_mode = "outlet" if (cfg.pad_method in ("auto", "outlet")
                                     and self.outlet is not None) else "reach"
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
        self._boxlike: Optional[np.ndarray] = None
        self._arm_seen = False
        self._why = ""                       # why what is at the spot is no box
        self._why_since = 0.0
        self._why_logged = ""
        # outlet counting
        self.pad_events: List[_PadOut] = []
        self._o_ref: Optional[np.ndarray] = None    # the paper as it last hung still
        self._o_ref_t = 0.0
        self._o_stats: Optional[Tuple[float, float]] = None   # brightness, creasing of paper
        self._o_prev = None                         # (picture, t, event) of the pad counted last
        self._o_present = False                     # a pad hangs from the outlet
        self._o_seen_any = False
        self._o_paper_since: Optional[float] = None
        self._o_bare_since: Optional[float] = None
        self._o_burst = False
        self._o_burst_start = 0.0
        self._o_on = 0
        self._o_quiet_since: Optional[float] = None
        self._o_recent: Deque[float] = deque(maxlen=3)
        self._finishing: List[Tuple[_Session, float]] = []
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
        if self.outlet_mask is not None:
            if self.pad_mode == "outlet" and cfg.count_pads:
                self._outlet_update(tr, frame, small, motion)
            # The paper hanging from the outlet is never part of a box. (Its
            # movement is not masked: the box's own reach measure already
            # leaves the outlet out, and a hand tearing a pad off there must
            # still keep a reach open and keep the box's first picture clean.)
            fg &= ~self.outlet_mask
        self.fg_since[~fg] = tr
        stable = fg & still
        self._stable = stable
        # What can be (part of) a box: in brown mode only cardboard-coloured
        # foreground. The packer's dark clothes right next to the box, the
        # grey floor showing where someone stood before — none of that is.
        if cfg.box_color == "brown":
            boxlike = fg & cv2.dilate(_as_u8(brown_mask(small)), _K5).astype(bool)
        else:
            boxlike = fg
        self._boxlike = boxlike

        events: List[CountEvent] = []
        for s in list(self.sessions):
            ev = self._update_session(s, tr, small, gray, fg, motion, stable)
            if ev is not None:
                events.append(ev)
        for s, deadline in list(self._finishing):
            # Waiting for the last pad of a box that has already left.
            if not self._o_burst or tr >= deadline:
                self._finishing.remove((s, deadline))
                ev = self._count_box(s)
                if ev is not None:
                    events.append(ev)
        self._update_candidates(tr, small, gray, fg, stable, motion)
        self._maintain_background(tr, small, fg_raw, still)
        if self.debug:
            self._dbg = (fg, stable, motion)
        return events

    # -- sessions ----------------------------------------------------------

    def _set_interior(self, s: _Session) -> None:
        # The inside of the box, where pads land: its outline shrunk past the
        # spread flaps and the rim. Deliberately the whole rectangle, not just
        # the box-coloured part: the pad hanging from the machine covers part
        # of the box from the camera's point of view, and a pad stuffed under
        # it is only visible as the hanging pad being torn off and replaced.
        x, y, w, h = s.bbox
        ins = int(round(self.cfg.interior_inset_frac * min(w, h)))
        iw, ih = max(4, w - 2 * ins), max(4, h - 2 * ins)
        s.interior = (x + (w - iw) // 2, y + (h - ih) // 2, iw, ih)
        s.fp_int = np.ones((ih, iw), bool)
        s.fp_int_area = iw * ih
        s.interior_stale = False

    def _set_geometry(self, s: _Session, bbox: Rect, fp: np.ndarray,
                      keep_interior: bool = False) -> None:
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
        if keep_interior and s.fp_int is not None:
            # Mid-reach: keep comparing the same rectangle; it is re-measured
            # at the next quiet moment.
            s.interior_stale = True
        else:
            self._set_interior(s)
        # The core of the box: whether the BOX moved is judged on this, so a
        # torn-off pad or a flap at the edge of the outline cannot make every
        # reach look like a shove.
        k = max(1, int(0.12 * min(w, h)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        core = cv2.erode(_as_u8(s.fp), ker).astype(bool)
        s.core = core if np.count_nonzero(core) >= 0.2 * s.fp_area else s.fp
        s.core_area = max(1, int(np.count_nonzero(s.core)))
        s.ref_exclude = None
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
        # The inside of the box as a whole, including where the paper hanging
        # from the machine covers it (left out of act). A hand tearing a pad
        # off there and carrying it into the box moves a lot of it; the
        # hanging pad swaying moves only its edges. This only keeps a reach
        # open — it never starts one.
        ix, iy, iw, ih = s.interior
        s.inside = float(np.count_nonzero(motion[iy:iy + ih, ix:ix + iw])) / max(1, iw * ih)
        s.busy = max(s.act, s.inside / _INSIDE_HOLD)
        s.recent_act.append(s.busy)

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
        if s.cov < _HALF_GONE:
            if s.half_since is None:
                s.half_since = tr
        else:
            s.half_since = None
        half_gone = s.half_since is not None and tr - s.half_since >= _HALF_GONE_S
        if s.cov < cfg.depart_cover or half_gone:
            if s.low_since is None:
                s.low_since = tr
            if tr - s.low_since >= cfg.depart_seconds or half_gone:
                # Gone. Counted right away: on a conveyor the next box can be
                # on the spot a second later and must start its own session.
                if s.in_reach and cfg.count_pads and self.pad_mode == "reach":
                    self._settle_hurried_reach(s)
                log.info("Box %d left the spot", s.sid)
                return self._finish(s)
        else:
            s.low_since = None

        if cfg.count_pads and self.pad_mode == "reach":
            self._reach_fsm(s, tr, gray, fg)
        else:
            # Pads are counted at the outlet; "hand in box" is just for show.
            s.in_reach = s.act >= cfg.reach_start_motion
            if self.pad_mode == "outlet":
                s.pads = self._outlet_pads_for(s, tr) * cfg.pieces_per_visit
        if not s.in_reach:
            self._refresh_geometry(s, tr, gray, stable)
        return None

    def _finish(self, s: _Session) -> Optional[CountEvent]:
        self.sessions.remove(s)
        if (self.pad_mode == "outlet" and self.cfg.count_pads and self._o_burst
                and s.start_t - _OUTLET_PRE_S <= self._o_burst_start <= s.last_present_t):
            # A pad was being taken as the box left (the packer pushed it on
            # straight after the last pad): wait for the outlet to settle so
            # that pad is counted for this box.
            self._finishing.append((s, self._tr + _OUTLET_WAIT_S))
            return None
        return self._count_box(s)

    def _count_box(self, s: _Session) -> Optional[CountEvent]:
        cfg = self.cfg
        if self.pad_mode == "outlet" and cfg.count_pads:
            n = 0
            for e in self.pad_events:
                if e.credited is None and self._outlet_window(s, e):
                    e.credited = s.sid
                    n += 1
            s.pads = n * cfg.pieces_per_visit
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

    # -- pads: at the outlet --------------------------------------------------

    @staticmethod
    def _outlet_window(s: _Session, e: _PadOut) -> bool:
        return s.start_t - _OUTLET_PRE_S <= e.start <= s.last_present_t + 0.2

    def _outlet_pads_for(self, s: _Session, tr: float) -> int:
        return sum(1 for e in self.pad_events
                   if e.credited is None and e.start >= s.start_t - _OUTLET_PRE_S)

    def _outlet_update(self, tr: float, frame: np.ndarray, small: np.ndarray,
                       motion: np.ndarray) -> None:
        """Count pads where they come out of the machine. Two signs that the
        pad hanging there was taken, whichever comes first:

        1. The strip right below the outlet no longer shows paper (the pad
           was pulled off and the machine has not pushed out the next yet).
        2. The paper hangs still again but is a DIFFERENT crumpled pad than
           before (the machine pushed the next one out straight away).

        The same pad hanging a little differently (dragged along by a box,
        brushed by a hand) is recognised and not counted, and if the pad
        that seemed gone turns out to hang there still (a hand was in front
        of it), the count is taken back."""
        cfg = self.cfg
        x, y, w, h = self.watch
        mx_, my_, mw_, mh_ = self.mouth
        # (movement anywhere from the outlet down to the end of the window:
        # a new pad sliding out reaches the outlet strip before the window,
        # and nothing may be judged while it is still on its way)
        mo = (float(np.count_nonzero(motion[my_:y + h, x:x + w]))
              / (w * max(1, y + h - my_)))
        self._o_recent.append(mo)
        quiet = float(np.median(self._o_recent)) < 0.015
        # The paper is compared at the camera's full resolution: two
        # different crumpled pads must not look alike by chance.
        fx, fy, fw_, fh_ = self.to_full(self.watch)
        patch = cv2.GaussianBlur(
            cv2.cvtColor(frame[fy:fy + fh_, fx:fx + fw_], cv2.COLOR_BGR2GRAY), (3, 3), 0)
        fmx, fmy, fmw, fmh = self.to_full(self.mouth)
        strip = cv2.cvtColor(frame[fmy:fmy + fmh, fmx:fmx + fmw], cv2.COLOR_BGR2GRAY)

        paper_now = paper_look(small[my_:my_ + mh_, mx_:mx_ + mw_], strip, self._o_stats)

        def learn_paper() -> None:
            m, r = float(strip.mean()), float(_creases(strip).std())
            if self._o_stats is None:
                self._o_stats = (m, r)
            else:
                self._o_stats = (0.7 * self._o_stats[0] + 0.3 * m,
                                 0.7 * self._o_stats[1] + 0.3 * r)

        def taken(start: float, why: str) -> _PadOut:
            ev = _PadOut(start=start, end=tr)
            self.pad_events.append(ev)
            if len(self.pad_events) > 500:
                self.pad_events = self.pad_events[-200:]
            self.note = "a pad was taken from the machine"
            log.info("Outlet: a pad was taken (%s)", why)
            return ev

        def same_as_before() -> bool:
            """The pad that was counted as taken hangs there after all."""
            prev = self._o_prev
            if (prev is None or prev[0] is None or tr - prev[1] > 6.0
                    or same_pad(prev[0], patch) < _WATCH_SAME):
                return False
            ev = prev[2]
            if ev is not None and ev.credited is None and ev in self.pad_events:
                self.pad_events.remove(ev)
                log.info("Outlet: the earlier pad still hangs there — not counted after all")
            self._o_prev = None
            return True

        # -- 1. paper at the outlet strip: there / gone -----------------------
        if paper_now:
            self._o_bare_since = None
            if self._o_paper_since is None:
                self._o_paper_since = tr
            if not self._o_present and tr - self._o_paper_since >= _PAPER_BACK_S:
                self._o_present = True
                self._o_ref = None               # not seen hanging still yet
                if self._o_seen_any:
                    log.info("Outlet: a new pad came out")
                self._o_seen_any = True
        else:
            self._o_paper_since = None
            if self._o_bare_since is None:
                self._o_bare_since = tr
            if self._o_present and tr - self._o_bare_since >= _PAPER_GONE_S:
                start = min(self._o_bare_since,
                            self._o_burst_start if self._o_burst else self._o_bare_since)
                ev = taken(start, "the outlet is empty")
                self._o_prev = (self._o_ref, tr, ev)
                self._o_present = False
                self._o_ref = None

        # -- 2. the paper hanging still: the same pad or another one? ---------
        if not self._o_burst:
            if mo >= 0.04:
                self._o_on += 1
                if self._o_on >= 2:
                    self._o_burst = True
                    self._o_burst_start = tr
                    self._o_quiet_since = None
            else:
                self._o_on = 0
                if (quiet and self._o_present and paper_now
                        and tr - self._o_ref_t >= 1.0):
                    if self._o_ref is None:
                        if not same_as_before():
                            learn_paper()
                        self._o_ref, self._o_ref_t = patch.copy(), tr
                    elif same_pad(self._o_ref, patch) >= _WATCH_SAME:
                        # follow slow drift (light, paper relaxing)
                        self._o_ref, self._o_ref_t = patch.copy(), tr
            return

        if not quiet:
            self._o_quiet_since = None
            if tr - self._o_burst_start > 60.0:
                # something resting at the outlet for a minute: start over
                self._o_burst = False
                self._o_ref, self._o_prev = None, None
            return
        if self._o_quiet_since is None:
            self._o_quiet_since = tr
        if tr - self._o_quiet_since < cfg.outlet_quiet_seconds:
            return

        # Still again after something moved there.
        self._o_burst = False
        self._o_on = 0
        if not (self._o_present and paper_now):
            return                               # bare outlet: part 1 covers it
        if self._o_ref is None:
            if not same_as_before():
                learn_paper()
            self._o_ref, self._o_ref_t = patch.copy(), tr
            return
        sim = same_pad(self._o_ref, patch)
        log.debug("Outlet: still again after %.1f s; match with the pad before %.2f",
                  tr - self._o_burst_start, sim)
        if sim >= _WATCH_SAME:
            self._o_ref, self._o_ref_t = patch.copy(), tr
            return
        if same_as_before():
            self._o_ref, self._o_ref_t = patch.copy(), tr
            return
        ev = taken(self._o_burst_start, "the next one hangs there")
        self._o_prev = (self._o_ref, tr, ev)
        self._o_ref, self._o_ref_t = patch.copy(), tr
        learn_paper()

    # -- pads: per reach --------------------------------------------------------

    def _take_ref(self, s: _Session, gray: np.ndarray, tr: float) -> None:
        s.ref_exclude = None
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

    def _core_cover_in(self, fg_outer: np.ndarray, at: Rect, s: _Session) -> float:
        x, y, w, h = s.bbox
        ox, oy = x - at[0], y - at[1]
        H, W = fg_outer.shape
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(W, ox + w), min(H, oy + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        part = s.core[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return float(np.count_nonzero(fg_outer[y0:y1, x0:x1] & part)) / s.core_area

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
        if s.ref_exclude is not None and s.ref_exclude.shape == ch.shape:
            ch &= ~s.ref_exclude
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
                if s.busy < cfg.reach_end_motion and tr - s.ref_t >= 0.5:
                    if s.interior_stale:
                        self._set_interior(s)
                    self._take_ref(s, gray, tr)
            return

        s.hist.append((tr, s.cov, s.busy, self._crop(gray, s.interior).copy(),
                       self._crop(gray, s.outer).copy(), self._crop(fg, s.outer).copy()))
        if s.low_since is not None:
            # The box no longer covers its spot: it is leaving. Nothing is
            # judged now — if it really goes, the open reach is settled from
            # the frames before it moved (see _settle_hurried_reach).
            s.quiet_since = None
            return
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
            out_img = fg_out = None
        if out_img is None or fg_out is None or out_img.shape[:2] != (s.outer[3], s.outer[2]):
            # (frame recorded before the outline grew: use the current one)
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
        core_cov = self._core_cover_in(fg_out, old_outer if how else s.outer, s)
        moved = far or core_cov < _MOVED_KEEP
        self._judge_reach(s, frac, moved, tr, how)
        # The next reach is compared with this same quiet moment — not with
        # the current frame, which may already show the next hand coming in.
        s.ref_exclude = None
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
        if not hist or s.ref is None:
            return
        # When did the box begin to leave? The last moment it still covered
        # its outline about as well as it did during the reach. (A hand over
        # the box makes the coverage wobble by a percent or two; leaving makes
        # it collapse.)
        base = max(h[1] for h in hist)
        in_place = [i for i, h in enumerate(hist) if h[1] >= base - _LEAVE_DROP]
        if not in_place:
            return
        last = in_place[-1]
        t_moved = hist[last][0]
        if t_moved - s.reach_start_t < _HURRIED_MIN_S:
            return
        before = [h for h in hist[:last + 1] if h[0] <= t_moved - 0.1][-15:]
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
        if s.low_since is not None:
            return          # leaving: whatever else stands there is not this box
        if not force and (s.cov >= _GEOMETRY_KEEP or tr - s.geo_t < _GEOMETRY_EVERY_S
                          or s.act >= self.cfg.reach_end_motion):
            return
        s.geo_t = tr
        x, y, w, h = s.bbox
        m = max(4, int(0.5 * max(w, h)))
        wx, wy, ww, wh = expand_rect(s.bbox, m, self.ww, self.wh)
        win = cv2.morphologyEx(
            _as_u8((stable & self._boxlike)[wy:wy + wh, wx:wx + ww]),
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

    def _edge_contact(self, mask: np.ndarray) -> int:
        """How many pixels of the mask lie on the picture edge."""
        x0, y0, rw, rh = self.roi
        x1, y1 = x0 + rw - 1, y0 + rh - 1
        return int(np.count_nonzero(mask[y0, x0:x1 + 1])
                   + np.count_nonzero(mask[y1, x0:x1 + 1])
                   + np.count_nonzero(mask[y0:y1 + 1, x0])
                   + np.count_nonzero(mask[y0:y1 + 1, x1]))

    def _strip_arms(self, piece: np.ndarray) -> Optional[np.ndarray]:
        """Cut off what reaches in from the picture edge like an arm.

        A blob that touches the edge of the picture may be an arm, a box with
        an arm across it, or a box that is simply not completely in view.
        Whatever is clearly fatter than the part crossing the edge is kept;
        thin parts hanging off it that run to the edge are arms and go. If
        nothing fat is left the whole blob was an arm: None."""
        contact = self._edge_contact(piece)
        if contact == 0:
            return piece
        area = int(np.count_nonzero(piece))
        r = max(2.5, 0.5 * min(1.6 * contact, 0.6 * area ** 0.5))
        dt = cv2.distanceTransform(_as_u8(piece), cv2.DIST_L2, 3)
        core = dt >= r
        if not core.any():
            return None
        near_core = cv2.distanceTransform(_as_u8(~core), cv2.DIST_L2, 3) <= 2.0 * r
        rest = _as_u8(piece & ~near_core)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        if n <= 1:
            return piece
        x0, y0, rw, rh = self.roi
        out = piece.copy()
        for i in range(1, n):
            bx, by, bw, bh = (int(v) for v in stats[i, :4])
            if (bx <= x0 or by <= y0 or bx + bw >= x0 + rw or by + bh >= y0 + rh):
                out[labels == i] = False
        return out

    def _fill_arm_gaps(self, mask: np.ndarray, fg: np.ndarray) -> np.ndarray:
        """An arm lying across the box when its outline is taken leaves a
        slot in it (a sleeve is not cardboard-coloured), and later reaches
        along that same path would go unnoticed. Close slots up to an arm
        wide — only where something is actually there now, so the gap
        between two flaps or the pad outlet stays open."""
        k = int(0.45 * self.min_area ** 0.5) | 1
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return mask
        x0, x1 = max(0, int(xs.min()) - k), min(self.ww, int(xs.max()) + k + 1)
        y0, y1 = max(0, int(ys.min()) - k), min(self.wh, int(ys.max()) + k + 1)
        sub = np.zeros((y1 - y0 + 2 * k, x1 - x0 + 2 * k), np.uint8)
        sub[k:-k, k:-k] = mask[y0:y1, x0:x1]
        closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, ker)[k:-k, k:-k].astype(bool)
        out = mask.copy()
        out[y0:y1, x0:x1] |= closed & fg[y0:y1, x0:x1]
        return out

    def _box_pieces(self, stable: np.ndarray, protect: np.ndarray
                    ) -> List[Tuple[np.ndarray, int]]:
        """The still, box-like blobs in the packing zone: [(mask, area)].
        Thin strips are dropped first (the flickering edge of someone standing
        next to the box must not tie the box to the picture edge), arms
        reaching in from the edge are cut off."""
        m = _as_u8(stable & self._boxlike & self.zone_mask & ~protect)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K7)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        speck = max(30.0, 0.05 * self.min_area)
        x0, y0, rw, rh = self.roi
        pieces = []
        self._arm_seen = False
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < speck:
                continue
            piece = labels == i
            bx, by, bw, bh = (int(v) for v in stats[i, :4])
            if bx <= x0 or by <= y0 or bx + bw >= x0 + rw or by + bh >= y0 + rh:
                piece = self._strip_arms(piece)
                if piece is None:
                    self._arm_seen = True
                    continue
                area = int(np.count_nonzero(piece))
                if area < speck:
                    continue
            pieces.append((piece, area))
        return pieces

    def _update_candidates(self, tr: float, small: np.ndarray, gray: np.ndarray,
                           fg: np.ndarray, stable: np.ndarray,
                           motion: np.ndarray) -> None:
        cfg = self.cfg
        protect = np.zeros((self.wh, self.ww), bool)
        for s in self.sessions:
            ox, oy, ow, oh = s.outer
            protect[oy:oy + oh, ox:ox + ow] = True
        pieces = self._box_pieces(stable, protect)

        # Half size is enough to KEEP following a box a hand is partly
        # covering; a new candidate (and any confirmation) needs full size.
        keep_area = 0.5 * self.min_area
        labels = np.zeros((self.wh, self.ww), np.int32)
        found = []
        why = ""

        def add(label: int, mask: np.ndarray, area: int) -> None:
            ys, xs = np.nonzero(mask)
            bbox = (int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            labels[mask] = label
            found.append((label, bbox, area, float(xs.mean()), float(ys.mean())))

        if self.sessions:
            # A box is being packed: other things at the spot are followed
            # one by one (a piece of that box, or the next box waiting).
            for k, (mask, area) in enumerate(pieces):
                if keep_area <= area <= self.max_area:
                    add(k + 1, mask, area)
        elif pieces:
            # The spot packs one box at a time, so everything box-like that
            # stands still in it is ONE box: an arm across the box, or the
            # paper hanging into it, often cuts it into pieces that would
            # each be too small.
            total = sum(a for _, a in pieces)
            if total > self.max_area:
                why = (f"too big to be a box ({100 * total / self.roi_area:.0f}% of the "
                       f"picture, boxes are up to {100 * cfg.max_box_frac:.0f}%)")
            elif total >= keep_area:
                union = pieces[0][0].copy()
                for mask, _ in pieces[1:]:
                    union |= mask
                add(1, union, int(total))
                if total < self.min_area and not self.candidates:
                    why = (f"too small to be a box ({100 * total / self.roi_area:.0f}% of the "
                           f"picture, boxes are at least {100 * cfg.min_box_frac:.0f}%)")
        if not found and not why and not self.sessions:
            if self._arm_seen:
                why = "it reaches in from the picture edge like an arm"
            elif cfg.box_color == "brown":
                anything = np.count_nonzero(stable & self.zone_mask)
                if anything >= self.min_area:
                    why = "not cardboard-coloured"
        self._set_why(tr, why)

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
            if c.label != -1:
                comp_now = labels == c.label
                c.seen = comp_now if c.seen is None else (c.seen | comp_now)
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
                c.busy = True
            # The box before anyone reached into it, so a pad put in before
            # the box is confirmed is still counted: its very first look,
            # minus whatever was moving in that look (a hand); the gaps are
            # filled in during the first moments if those parts go still.
            # (A pad needs a reach to the machine first, so it cannot be in
            # the box that early.)
            if c.early is None:
                c.early = gray.copy()
                c.early_known = ~cv2.dilate(_as_u8(motion), _K7).astype(bool)
            elif tr - c.first_t <= _EARLY_FILL_S:
                new_px = stable & ~c.early_known
                c.early[new_px] = gray[new_px]
                c.early_known |= new_px
            # Confirm at a quiet moment if there is one (a hand over the box
            # is then not mistaken for part of it), otherwise once the box
            # has stood long enough with hands working over it. A box still
            # sliding along the rollers is never still long enough to get
            # here, and a passing one would not stay min_pack_seconds anyway.
            ready = quiet or c.age(tr) >= cfg.confirm_busy_seconds
            if c.age(tr) < cfg.confirm_seconds or tr < c.rejected_until or not ready:
                continue
            comp = labels == c.label
            size = c.area
            if not quiet and c.seen is not None:
                # Busy: the box is where it has been seen still and still is
                # not empty floor now — also the part a hand is covering at
                # this moment.
                comp = (c.seen | comp) & fg
                size = int(np.count_nonzero(comp))
            if size < self.min_area:
                continue
            if not self.sessions:
                # A box arrives all at once. What was already there before it
                # came — someone standing at the spot, the pad the machine
                # pushed out during the box before — is not part of it.
                arrived = float(np.median(self.fg_since[comp]))
                came = comp & (self.fg_since >= arrived - _ARRIVE_S)
                if np.count_nonzero(came) >= max(self.min_area, 0.5 * size):
                    comp = came
            comp = self._clean_footprint(comp, small)
            comp = self._fill_arm_gaps(comp, fg & self.zone_mask)
            if self.sessions:
                # One box is packed at a time. Anything else that stands still
                # at the spot meanwhile is either a piece of that box (a hand
                # across it splits it in two; part of it was hidden when it was
                # confirmed) — or it waits until that box has left.
                owner = self._overlapping_session(comp)
                if owner is not None:
                    # ... a piece of the box was there since the box came (it
                    # was only hidden). What appeared later — the next pad
                    # hanging out of the machine — is not.
                    part = comp & (self.fg_since <= owner.start_t + 2 * _ARRIVE_S)
                    if np.count_nonzero(part) >= 0.5 * np.count_nonzero(comp):
                        self._grow_footprint(owner, part)
                    self.candidates.remove(c)
                continue
            ok, reason = self._accept(comp, int(np.count_nonzero(comp)), small)
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
            self._absorb_fragments(c, comp)
            self._start_session(tr, comp, c, gray)

    def _set_why(self, tr: float, why: str) -> None:
        """Remember why what stands at the spot is not taken as a box, and
        say so in the log once it has stayed that way for a moment."""
        if why != self._why:
            self._why, self._why_since = why, tr
        if why and why != self._why_logged and tr - self._why_since >= 2.0:
            self._why_logged = why
            log.info("Something at the spot is not taken as a box: %s", why)
        elif not why and tr - self._why_since >= 2.0:
            self._why_logged = ""

    def _overlapping_session(self, comp: np.ndarray) -> Optional[_Session]:
        """The session whose box this blob touches or nearly touches."""
        ys, xs = np.nonzero(comp)
        if xs.size == 0:
            return None
        cb = (int(xs.min()), int(ys.min()),
              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        for s in self.sessions:
            x, y, w, h = s.bbox
            # Touching, or separated only by an arm's width.
            near = expand_rect(s.bbox, max(4, int(0.25 * max(w, h))), self.ww, self.wh)
            if rect_iou(cb, near) > 0.0:
                return s
        return None

    def _grow_footprint(self, s: _Session, comp: np.ndarray) -> None:
        """Add a newly seen part of the box to its outline, within reason."""
        x, y, w, h = s.bbox
        whole = np.zeros((self.wh, self.ww), bool)
        whole[y:y + h, x:x + w] = s.fp
        union = whole | comp
        if np.count_nonzero(union) > self.max_area:
            return                       # bigger than any box: not a part
        ys, xs = np.nonzero(union)
        nx, ny = int(xs.min()), int(ys.min())
        nw, nh = int(xs.max()) - nx + 1, int(ys.max()) - ny + 1
        if (nw, nh) == (w, h) and (nx, ny) == (x, y) and np.count_nonzero(union) == s.fp_area:
            return
        ref, ref_outer, excl, hist = s.ref, s.ref_outer, s.ref_exclude, list(s.hist)
        self._set_geometry(s, (nx, ny, nw, nh), union[ny:ny + nh, nx:nx + nw],
                           keep_interior=True)
        # The pad comparison keeps its rectangle and before-picture; the outer
        # crop changed size, so the shove check restarts from the next frame.
        s.ref, s.ref_exclude = ref, excl
        s.ref_outer = None
        s.hist.extend(h[:4] + (None, None) for h in hist)
        log.info("Box %d outline extended (part of it was hidden)", s.sid)

    def _absorb_fragments(self, c: _Candidate, comp: np.ndarray) -> None:
        """A hand across a small box splits it into pieces that were followed
        as separate candidates. Whatever lies on the confirmed box was the box
        all along: take over the earliest clock and "before" picture."""
        for o in list(self.candidates):
            x, y, w, h = o.bbox
            cx, cy = min(x + w // 2, self.ww - 1), min(y + h // 2, self.wh - 1)
            if not comp[cy, cx]:
                continue
            first = min(c.first_t, o.first_t)
            if o.early is not None:
                # Only a view from the box's first moments is a "before"
                # picture: a piece first seen later may already show a pad.
                if c.early is None:
                    if o.first_t - first <= _EARLY_FILL_S:
                        c.early, c.early_known = o.early, o.early_known
                elif o.first_t - first <= _EARLY_FILL_S:
                    take = o.early_known & (~c.early_known | (o.first_t < c.first_t))
                    c.early[take] = o.early[take]
                    c.early_known |= o.early_known
            c.first_t = first
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
        # Every sizeable cardboard piece, not just the largest: an arm resting
        # across the box (and the pad outlet, which is left out) can cut the
        # box into two halves of about the same size.
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = 1 + np.flatnonzero(areas >= 0.25 * areas.max())
        piece = np.isin(labels, big)
        if np.count_nonzero(piece) < 0.5 * np.count_nonzero(comp):
            return comp
        return cv2.morphologyEx(_as_u8(piece), cv2.MORPH_CLOSE, _K7).astype(bool) & comp

    def _accept(self, comp: np.ndarray, area: int, small: np.ndarray) -> Tuple[bool, str]:
        cfg = self.cfg
        # 1. An arm meets the picture edge with a narrow wrist or forearm.
        #    (Arms across a box were already cut off the pieces; this catches
        #    what is left when the whole thing is one.)
        contact = self._edge_contact(comp)
        if contact > 0 and contact / max(1.0, area ** 0.5) < cfg.arm_contact_ratio:
            stripped = self._strip_arms(comp)
            if stripped is None or np.count_nonzero(stripped) < self.min_area:
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
            # Hands were already at work on the box while we were making sure
            # it is a box: that is a reach in progress. It will be judged when
            # it ends, against how the box looked when it arrived (parts never
            # seen uncovered are left out of that comparison).
            early = np.where(c.early_known, c.early, gray)
            s.ref = snapshot(self._crop(early, s.interior))
            s.ref_outer = self._crop(early, s.outer).astype(np.float32)
            s.ref_exclude = ~self._crop(c.early_known, s.interior).copy()
            s.in_reach = True
            s.reach_start_t = c.first_t
            s.hist.clear()
            # The hand may already have left the box while it was being
            # recognised: that quiet time counts towards ending this reach
            # (otherwise a quick next pad would be taken as part of it).
            # "Quiet since" = when the last 2% of the box last moved.
            ox, oy, ow, oh = s.outer
            lm = self.last_motion[oy:oy + oh, ox:ox + ow][s.act_mask]
            last = (float(np.percentile(lm, 100.0 * (1.0 - self.cfg.reach_end_motion)))
                    if lm.size else tr)
            s.quiet_since = last if last < tr else None

    # -- background upkeep ---------------------------------------------------

    def _maintain_background(self, tr: float, small: np.ndarray,
                             fg_raw: np.ndarray, still: np.ndarray) -> None:
        cfg = self.cfg
        protect = np.zeros((self.wh, self.ww), bool)
        for s in self.sessions:
            ox, oy, ow, oh = expand_rect(s.outer, 4, self.ww, self.wh)
            protect[oy:oy + oh, ox:ox + ow] = True
        keep_present = protect.copy()
        # Box-sized things at the spot are never folded into the empty-spot
        # picture while they are being looked at — including while one is set
        # aside for a moment (an arm resting on it, say). A box that stays
        # still for long must not vanish just because of that.
        for c in self.candidates:
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
        if self._why and tr - self._why_since >= 1.0:
            return f"waiting for a box (ignoring what is at the spot: {self._why})"
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
            "outlet": self.to_full(self.outlet) if self.outlet else None,
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
