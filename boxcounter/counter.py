"""Directional line-crossing counter with hysteresis.

A virtual line sits across the frame perpendicular to belt travel. A track is
counted exactly once, when it moves from confidently on one side of the line
(beyond the hysteresis band) to confidently on the other side, in the
configured direction, having travelled a minimum distance. This makes the
count immune to bounding-box jitter at the line and to conveyor vibration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .tracker import Track

log = logging.getLogger(__name__)


from typing import Optional


@dataclass
class CountEvent:
    ts: float
    track_id: int
    bbox: Tuple[int, int, int, int]
    area: float
    direction: int      # +1 = crossed toward increasing coordinate
    # Filled in by the pipeline from the packing monitor, when enabled:
    pieces: Optional[int] = None          # pads placed into this box
    pack_seconds: Optional[float] = None  # arrival -> departure at the station


_EDGE_PX = 2      # bbox closer than this to a travel-axis frame edge = touching


class LineCounter:
    def __init__(self, axis: str, line_px: float, hysteresis_px: float,
                 direction: str, min_travel_px: float, min_hits: int,
                 bounds_px: Optional[Tuple[float, float]] = None):
        self.axis_index = 0 if axis == "x" else 1
        self.line = float(line_px)
        self.hyst = max(0.0, float(hysteresis_px))
        self.direction = direction          # positive | negative | any
        self.min_travel = float(min_travel_px)
        self.min_hits = int(min_hits)
        # Extent of the detector's view along the travel axis (the ROI edges,
        # since detections are clipped to the ROI). When set, a track whose
        # bbox touches either travel-axis edge cannot ACQUIRE a confident-side
        # state: a packer's arm (always connected to an edge) therefore never
        # builds the "was before the line" history needed to be counted, no
        # matter how its blob moves. Crossings themselves stay honoured, so a
        # long box whose tail is already clipped at the exit edge while its
        # centroid crosses is still counted from the history it gained while
        # fully inside.
        self.bounds = (float(bounds_px[0]), float(bounds_px[1])) \
            if bounds_px is not None else None
        # Per track: have we ever seen it confidently before / past the line?
        # These are geometric facts, updated every frame and never consumed —
        # so the count gates (min_hits / min_travel / direction) can be
        # re-checked on later frames while the track is still past the line.
        self._seen_minus: Dict[int, bool] = {}
        self._seen_plus: Dict[int, bool] = {}

    def _direction_ok(self, crossing: int) -> bool:
        if self.direction == "any":
            return True
        return crossing == (1 if self.direction == "positive" else -1)

    def update(self, tracks: List[Track]) -> List[CountEvent]:
        events: List[CountEvent] = []
        live_ids = set()
        for tr in tracks:
            tid = tr.track_id
            live_ids.add(tid)
            p = tr.centroid[self.axis_index]

            touching = False
            if self.bounds is not None:
                b_lo, b_hi = self.bounds
                # A coasted track drifting out of view can neither gain sides
                # nor cross.
                if p < b_lo or p > b_hi:
                    continue
                lo = tr.bbox[self.axis_index]
                hi = lo + tr.bbox[self.axis_index + 2]
                touching = lo < b_lo + _EDGE_PX or hi > b_hi - _EDGE_PX

            # Which side is the track confidently on this frame (None = in the
            # hysteresis dead band)?
            side = None
            if p < self.line - self.hyst:
                side = -1
            elif p > self.line + self.hyst:
                side = +1

            # A countable crossing exists when the track is confidently on one
            # side now AND was confidently on the opposite side at some earlier
            # frame. Crucially we do NOT clear the "seen" history on a count
            # attempt, so if the gates fail on the exact crossing frame the
            # crossing can still be honoured a few frames later once the track
            # has accumulated enough travel / hits.
            crossing = 0
            if side == +1 and self._seen_minus.get(tid):
                crossing = +1
            elif side == -1 and self._seen_plus.get(tid):
                crossing = -1

            if crossing != 0 and not tr.counted:
                travel = tr.travel(self.axis_index)
                if (tr.hits >= self.min_hits
                        and self._direction_ok(crossing)
                        and abs(travel) >= self.min_travel
                        and (travel > 0) == (crossing > 0)):
                    tr.counted = True
                    events.append(CountEvent(
                        ts=time.time(),
                        track_id=tid,
                        bbox=tr.bbox,
                        area=tr.area,
                        direction=crossing,
                    ))
                    log.debug("Counted track %d (travel %.0f px)", tid, travel)

            # Record the confident side AFTER the crossing check — and never
            # while the bbox touches a travel-axis edge (see bounds above).
            if not touching:
                if side == -1:
                    self._seen_minus[tid] = True
                elif side == +1:
                    self._seen_plus[tid] = True

        # Drop per-track state for tracks that no longer exist.
        for store in (self._seen_minus, self._seen_plus):
            for tid in [t for t in store if t not in live_ids]:
                del store[tid]
        return events
