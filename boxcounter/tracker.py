"""Lightweight centroid tracker.

Conveyor motion is nearly constant-velocity and unidirectional, so a simple
predict-and-match centroid tracker with greedy nearest-neighbour assignment
is accurate and costs microseconds per frame — no Kalman filter needed.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

from .detector import Detection

log = logging.getLogger(__name__)

_VEL_ALPHA = 0.5  # velocity smoothing: v = (1-a)*v + a*measured


class Track:
    def __init__(self, track_id: int, det: Detection):
        self.track_id = track_id
        self.centroid: Tuple[float, float] = det.centroid
        self.bbox = det.bbox
        self.area = det.area
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.hits = 1                    # frames with a real detection
        self.misses = 0                  # consecutive frames without one
        self.counted = False             # set by the counter after crossing
        self.start_centroid = det.centroid
        self.history: deque = deque(maxlen=64)
        self.history.append(det.centroid)

    def predict(self) -> Tuple[float, float]:
        return (self.centroid[0] + self.velocity[0],
                self.centroid[1] + self.velocity[1])

    def update(self, det: Detection) -> None:
        dx = det.centroid[0] - self.centroid[0]
        dy = det.centroid[1] - self.centroid[1]
        self.velocity = ((1 - _VEL_ALPHA) * self.velocity[0] + _VEL_ALPHA * dx,
                         (1 - _VEL_ALPHA) * self.velocity[1] + _VEL_ALPHA * dy)
        self.centroid = det.centroid
        self.bbox = det.bbox
        self.area = det.area
        self.hits += 1
        self.misses = 0
        self.history.append(det.centroid)

    def coast(self) -> None:
        """Advance position by current velocity while unseen, so a track that
        blinks out right at the counting line still crosses it."""
        self.misses += 1
        self.centroid = self.predict()
        self.history.append(self.centroid)

    def travel(self, axis_index: int) -> float:
        """Signed displacement along the travel axis since first seen."""
        return self.centroid[axis_index] - self.start_centroid[axis_index]


class CentroidTracker:
    def __init__(self, max_distance: float, max_disappeared: int = 10):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.tracks: Dict[int, Track] = {}
        self._next_id = itertools.count(1)

    def update(self, detections: List[Detection]) -> List[Track]:
        """Match detections to tracks; returns all live tracks."""
        # Greedy assignment: consider all (track, detection) pairs closest
        # first, gate by max_distance.
        pairs = []
        track_items = list(self.tracks.values())
        for ti, tr in enumerate(track_items):
            px, py = tr.predict()
            for di, det in enumerate(detections):
                d = ((px - det.centroid[0]) ** 2 + (py - det.centroid[1]) ** 2) ** 0.5
                if d <= self.max_distance:
                    pairs.append((d, ti, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_dets = set()
        for d, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            track_items[ti].update(detections[di])
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Unmatched tracks coast, then expire.
        for ti, tr in enumerate(track_items):
            if ti not in matched_tracks:
                tr.coast()
                if tr.misses > self.max_disappeared:
                    del self.tracks[tr.track_id]

        # Unmatched detections spawn new tracks.
        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = next(self._next_id)
                self.tracks[tid] = Track(tid, det)

        return list(self.tracks.values())
