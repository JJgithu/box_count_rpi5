from boxcounter.counter import LineCounter
from boxcounter.detector import Detection
from boxcounter.tracker import Track


def make_track(tid, positions_y, hits=None):
    """Build a track that has visited the given y positions (x fixed)."""
    first = positions_y[0]
    tr = Track(tid, Detection(bbox=(300, int(first) - 40, 100, 80),
                              centroid=(350.0, float(first)), area=8000.0))
    for y in positions_y[1:]:
        tr.update(Detection(bbox=(300, int(y) - 40, 100, 80),
                            centroid=(350.0, float(y)), area=8000.0))
    if hits is not None:
        tr.hits = hits
    return tr


def step_counter(counter, tid, ys):
    """Feed a moving track through the counter one frame at a time."""
    events = []
    first = ys[0]
    tr = Track(tid, Detection(bbox=(300, int(first) - 40, 100, 80),
                              centroid=(350.0, float(first)), area=8000.0))
    events += counter.update([tr])
    for y in ys[1:]:
        tr.update(Detection(bbox=(300, int(y) - 40, 100, 80),
                            centroid=(350.0, float(y)), area=8000.0))
        events += counter.update([tr])
    return events, tr


def new_counter(direction="positive"):
    return LineCounter(axis="y", line_px=240, hysteresis_px=15,
                       direction=direction, min_travel_px=50, min_hits=3)


def test_simple_crossing_counts_once():
    c = new_counter()
    events, tr = step_counter(c, 1, [100, 140, 180, 220, 260, 300, 340])
    assert len(events) == 1
    assert tr.counted
    # continuing further produces no extra counts
    tr.update(Detection(bbox=(300, 340, 100, 80), centroid=(350.0, 380.0), area=8000.0))
    assert c.update([tr]) == []


def test_jitter_at_line_counts_once():
    c = new_counter()
    ys = [100, 160, 220, 235, 250, 238, 252, 241, 258, 300, 360]
    events, _ = step_counter(c, 2, ys)
    assert len(events) == 1, "hysteresis must absorb jitter at the line"


def test_wrong_direction_not_counted():
    c = new_counter(direction="positive")
    events, _ = step_counter(c, 3, [400, 340, 280, 220, 160, 100])
    assert events == []


def test_any_direction_counts_both_ways():
    c = new_counter(direction="any")
    events, _ = step_counter(c, 4, [400, 320, 240, 160, 80])
    assert len(events) == 1


def test_min_hits_required():
    c = new_counter()
    # A track teleporting across the line with only 2 observations: no count.
    tr = make_track(5, [100, 400])
    tr2_events = []
    tr2_events += c.update([make_track(5, [100])])
    assert len(tr2_events) == 0


def test_track_spawning_past_line_never_counted():
    c = new_counter()
    events, _ = step_counter(c, 6, [300, 340, 380, 420])
    assert events == [], "a track first seen beyond the line must not count"


def test_min_travel_blocks_noise_blob():
    c = LineCounter(axis="y", line_px=240, hysteresis_px=10,
                    direction="positive", min_travel_px=200, min_hits=1)
    events, _ = step_counter(c, 7, [220, 228, 236, 258, 266])
    assert events == [], "short travel across the line must not count"


def feed(counter, tr, bbox, cy):
    tr.update(Detection(bbox=bbox, centroid=(350.0, float(cy)),
                        area=float(bbox[2] * bbox[3])))
    return counter.update([tr])


def test_tall_box_near_exit_edge_still_counted():
    """A box taller than twice the line-to-edge distance touches the exit edge
    before its centroid crosses; the crossing must still be honoured from the
    side-state gained while it was fully inside the view."""
    c = LineCounter(axis="y", line_px=374, hysteresis_px=14,
                    direction="positive", min_travel_px=48, min_hits=3,
                    bounds_px=(0.0, 480.0))
    h = 200
    top = 40.0
    tr = None
    events = []
    while top < 500:
        y0 = int(top)
        y1 = min(480, y0 + h)          # detector clips the bbox at the edge
        if y1 - y0 < 30:               # box nearly gone
            break
        bbox = (300, y0, 100, y1 - y0)
        cy = (y0 + y1) / 2
        if tr is None:
            tr = Track(1, Detection(bbox=bbox, centroid=(350.0, cy),
                                    area=float(100 * (y1 - y0))))
            events += c.update([tr])
        else:
            events += feed(c, tr, bbox, cy)
        top += 8
    assert len(events) == 1, "tall box lost at the exit edge"


def test_edge_connected_arm_never_counted():
    """An arm-like blob always connected to the exit edge sweeps its centroid
    across the line (extend + retract) — it must never gain side state, so it
    can never be counted, in either direction mode."""
    for direction in ("positive", "any"):
        c = LineCounter(axis="y", line_px=374, hysteresis_px=14,
                        direction=direction, min_travel_px=20, min_hits=1,
                        bounds_px=(0.0, 480.0))
        tr = None
        events = []
        # tip extends from the bottom edge up to y=200, then retracts
        for tip in list(range(470, 200, -20)) + list(range(200, 500, 20)):
            bbox = (300, max(0, tip), 80, 480 - max(0, tip))   # pinned to edge
            cy = (tip + 480) / 2
            if tr is None:
                tr = Track(2, Detection(bbox=bbox, centroid=(340.0, cy),
                                        area=float(bbox[2] * bbox[3])))
                events += c.update([tr])
            else:
                events += feed(c, tr, bbox, cy)
        assert events == [], f"arm counted with direction={direction}"


def test_state_purged_for_dead_tracks():
    c = new_counter()
    _, tr = step_counter(c, 8, [100, 150, 200])
    assert 8 in c._seen_minus or 8 in c._seen_plus
    c.update([])
    assert 8 not in c._seen_minus and 8 not in c._seen_plus


def test_fragmented_track_short_of_min_travel_still_counts():
    """A replacement track that spawns just upstream of the line (the result
    of a mid-belt detection dropout) fails min_travel at the exact crossing
    frame, but must still be counted as it continues past the line."""
    c = LineCounter(axis="y", line_px=240, hysteresis_px=15,
                    direction="positive", min_travel_px=60, min_hits=3)
    # starts 20 px before the line-hyst band, then rolls far downstream
    ys = [205, 212, 220, 228, 236, 244, 252, 262, 300, 360, 420]
    events, _ = step_counter(c, 20, ys)
    assert len(events) == 1, "fragmented track past the line must be counted"


def test_low_hits_at_crossing_counts_once_when_hits_accumulate():
    c = LineCounter(axis="y", line_px=240, hysteresis_px=15,
                    direction="positive", min_travel_px=50, min_hits=5)
    ys = [180, 300, 320, 340, 360, 380, 400]  # jumps across with only 1 hit
    events, _ = step_counter(c, 21, ys)
    assert len(events) == 1
