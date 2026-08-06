"""Unit tests for the packing monitor state machine (fabricated frames)."""

import numpy as np

from boxcounter.config import PackingConfig
from boxcounter.detector import Detection
from boxcounter.packing import PackingMonitor
from boxcounter.tracker import Track

W, H = 640, 480
BOX = (240, 120, 160, 120)          # center (320, 180) — inside default zone
DT = 1 / 30


def mk_track(tid=1, bbox=BOX):
    x, y, w, h = bbox
    return Track(tid, Detection(bbox=bbox, centroid=(x + w / 2, y + h / 2),
                                area=float(w * h)))


def scene(arm_tip=None, jitter=0, pad=False):
    """Gray frame + foreground mask: box always present; optional arm reaching
    up from the bottom edge to arm_tip (y). jitter varies arm pixels so the
    interior sees motion; pad draws a bright pad inside the box."""
    gray = np.full((H, W), 110, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    x, y, w, h = BOX
    gray[y:y + h, x:x + w] = 150
    mask[y:y + h, x:x + w] = 255
    if pad:
        gray[y + 30:y + 70, x + 40:x + 110] = 210
    if arm_tip is not None:
        ax0, ax1 = 275, 365          # 90 px wide, centered on the box
        gray[arm_tip:H, ax0:ax1] = 60 + jitter * 45
        mask[arm_tip:H, ax0:ax1] = 255
    return gray, mask


def run(mon, tr, n, t, arm_tip=None, moving_arm=True, pad=False):
    for i in range(n):
        gray, mask = scene(arm_tip=arm_tip, jitter=(i % 2) if moving_arm else 0,
                           pad=pad)
        mon.update(t, gray, mask, [tr])
        t += DT
    return t


def new_monitor(**kw):
    # min_arrival_px=0: unit-test tracks are born parked; the arrival gate
    # has its own dedicated test.
    kw.setdefault("min_arrival_px", 0.0)
    cfg = PackingConfig(enabled=True, **kw)
    return PackingMonitor(cfg, (H, W))


def test_activation_after_dwell():
    mon = new_monitor()
    tr = mk_track()
    t = run(mon, tr, 4, 0.0)
    assert mon.session is None, "needs dwell_frames slow frames"
    t = run(mon, tr, 2, t)
    assert mon.session is not None
    assert mon.session.track_id == 1


def test_single_visit_counts_one_piece():
    mon = new_monitor()
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)                       # activate
    t = run(mon, tr, 8, t, arm_tip=160)            # arm deep in the box, moving
    t = run(mon, tr, 6, t)                         # arm withdrawn
    assert mon.session.pieces == 1
    t = run(mon, tr, 8, t, arm_tip=160)            # second reach
    t = run(mon, tr, 6, t)
    assert mon.session.pieces == 2


def test_arm_flicker_ignored():
    mon = new_monitor()
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)
    t = run(mon, tr, 1, t, arm_tip=160)            # 1 frame < enter_frames
    t = run(mon, tr, 8, t)
    assert mon.session.pieces == 0


def test_visit_without_interior_motion_not_counted():
    mon = new_monitor()
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)
    # arm stays in the ring band below the box but never reaches the interior,
    # and doesn't change pixels (static)
    t = run(mon, tr, 8, t, arm_tip=250, moving_arm=False)
    t = run(mon, tr, 6, t)
    assert mon.session.visits >= 1, "the arm visit itself is registered"
    assert mon.session.pieces == 0, "but no insertion without interior motion"


def test_departure_completes_session_and_claim():
    mon = new_monitor()
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)
    t = run(mon, tr, 8, t, arm_tip=160)
    t = run(mon, tr, 6, t)
    start = mon.session.start_t
    # box drives away: bbox slides out of the packed position
    tr.bbox = (240, 340, 160, 120)
    tr.centroid = (320.0, 400.0)
    tr.velocity = (0.0, 8.0)
    t = run(mon, tr, 8, t)
    assert mon.session is None, "session should have ended on departure"
    s = mon.claim_for_track(1, t)
    assert s is not None and s.pieces == 1
    assert s.pack_seconds is not None and s.pack_seconds > 0
    assert mon.claim_for_track(1, t) is None, "a session is claimed only once"


def test_claim_falls_back_to_fifo():
    mon = new_monitor()
    tr = mk_track(tid=5)
    t = run(mon, tr, 6, 0.0)
    t = run(mon, tr, 8, t, arm_tip=160)
    t = run(mon, tr, 6, t)
    tr.bbox = (240, 360, 160, 120)
    tr.centroid = (320.0, 420.0)
    tr.velocity = (0.0, 9.0)
    t = run(mon, tr, 8, t)
    s = mon.claim_for_track(99, t)     # different id (tracking churn)
    assert s is not None and s.pieces == 1


def test_session_rebinds_after_track_loss():
    mon = new_monitor()
    tr = mk_track(tid=1)
    t = run(mon, tr, 6, 0.0)
    assert mon.session.track_id == 1
    tr2 = mk_track(tid=7)              # tracker lost the box, re-spawned it
    t = run(mon, tr2, 3, t)
    assert mon.session is not None
    assert mon.session.track_id == 7, "session should re-bind by overlap"


def test_pieces_per_visit_multiplier():
    mon = new_monitor(pieces_per_visit=3)
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)
    t = run(mon, tr, 8, t, arm_tip=160)
    t = run(mon, tr, 6, t)
    assert mon.session.pieces == 3


def test_arm_track_never_activates_session():
    """A stationary arm blob touches the frame edge — not a box."""
    mon = new_monitor()
    arm_bbox = (280, 300, 80, 180)     # reaches the bottom edge (y+h == 480)
    tr = mk_track(bbox=arm_bbox)
    run(mon, tr, 10, 0.0)
    assert mon.session is None


def test_ghost_blob_born_parked_never_activates():
    """A blob that never moved (background-rebuild ghost) can't get a session."""
    mon = new_monitor(min_arrival_px=30.0)
    tr = mk_track()                    # born exactly where it sits
    run(mon, tr, 10, 0.0)
    assert mon.session is None
    # ...but a box seen arriving does activate
    tr2 = mk_track(tid=2)
    tr2.start_centroid = (320.0, 40.0)   # entered from the top of the frame
    run(mon, tr2, 6, 1.0)
    assert mon.session is not None


def _add_neighbor(gray, mask):
    """The next box queued 10 px below the frozen box, inside the 28 px band."""
    gray[250:370, 230:410] = 145
    mask[250:370, 230:410] = 255


def _add_side_arm(gray, mask, jitter=0, tip_x=330):
    """Horizontal reach from the left frame edge into the box interior —
    the realistic geometry when the lower band is occupied by the queue."""
    gray[150:210, 0:tip_x] = 60 + jitter * 45
    mask[150:210, 0:tip_x] = 255


def test_queued_neighbor_at_start_is_baselined():
    """A box already parked inside the watch band when the session starts must
    not read as a hand, and a real reach must still be counted."""
    mon = new_monitor()
    tr = mk_track()

    def frame(arm=False, jitter=0):
        gray, mask = scene()
        _add_neighbor(gray, mask)
        if arm:
            _add_side_arm(gray, mask, jitter)
        return gray, mask

    t = 0.0
    for i in range(6):                              # activate with neighbor
        g, m = frame()
        mon.update(t, g, m, [tr]); t += DT
    assert mon.session is not None
    for i in range(20):                             # idle: neighbor is static
        g, m = frame()
        mon.update(t, g, m, [tr]); t += DT
    assert mon.session.pieces == 0, "static neighbor must not count"
    for i in range(8):                              # real reach from the side
        g, m = frame(arm=True, jitter=i % 2)
        mon.update(t, g, m, [tr]); t += DT
    for i in range(8):
        g, m = frame()
        mon.update(t, g, m, [tr]); t += DT
    assert mon.session.pieces == 1, "reach with neighbor present must count"


def test_neighbor_arriving_mid_session_self_heals():
    """A box queuing INTO the band mid-session must not latch the hand state
    forever — the static-exit closes it and re-baselines."""
    mon = new_monitor(static_exit_frames=10)
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)                        # activate, clean band

    def frame(arm=False, jitter=0):
        gray, mask = scene()
        _add_neighbor(gray, mask)
        if arm:
            _add_side_arm(gray, mask, jitter)
        return gray, mask

    for i in range(30):                             # neighbor arrives, static
        g, m = frame()
        mon.update(t, g, m, [tr]); t += DT
    assert not mon.hand_in, "static occupant must not hold the hand state"
    assert mon.session.pieces == 0
    for i in range(8):                              # then a real reach
        g, m = frame(arm=True, jitter=i % 2)
        mon.update(t, g, m, [tr]); t += DT
    for i in range(8):
        g, m = frame()
        mon.update(t, g, m, [tr]); t += DT
    assert mon.session.pieces == 1


def test_occluded_departure_no_phantom_piece():
    """Box pushed/slid away while its own track is lost: the ring empties like
    an arm withdrawing, but no piece may be awarded (box is gone)."""
    mon = new_monitor()
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)
    assert mon.session is not None
    # box slides: mask shows the box moving down through its own ring while
    # the bound track goes missing (tracker lost it in the merge)
    x, y, w, h = BOX
    for i in range(14):
        gray = np.full((H, W), 110, np.uint8)
        mask = np.zeros((H, W), np.uint8)
        ny = y + 12 * (i + 1)
        gray[ny:min(H, ny + h), x:x + w] = 150 + (i % 2) * 20
        mask[ny:min(H, ny + h), x:x + w] = 255
        mon.update(t, gray, mask, [])               # no tracks at all
        t += DT
    # belt now empty, ring clear
    for i in range(8):
        gray = np.full((H, W), 110, np.uint8)
        mon.update(t, gray, np.zeros((H, W), np.uint8), [])
        t += DT
    s = mon.session
    total = (s.pieces if s else 0) + sum(c.pieces for c in mon.completed)
    assert total == 0, "an occluded departure must not create a phantom piece"


def test_rebind_refuses_travelling_box():
    """A DIFFERENT box arriving over the old spot must not inherit the session."""
    mon = new_monitor()
    tr = mk_track(tid=1)
    t = run(mon, tr, 6, 0.0)
    assert mon.session.track_id == 1
    # our track vanishes; a new track overlaps the spot but was born upstream
    # and is still rolling — that's the NEXT box, not ours
    newcomer = mk_track(tid=9)
    newcomer.start_centroid = (320.0, 20.0)
    newcomer.velocity = (0.0, 6.0)
    for _ in range(5):
        gray, mask = scene()
        mon.update(t, gray, mask, [newcomer]); t += DT
    assert mon.session is None or mon.session.track_id != 9


def test_timeout_session_resumes_for_same_box():
    """A long break mid-pack (session timeout) must not fragment: when the
    same box re-activates, its pieces are preserved in one session."""
    mon = new_monitor(max_session_s=1.0)
    tr = mk_track()
    t = run(mon, tr, 6, 0.0)
    t = run(mon, tr, 8, t, arm_tip=160)
    t = run(mon, tr, 6, t)
    assert mon.session.pieces == 1
    # 40 frames > 1.0 s: the session times out mid-run and, because the box is
    # still parked, re-activates — resuming the SAME session with its pieces.
    t = run(mon, tr, 40, t)
    for _ in range(60):     # ride out timeout/re-dwell phase alignment
        t = run(mon, tr, 1, t)
        if mon.session is not None:
            break
    assert mon.session is not None
    assert mon.session.pieces == 1, "resumed session must keep its pieces"
    assert len([s for s in mon.completed if not s.claimed]) == 0, \
        "timeouts must not leave duplicate unclaimed sessions behind"
