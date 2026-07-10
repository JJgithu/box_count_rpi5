from boxcounter.detector import Detection
from boxcounter.tracker import CentroidTracker


def det(cx, cy, w=100, h=80):
    return Detection(bbox=(int(cx - w / 2), int(cy - h / 2), w, h),
                     centroid=(cx, cy), area=float(w * h))


def test_track_identity_stable_while_moving():
    tr = CentroidTracker(max_distance=60, max_disappeared=5)
    ids = set()
    for y in range(50, 400, 15):
        tracks = tr.update([det(320, y)])
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert len(ids) == 1, "moving object should keep one ID"


def test_two_objects_keep_separate_ids():
    tr = CentroidTracker(max_distance=60, max_disappeared=5)
    first = tr.update([det(150, 100), det(450, 300)])
    id_left = next(t.track_id for t in first if t.centroid[0] < 300)
    id_right = next(t.track_id for t in first if t.centroid[0] > 300)
    for step in range(1, 10):
        tracks = tr.update([det(150, 100 + 12 * step), det(450, 300 + 12 * step)])
        assert len(tracks) == 2
        left = next(t for t in tracks if t.centroid[0] < 300)
        right = next(t for t in tracks if t.centroid[0] > 300)
        assert left.track_id == id_left
        assert right.track_id == id_right


def test_track_survives_short_dropout_by_coasting():
    tr = CentroidTracker(max_distance=60, max_disappeared=5)
    tid = None
    y = 100
    for _ in range(5):
        y += 15
        tracks = tr.update([det(320, y)])
        tid = tracks[0].track_id
    # object vanishes for 3 frames but keeps moving
    for _ in range(3):
        tracks = tr.update([])
        assert len(tracks) == 1
    y += 15 * 4
    tracks = tr.update([det(320, y)])
    assert len(tracks) == 1
    assert tracks[0].track_id == tid, "should re-match after dropout via coasting"


def test_track_expires_after_max_disappeared():
    tr = CentroidTracker(max_distance=60, max_disappeared=3)
    tr.update([det(320, 100)])
    for _ in range(3):
        assert len(tr.update([])) == 1
    assert len(tr.update([])) == 0


def test_new_detection_far_away_gets_new_id():
    tr = CentroidTracker(max_distance=50, max_disappeared=5)
    t1 = tr.update([det(100, 100)])
    tracks = tr.update([det(500, 400)])
    ids = {t.track_id for t in tracks}
    assert t1[0].track_id in ids and len(ids) == 2
