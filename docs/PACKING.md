# Packing-Station Monitoring

With `packing.enabled: true`, the system does three things per box:

1. **counts the box** when it crosses the counting line (as always),
2. **counts the pieces** (pads / sheets of paper) the packer places into it,
3. **times the pack** — from the box stopping at the station to it leaving.

Every count event then carries `pieces` and `pack_seconds` — in the log line,
the dashboard, the SQLite database, and the daily CSV:

```
BOX #14 (track 27, 152x121 px) — 3 pieces, packed in 11.4 s
```

## How it works (so you can reason about it)

- A box that was **seen arriving and stops inside the packing zone** starts a
  *packing session*. Its outline is frozen for the whole session.
- The system watches a **band around the box**: the packer's arm must cross
  it to reach inside (arms always come from the frame edge). Foreground
  appearing in the band *above its baseline* = *hand present*. A neighbor
  already parked in the band when the session starts is baselined out from
  frame one; one that arrives mid-session is recognized as static (no motion
  inside the box) within ~1 s and absorbed. Best practice remains a physical
  gap of more than `ring_px` between the packing spot and the queue — see
  Tuning.
- While the box is parked, it is **stationary** — so any pixel motion inside
  the box interior is the hand or a dropped pad. One completed
  reach-in/reach-out **with real motion inside the box** = one insertion.
- When the box **slides away**, the session ends. Its piece count and pack
  time attach to the box's count event at the line (matched by track,
  falling back to first-in-first-out — conveyor order is preserved).

### What it can and cannot count

The camera counts **reaches into the box**, verified by motion inside it.

| Situation | Result |
|---|---|
| One pad per reach (normal rhythm) | counted exactly |
| Packer's hand pauses, adjusts, hovers | still one visit — hysteresis + debounce absorb it |
| Empty-handed reach that touches the contents | **counted** — enable `appearance_check: true` to require the contents to look different afterwards |
| Two pads placed in a single reach | counted **once** — set `pieces_per_visit` if your process uses fixed bundles |
| Arm passing near the box without reaching in | not counted (no motion inside the box) |
| Box pushed out by hand | not counted as an insertion (the box must still be at its spot when a reach ends for the reach to count) |
| Packer takes a long break mid-pack | the session times out, then **resumes with its pieces** when work continues; pack time spans arrival → departure, breaks included |

If the count must be exact against multi-pad reaches, the camera alone cannot
guarantee it — pair the system with process discipline (one pad per reach) or
use `expected_pieces` as a QA check instead of an absolute truth.

## Station layout

```
          camera (top-down)
   ┌──────────────────────────┐
   │  ┌────────────────────┐  │   ← boxes enter here
   │  │    PACKING ZONE    │  │
   │  │   box stops here   │  │   ← packer stands to the LEFT/RIGHT/TOP
   │  └────────────────────┘  │
   │──────── count line ──────│   ← line at 0.78, past the zone
   └──────────────────────────┘   ← boxes exit here
```

- The **zone** (`packing.zone`) covers where boxes stop. The **line**
  (`counting.line_position`) sits *after* it: pack first, count after.
- Keep the line at least **half a box-length** away from the exit edge of the
  frame, and the zone at least a box-length away from the line, so a box plus
  the packer's arm never straddles the line.
- The packer may stand on any side. Arms are automatically excluded from box
  counting because they stay connected to the frame edge — a box crossing the
  line never touches the travel-axis edges at that moment.
- `tracking.max_disappeared: 30` (set in the shipped config) lets the box
  keep its identity while the packer's arm briefly merges with or occludes it.

## Verifying it on your bench (no hardware)

```bash
python3 tools/make_packing_video.py --out data/packing_test.mp4 --pieces 3 2 4
python3 -m boxcounter --source data/packing_test.mp4 --no-web
```

The generator simulates the full station (indexed belt, boxes stopping, an
arm placing pads, departure across the line) and prints the ground truth.
The automated tests (`tests/test_packing*.py`) assert exact piece counts on
these videos.

## Tuning

Watch the live view (`http://<pi>:8080`): the teal rectangle is the zone, the
active box shows `pcs:N 12s` and flips yellow with `HAND` while a reach is in
progress. Watch a few boxes get packed and check the overlay counts along.

| Symptom | Fix |
|---|---|
| Reaches missed entirely | lower `arm_enter_frac` (0.06 → 0.04); check the arm actually crosses the band in the mask view; widen `ring_px` |
| Quick reaches missed | lower `min_visit_frames` (4 → 2); raise camera fps |
| Double-counted reaches (hand hovering at the box edge) | raise `exit_frames` (3 → 6) so brief pull-backs don't split one reach in two |
| Empty-handed adjustments counted | `appearance_check: true`; raise `interior_motion_frac` |
| Pieces counted while the box arrives/leaves | shrink the zone so boxes only qualify when fully parked; raise `dwell_frames` |
| Session never starts | box must be *fully inside the frame*, *seen arriving* (`min_arrival_px`) and *stopped*; check the zone covers the parking spot (and lies inside `processing.roi`); check `dwell_speed_px` vs residual jitter |
| Reaches merge into one while the queue touches the band | keep a physical gap larger than `ring_px` between the packing spot and the queued box (or reduce `ring_px`) — with a box inside the band, individual reaches can't always be separated |
| Session ends mid-pack | raise `track_grace_frames`; check the mask view — the box must stay detected while parked (freeze-learning keeps it, but verify) |
| Wrong pack times | pack time runs from stop to departure; `depart_frames` adds ~0.2 s; persistent offsets usually mean the box creeps (raise `dwell_speed_px`) |
| Next box queues right against the ring | leave a gap between packing spot and queue, or reduce `ring_px` |

## The data

- **Dashboard**: pieces total, last box (pcs/time), averages, live
  "packing now" status; per-event pcs + pack columns, highlighted red when
  `expected_pieces` is set and missed.
- **SQLite** (`data/boxcount.db`): `events.pieces`, `events.pack_seconds`
  (NULL for boxes counted with packing disabled). Existing databases are
  migrated automatically on first start.
- **CSV**: two extra columns, `pieces` and `pack_seconds`.
- **API**: `GET /api/stats` gains `pieces_total`, `last_pieces`,
  `last_pack_s`, `avg_pieces`, `avg_pack_s`, and `packing_now` with the live
  session state.

Useful queries:

```bash
# pieces and pack time per box, today
sqlite3 data/boxcount.db \
  "SELECT iso, pieces, pack_seconds FROM events WHERE iso >= date('now') ORDER BY id;"

# average pack time per hour
sqlite3 data/boxcount.db \
  "SELECT substr(iso,1,13) h, COUNT(*), AVG(pack_seconds) FROM events GROUP BY h;"

# boxes that left with the wrong piece count (expected 3)
sqlite3 data/boxcount.db \
  "SELECT iso, pieces FROM events WHERE pieces IS NOT NULL AND pieces != 3;"
```
