# Calibration — Step by Step

Follow these steps **in order**. Each one depends on the previous being right,
and most "it counts wrong" problems are really an earlier step being skipped.

Everything you change lives in one file, `config/config.yaml`. After every
change:

```bash
sudo systemctl restart boxcounter      # if running as a service
```

**Golden rule: the mask is the truth.** Open `http://<pi-ip>:8080` and click
**show mask**. A correctly calibrated system shows **solid white boxes on a
pure black belt** — nothing else. If the mask is right, counting is almost
always right. Keep that page open throughout.

Budget about 45 minutes the first time. Steps 1–6 give you box counting;
steps 7–10 add piece counting and pack times.

---

# Part A — Box counting

## Step 1. Get a stable, well-exposed image

Everything downstream compares each frame against a learned picture of the
empty belt, so the image must not flicker, drift, or blur.

Stop the service and look at a frame:

```bash
sudo systemctl stop boxcounter
cd ~/box_count_rpi5
python3 tools/calibrate.py
```

Copy `data/calibration_snapshot.jpg` to your PC and look at it:

```bash
# run this ON YOUR PC
scp <pi-user>@boxcounter.local:box_count_rpi5/data/calibration_snapshot.jpg .
```

Check, in this order:

| Check | If wrong |
|---|---|
| **Whole belt width visible**, small margin each side | move the camera up/down (≈1.21 × height across) |
| **At least 2 box-lengths** of belt visible along travel | raise the camera |
| Image **sharp**, boxes not smeared | see motion blur below |
| Exposure sensible — belt not blown white or crushed black | see below |
| Camera **rigid**, no vibration | fix the mount before anything else |

**Motion blur** (boxes smeared along travel): set a manual exposure —
`camera.exposure_time_us: 4000` and raise `camera.analogue_gain` (e.g. `4.0`)
to compensate for the darker image. Shorter exposure = less blur.

**Flicker or brightness pumping:** keep `camera.lock_exposure: true` (the
default). Never use PWM-dimmed LED lighting.

> Do not continue until the image is stable and sharp. Every later step
> inherits these problems.

## Step 2. Set the region of interest (ROI)

The ROI is the only part of the image the detector looks at. It must cover
**belt surface only** — no rails, no rollers, no floor, no walkway where
people pass.

Read the fractions off the grid in your snapshot (labelled 0.1 … 0.9), then:

```yaml
processing:
  roi: [0.05, 0.0, 0.90, 1.0]     # [x, y, width, height] as fractions
```

Re-run `python3 tools/calibrate.py` and check the **blue** rectangle sits on
belt only. Repeat until it does.

## Step 3. Make boxes appear cleanly in the mask

Start the counter in the foreground and watch the mask:

```bash
python3 -m boxcounter
# browser -> http://<pi-ip>:8080 -> "show mask"
```

Run boxes past and compare what you see to this table:

| Mask shows | Meaning | Fix |
|---|---|---|
| Solid white boxes, black belt | correct | continue to step 4 |
| Box **split into pieces** (open box: only a rim) | interior looks like belt | raise `close_kernel` 31 → 41 → 51; raise `merge_gap_px` 24 → 40 |
| Box faint, patchy, or invisible | too little contrast | lower `mog2_var_threshold` 32 → 20; confirm `use_color: true`; add light |
| Speckles all over | sensor noise / flicker | raise `blur_kernel` 5 → 7, `open_kernel` 5 → 9, `mog2_var_threshold` |
| A grey/white shadow trailing each box | shadows | keep `detect_shadows: true`; make the lighting more diffuse |
| Belt edges or rollers white | ROI too big | back to step 2 |
| Boxes **fade out** after running a while | background absorbed the traffic | confirm `freeze_learning_fg_fraction: 0.02`; or switch to `method: static` (below) |
| Whole frame flashes white | exposure hunting | `lock_exposure: true`, or set manual exposure (step 1) |

**Rule of thumb:** `close_kernel` ≈ ¼ of a box's width **in pixels**. A
160 px-wide box → about 40.

**If your lighting is constant** (indoor, artificial), the `static` method is
more stable — it compares against a fixed photo of the empty belt and can
never "absorb" a box:

```bash
# belt RUNNING and EMPTY:
python3 tools/capture_background.py
# then in config.yaml:  processing.method: static
```

## Step 4. Set the accepted box sizes

With boxes flowing, measure what the detector actually sees:

```bash
python3 tools/calibrate.py --check --seconds 30
```

It prints something like:

```
Blob area (fraction of ROI): p10=0.071  median=0.084  p90=0.096
Suggested config:  min_area_frac: 0.035   max_area_frac: 0.240
```

Copy those two suggested values into `config.yaml`. They reject
noise specks (too small) and lighting-change blobs (too big).

If it reports **"No blobs detected"**, go back to step 3 — the mask is wrong.

## Step 5. Place the counting line

```yaml
counting:
  axis: y              # y = boxes travel vertically in the image
  line_position: 0.78  # fraction of the frame
  direction: positive  # positive = down/right in the IMAGE
```

Rules:
- Put the line where boxes are **fully visible and moving steadily**.
- Keep it **at least half a box-length** from the frame edge.
- If you're using the packing station (Part B), the line must be **after**
  the packing zone — pack first, count after.
- Boxes travelling **up** the image → `direction: negative`. Travelling
  **horizontally** → `axis: x`.

Check the **yellow** line in a fresh `python3 tools/calibrate.py` snapshot.

## Step 6. Verify box counting before going further

Run a known batch — at least 20 boxes at production speed and spacing, mixed
open and closed — and compare with the dashboard.

| Symptom | Fix |
|---|---|
| Nothing counted at all | `direction` is reversed — flip it |
| Two boxes counted as one | lower `merge_gap_px` / `close_kernel`; boxes that physically touch cannot be separated |
| One box counted twice | raise `tracking.max_disappeared` 30 → 45; raise `max_distance_frac` 0.15 → 0.25 |
| Boxes missed on a fast belt | raise `camera.fps`; raise `max_distance_frac`; lower `min_hits` to 2 |
| Counts with an empty belt | raise `min_area_frac`; raise `counting.min_travel_frac` |

**Do not move on until this is accurate.** Piece counting builds directly on
box tracking.

---

# Part B — Piece counting and pack time

Skip this entire part if you only need box counts (`packing.enabled: false`).

## Step 7. Set the packing zone

The zone is where boxes **stop to be filled**.

```yaml
packing:
  enabled: true
  zone: [0.05, 0.05, 0.90, 0.45]   # [x, y, w, h] fractions
```

Requirements:
- Covers the spot where a box **comes to rest**, with a little margin.
- Sits **inside** `processing.roi` (the detector is blind outside it) — the
  config warns you at startup if it isn't.
- Ends **before** `counting.line_position`.
- Leaves room around the parked box: the watch band extends `ring_px`
  (default 28 px) beyond the box on every side.

Verify with a snapshot — the zone is drawn in **teal**:

```bash
python3 tools/calibrate.py
```

## Step 8. Check the geometry on a real box

With the belt running and a box being packed:

```bash
python3 tools/calibrate.py --packing --seconds 60
```

The moment a box parks, it saves `data/packing_snapshot.jpg` showing the
actual geometry. Copy it to your PC and check:

- **teal** = the box the system locked onto (should match the real box),
- **yellow** = the watch band where arms are detected (should be clear belt,
  *not* overlapping the next queued box),
- **blue** = the interior it watches for pads landing (should cover where
  pads actually land).

If the yellow band overlaps the queued box, either leave a bigger physical
gap between the packing spot and the queue, or reduce `ring_px`.

If **no session ever starts**, the box must be: fully inside the frame,
inside the zone, *seen arriving* (`min_arrival_px: 30`), and actually
**stopped** (`dwell_speed_px: 1.5`). If your boxes creep while being packed,
raise `dwell_speed_px` to 3–4.

## Step 9. Tune the arm detection

Keep that same command running and **reach into the box as the packer does**,
several times. You get a live readout:

```
  time  state     signal bar           motion bar           pieces
   5.4  idle       0.000 ----|-------   0.000 ----|-------  0
   5.8  HAND IN    0.106 ####|##-----   0.284 ####|#######  1
   6.6  idle       0.000 ----|-------   0.000 ----|-------  1
```

- **signal** — how strongly an arm is seen in the watch band. The `|` in the
  bar marks the current threshold.
- **motion** — movement inside the box.
- **state** flips to `HAND IN` while a reach is detected; **pieces**
  increments when a completed reach is accepted.

What you want to see: `idle` while your hand is away, `HAND IN` for the whole
reach, and pieces incrementing **once per reach**.

At the end it prints statistics and recommended values:

```
arm signal while idle : p50=0.000 p95=0.004   (must stay BELOW arm_enter_frac)
arm signal while HAND : p50=0.110 p95=0.224   (must stay ABOVE arm_exit_frac)
interior motion, hand : p50=0.251 p95=0.319

Suggested:  arm_enter_frac: 0.060   arm_exit_frac: 0.025
Suggested:  interior_motion_frac: 0.061
```

Copy the suggested values into `config.yaml`. If it instead prints
**"idle and hand signals overlap"**, the arm isn't cleanly separable — widen
`ring_px`, improve the lighting, or move the queued box further away.

| Symptom | Fix |
|---|---|
| Reaches not detected at all | lower `arm_enter_frac`; widen `ring_px` |
| Quick reaches missed | lower `min_visit_frames` 4 → 2; raise `camera.fps` |
| One reach counted as two | raise `exit_frames` 3 → 6 (absorbs a brief pull-back) |
| Hand state stuck `HAND IN` | something static is in the band — check the geometry snapshot |

## Step 10. Tune what counts as a placed piece

A reach counts only if there was real **motion inside the box** during it.

- Empty-handed adjustments being counted → raise `interior_motion_frac`, or
  set `appearance_check: true` (also requires the contents to *look*
  different afterwards — stricter, best when pads visibly change the box).
- Real placements missed → lower `interior_motion_frac`; make sure the
  **blue** interior box from step 8 covers where pads land.

**If your packer places a fixed bundle per reach** (e.g. 3 pads in one
motion), the camera sees one reach — set:

```yaml
  pieces_per_visit: 3
```

**Optional QA check** — flag boxes that leave with the wrong count:

```yaml
  expected_pieces: 3    # warns in the log, highlights red on the dashboard
```

## Step 11. Verify piece counting

Pack 10 boxes normally and compare the dashboard against what you counted by
hand. Check the per-box table on the dashboard (`pcs` and `pack` columns), or:

```bash
sqlite3 data/boxcount.db \
  "SELECT iso, pieces, pack_seconds FROM events ORDER BY id DESC LIMIT 10;"
```

Pack times should match a stopwatch within about a second (the box's stop →
departure, so it includes any pause before the belt indexes on).

---

## Step 12. Lock it in

```bash
sudo systemctl start boxcounter        # or: restart
systemctl status boxcounter
journalctl -u boxcounter -f            # watch a few real boxes go by
```

Back up your calibration — it's the only thing here that's unique to your
line:

```bash
cp config/config.yaml ~/config-backup-$(date +%F).yaml
```

Re-check calibration after: moving the camera or lights, changing box sizes
or belt speed, or a seasonal daylight change if any daylight reaches the belt.

---

## Quick performance check

```bash
python3 tools/benchmark.py --frames 300
```

On a Pi 5 at 640×480 expect detection under 6 ms/frame and an end-to-end rate
comfortably above your camera fps. If detection is slow, something raised the
resolution — 640×480 is the intended operating point.

## Full parameter reference

Every parameter is documented inline in `config/config.yaml`. The packing
concept, station layout, and what the system can and cannot count are in
[PACKING.md](PACKING.md); day-to-day operation is in [RUNNING.md](RUNNING.md).
