# Calibration & Tuning

Everything is set in `config/config.yaml`. After each change:
`sudo systemctl restart boxcounter` (or Ctrl-C and rerun when testing in the
foreground). This page lists the workflow, then every knob with its symptoms.

## Workflow

1. `python3 tools/calibrate.py` → inspect `data/calibration_snapshot.jpg`
   (10 % grid + current ROI in blue + counting line in yellow).
2. Set `processing.roi` to belt-only. Repeat step 1 until right.
3. Set `counting.line_position` mid-path; set `counting.direction`.
4. Run boxes; `python3 tools/calibrate.py --check --seconds 30` suggests
   `min_area_frac`/`max_area_frac`.
5. Start the counter, open `http://<pi>:8080`, toggle **show mask**, and
   sanity-check with real traffic: solid white boxes, black belt.
6. Validate with a counted batch of ≥ 20 boxes, mixed open/closed.

## The mask is the truth

Nearly every mis-count is visible in the mask view (dashboard → "show mask"):

| Mask looks like | Problem | Fix |
|---|---|---|
| Box is white but split into 2+ pieces | interior resembles belt (open box) | raise `close_kernel` (e.g. 31→41), raise `merge_gap_px` |
| Box barely visible / patchy | low contrast | lower `mog2_var_threshold` (32→20); check `use_color: true`; improve lighting |
| Boxes fade out mid-belt after running a while | background absorbed the traffic | ensure `freeze_learning_fg_fraction: 0.02`; or switch to `method: static` |
| Random white speckles everywhere | sensor noise / flickering light | raise `blur_kernel` (5→7), raise `open_kernel` (5→9), raise `mog2_var_threshold`; fix the light (no PWM) |
| White stripe that follows a box (ghost) | shadows counted as foreground | keep `detect_shadows: true`; light more diffusely |
| Entire frame flashes white occasionally | auto-exposure hunting | keep `lock_exposure: true`; or set manual `exposure_time_us` |
| Belt edges/rollers always white | ROI too large | shrink `processing.roi` |

## Counting problems

| Symptom | Cause | Fix |
|---|---|---|
| Two boxes counted as one | they touch or nearly touch on the belt | lower `merge_gap_px`; lower `close_kernel`; if they physically touch, only physical spacing (or a smarter detector) separates them |
| One box counted twice | track ID lost mid-frame and re-crossed | raise `tracking.max_disappeared` (10→20); raise `max_distance_frac` for fast belts |
| Boxes missed at high belt speed | too few frames on screen | raise `camera.fps` (IMX219 does 47 fps at 1280×720-binned, ~30 at default); raise `max_distance_frac`; lower `min_hits` to 2 |
| Sporadic counts with empty belt | noise blobs crossing the line | raise `min_area_frac`; raise `counting.min_travel_frac`; raise `tracking.min_hits` |
| Nothing ever counted | direction reversed | flip `counting.direction` (`positive` = downward/rightward in the *image*) |
| Counts happen at the wrong place | line position | move `counting.line_position`; keep it ≥ one box-length away from where boxes first appear |

## Parameter reference (the ones that matter)

### camera
- `width/height/fps` — 640×480@30 is the sweet spot; the `sensor_size:
  [1640, 1232]` line keeps the full field of view (don't remove it).
- `lock_exposure: true` — freeze AE/AWB after `warmup_seconds`. For fully
  deterministic imaging set `exposure_time_us` (e.g. 8000) + `analogue_gain`
  (e.g. 2.0) instead; shorter exposure = less motion blur on fast belts.
- `hflip/vflip` — mount the image the way your brain likes it.

### processing
- `method` — `mog2` adapts by itself; `static` (+ `tools/capture_background.py`)
  is unconditionally stable under constant lighting. When in doubt start with
  `mog2`, switch to `static` if you fight absorption or slow drift artifacts.
- `use_color: true` — keep on. Grayscale misses brown-box-on-gray-belt.
- `mog2_var_threshold` — sensitivity (lower = more sensitive). 16–48 useful.
- `freeze_learning_fg_fraction: 0.02` — model learns only on (nearly) empty
  belt. If your belt is *never* empty, use `method: static`.
- `blur_kernel/open_kernel` — noise suppression before/after thresholding.
- `close_kernel` — the open-box fuser. ≈ ¼ box width in pixels. Too big
  merges separate boxes that pass close together.
- `min_area_frac/max_area_frac` — accepted blob size as fraction of ROI
  area. Get numbers from `calibrate.py --check`.
- `warmup_frames: 60` — detections ignored during model warm-up at startup.

### tracking
- `max_distance_frac: 0.15` — how far (fraction of frame diagonal) a box may
  move between frames and still be the same box. Belt speed in px/frame must
  stay well below this.
- `max_disappeared: 10` — frames a track survives without detection
  (it coasts on its velocity, so brief dropouts still cross the line).
- `min_hits: 3` — detections required before a track may count.

### counting
- `line_position` + `hysteresis_frac: 0.03` — a track must clear the line by
  the hysteresis band to count; the band absorbs jitter.
- `min_travel_frac: 0.10` — a counted track must have moved this far along
  the axis, killing stationary-noise counts.

## Performance check

```bash
python3 tools/benchmark.py --frames 300        # with the camera
```

Healthy Pi 5 output at 640×480: detection < 6 ms, tracking < 0.5 ms,
end-to-end ≥ camera fps. If detection is slow, the usual cause is running at
a much higher resolution than configured.
