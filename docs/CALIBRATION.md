# Calibration — Hyper-Detailed, Step by Step

This is the procedure for taking a freshly installed system that counts
nothing and turning it into one that counts every box, every pad, and every
pack time correctly.

**You do not need a browser, a second computer, or to copy files anywhere.**
Everything is done from the Pi's terminal.

---

## Before you start

Read this box once — it explains the two ideas everything else rests on.

> **1. The system compares each frame against a picture of the empty belt.**
> Anything different is "foreground". Boxes are foreground; the belt is not.
> If the image wobbles or the light flickers, everything looks like
> foreground and nothing works.
>
> **2. Four geometry numbers decide everything**, all written as *fractions*
> of the picture (0.0 = left/top edge, 1.0 = right/bottom edge):
>
> | Setting | Meaning |
> |---|---|
> | `processing.roi` | the part of the image that is belt. Everything outside is ignored. |
> | `counting.line_position` | boxes are counted when they cross this line |
> | `counting.direction` | which way boxes travel across it |
> | `packing.zone` | where boxes stop to be filled |

What you need:

- the belt running,
- at least 10 boxes you can send through,
- someone to pack them the normal way (for pads and times),
- about 30 minutes.

Stop the service so it doesn't fight you for the camera:

```bash
sudo systemctl stop boxcounter
cd ~/box_count_rpi5
```

---

# STEP 1 — Find out what is actually wrong

Do this first, always. It runs the real detection chain and tells you the
exact stage where things break.

```bash
python3 tools/wizard.py --diagnose --seconds 45
```

**While it runs (45 seconds), send 2–3 boxes down the belt and pack one
normally.** It needs to see real activity.

You will get a funnel like this:

```
Stage-by-stage funnel
  frames captured                  1350  ##############################
  frames with ANY foreground        980  ##############################
  frames with blobs (any size)      980  ##############################
  frames with ACCEPTED boxes          0
  tracks created                      0
  tracks that crossed the line        0
  BOXES COUNTED                       0
```

**Read it top to bottom and find the first row that drops to zero.** That is
your problem. The tool then names it and tells you the setting to change:

```
Verdict
  PROBLEM Blobs are found but ALL are rejected by the size filter.
     Observed blob area (fraction of ROI): median=0.0001  largest(p98)=0.2228
     Configured accept range: 0.5 .. 0.6
     -> even your BIGGEST blob is smaller than min_area_frac.
        Set processing.min_area_frac: 0.0891
```

### What each verdict means

| Verdict | What is happening | What to do |
|---|---|---|
| **No frames from the camera** | camera not working | `rpicam-hello --list-cameras` — see [TROUBLESHOOTING](TROUBLESHOOTING.md) |
| **Sees NOTHING moving** | every frame looks like empty belt | nothing passed during the test, or the ROI points at the wrong place, or contrast is too low. Go to **Step 2** |
| **Foreground but no blobs** | the noise filter is erasing your boxes | set `processing.open_kernel: 3` |
| **All blobs rejected by size** | boxes are bigger/smaller than the accepted range | **Step 3** fixes this automatically |
| **None crossed the line** | the counting line is not in the boxes' path | **Step 3** fixes this automatically |
| **Crossed but not counted** | usually the direction is backwards | **Step 3** fixes this automatically |
| **Counting works** | box counting is good | go to **Step 5** |

Keep this command in mind — **you re-run it after every change.**

---

# STEP 2 — See what the camera sees, and set the ROI

This is the only step the wizard cannot do for you: only you know which part
of the picture is belt and which is machinery.

```bash
python3 tools/wizard.py --live --seconds 60
```

You get a live picture drawn with characters:

```
--------------------------------------------------------
|                                                      |
|              ....::::::::....                        |
|            .:@@@@@@@@@@@@@@:.                        |
|            :@@@@@@@@@@@@@@@@:                        |
|            -|@@@@@@@@@@@@@@|-                        |
|            .:@@@@@@@@@@@@@@:.                        |
|                                                      |
========================================================
|                                                      |
--------------------------------------------------------
legend: ' '=belt  .:#@=foreground  |-|=detected box
        ===counting line  :.:=packing zone  ---=ROI
```

- **space** = belt, nothing detected. This is what an empty belt should look
  like: almost entirely blank.
- **`.` `:` `#` `@`** = foreground. A box should be a solid `@` shape.
- **`|` `-` in green** = a box the system **accepted**. This is the goal.
- **`---` blue border** = the current ROI.
- **`===` amber** = the counting line.

### What you are checking

**A. Is the empty belt blank?**
If the screen is full of `.` and `:` with no boxes passing, the detector is
seeing noise or machinery.

- Noise everywhere → raise `processing.blur_kernel` to 7 and
  `processing.open_kernel` to 7.
- A busy region in one place (a roller, a person's walkway, a hanging cable)
  → exclude it with the ROI, below.

**B. Do boxes appear as solid `@` shapes?**
If a box appears as a hollow ring (an open box, where the inside looks like
the belt), raise `processing.close_kernel` from 31 to 41, then 51.

If boxes barely appear at all, lower `processing.mog2_var_threshold` from 32
to 20, then 16.

**C. Does the ROI (blue border) cover belt only?**

Read the position off the screen. The view is the whole camera image, so:

- the **left edge** of the picture is x = 0.0, the **right edge** is x = 1.0
- the **top** is y = 0.0, the **bottom** is y = 1.0
- something a quarter of the way across is x ≈ 0.25

Then edit the config:

```bash
nano config/config.yaml
```

Find the `processing:` section and set:

```yaml
  roi: [0.05, 0.0, 0.90, 1.0]
```

That reads: start 5% in from the left, 0% from the top, and cover 90% of the
width and 100% of the height. **Save with Ctrl-O, Enter, then exit with
Ctrl-X.**

Re-run `python3 tools/wizard.py --live` and check the blue border now sits on
belt only.

> **Rule:** it is better to make the ROI too small than too big. A box only
> needs to be *mostly* inside it.

---

# STEP 3 — Let the wizard learn your line

Now the wizard watches real boxes and works out the numbers itself.

```bash
python3 tools/wizard.py --geometry --seconds 60
```

**Run at least 3 complete cycles while it watches:** a box arrives, stops,
gets packed, and leaves. The more complete cycles, the better.

It reports what it saw and proposes settings:

```
What was observed
  boxes travelled x=0.38..0.50  y=0.07..0.94
  travel direction along y: positive
  box area p10=0.0602 p90=0.2176 (fraction of ROI)
  boxes stopped around x=0.45 y=0.33 (236 samples)

Proposed configuration
  processing.min_area_frac: 0.0301
  processing.max_area_frac: 0.544
  counting.direction: positive
  counting.line_position: 0.62
  packing.zone: [0.29, 0.2, 0.31, 0.38]

Write these into config/config.yaml? [y/N]
```

Type **`y`** and press Enter.

**Your old config is backed up automatically** to
`config/config.backup-YYYYMMDD-HHMMSS.yaml`, and comments are preserved. To
undo, copy the backup back over `config/config.yaml`.

### Things that can go wrong here

| Message | Meaning | Fix |
|---|---|---|
| "The biggest moving thing was only 0.05% of the region of interest — that is noise" | no box passed while it was measuring, or the ROI is off the belt | send boxes past **during** the test; check **Step 2** |
| "Only 8 box sightings — too little to calibrate from" | not enough evidence | run more boxes, or add `--seconds 90` |
| "current size limits reject every blob, so geometry was inferred from raw detections" | normal when starting from a bad config — it repairs detection, then automatically watches again | nothing, let it run |
| "no stopping point seen" | no box ever stopped, so no packing zone could be set | if boxes *do* stop, raise `packing.dwell_speed_px` to 3.0 and repeat |
| "The values worked out here do not form a valid configuration" | it refused to write something the counter could not load | run more boxes past and retry with `--seconds 90` |
| "Detection is still not accepting boxes after this change" | something more basic is wrong, almost always the ROI | go back to **Step 2** |

> **Your belt may run left-to-right** rather than top-to-bottom in the
> picture. The wizard detects this and sets `counting.axis` for you. If you
> ever see boxes tracked but never counted, `--diagnose` will tell you the
> axis is wrong.

**To undo anything the wizard wrote**, use the backup it prints:

```bash
cp config/config.backup-20260812-155737.yaml config/config.yaml
```

---

# STEP 4 — Confirm box counting works

```bash
python3 tools/wizard.py --diagnose --seconds 45
```

Send boxes through again. You want:

```
  BOXES COUNTED                      3  ###

Verdict
  OK    Counting works: 3 boxes counted in this test.

Packing station
  Boxes stopped around x=0.45 y=0.33 (fractions of the frame).
  OK    209/236 stopped samples are inside packing.zone.
```

**Count the boxes yourself and compare.** If the number is wrong:

| Problem | Fix |
|---|---|
| Two touching boxes counted as one | lower `processing.merge_gap_px` to 12 and `close_kernel` to 21. Boxes that physically touch cannot be separated |
| One box counted twice | raise `tracking.max_disappeared` to 45 |
| Boxes missed on a fast belt | raise `camera.fps` to 40; lower `tracking.min_hits` to 2 |
| Counts with an empty belt | raise `processing.min_area_frac` |

**Do not go further until box counting is right.** Pad counting is built on
top of box tracking.

---

# STEP 5 — Tune pad counting

This is the sensitive part, and the defaults will almost certainly need
adjusting for your station.

```bash
python3 tools/calibrate.py --packing --seconds 60
```

When a box parks, it saves a picture of the geometry and then shows a live
readout. **Reach into the box the way your packer does, several times.**

```
  time  state     signal bar           motion bar           pieces
   5.4  idle       0.000 ----|-------   0.000 ----|-------  0
   5.8  HAND IN    0.106 ####|##-----   0.284 ####|#######  1
   6.6  idle       0.000 ----|-------   0.000 ----|-------  1
```

- **signal** — how strongly an arm is detected. The `|` marks the threshold.
- **motion** — movement inside the box.
- **state** should read `HAND IN` for the whole reach and `idle` between.
- **pieces** should go up by exactly 1 per reach.

At the end it prints recommended values:

```
arm signal while idle : p50=0.000 p95=0.004   (must stay BELOW arm_enter_frac)
arm signal while HAND : p50=0.110 p95=0.224   (must stay ABOVE arm_exit_frac)
Suggested:  arm_enter_frac: 0.060   arm_exit_frac: 0.025
Suggested:  interior_motion_frac: 0.061
```

Put those into the `packing:` section of `config/config.yaml`.

| Problem | Fix |
|---|---|
| Reaches not detected | lower `arm_enter_frac`; raise `ring_px` to 40 |
| Quick reaches missed | lower `min_visit_frames` to 2 |
| One reach counted twice | raise `exit_frames` to 6 |
| Empty-handed adjustments counted | raise `interior_motion_frac`; or set `appearance_check: true` |
| "idle and hand signals overlap" | the arm is not clearly separable — improve lighting, or move the queued box further from the packing spot |

**If your packer places a fixed bundle per reach** (e.g. 3 pads in one
motion), the camera sees one reach. Set `pieces_per_visit: 3`.

---

# STEP 6 — Verify the whole thing

```bash
python3 -m boxcounter
```

Pack 10 boxes normally and watch the panel:

```
┌─ TOTALS ──────────────────────────────────────────┐
│ Boxes counted                                  10 │
│ Average pack time                          11.4 s │
│ Average pads per box                          3.0 │
└───────────────────────────────────────────────────┘
```

Compare against what you counted by hand. Check the per-box rows in
`RECENT BOXES` — a wrong pad count on one box points at Step 5; a wrong box
count points at Step 4.

Press **Ctrl-C** to stop.

---

# STEP 7 — Lock it in

```bash
cp config/config.yaml ~/config-working-$(date +%F).yaml   # keep a copy!
sudo systemctl start boxcounter
systemctl status boxcounter
```

Your calibration is the only thing here unique to your line. Keep that copy
somewhere safe.

---

## Re-calibrate when any of these change

- the camera or light is moved or bumped,
- box sizes change,
- belt speed changes,
- daylight starts reaching the belt at a different time of year.

Quick health check any time:

```bash
python3 tools/wizard.py --diagnose --seconds 30
```

## Command summary

| Command | Purpose |
|---|---|
| `python3 tools/wizard.py` | guided run through all steps |
| `python3 tools/wizard.py --diagnose` | **where is it broken?** |
| `python3 tools/wizard.py --live` | see what the camera sees |
| `python3 tools/wizard.py --geometry` | learn and write the geometry |
| `python3 tools/calibrate.py --packing` | tune pad detection |
| `python3 -m boxcounter --check` | verify install and config |
| `python3 -m boxcounter` | run it |

Full parameter documentation is inline in `config/config.yaml`. The packing
concept and its limits are in [PACKING.md](PACKING.md); day-to-day operation
is in [RUNNING.md](RUNNING.md).
