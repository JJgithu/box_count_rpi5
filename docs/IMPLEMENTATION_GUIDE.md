# Implementation Guide — Conveyor Box Counter on Raspberry Pi 5

This is the complete, ordered walkthrough from empty SD card to a box counter
running unattended at every boot. Allow ~2 hours for first deployment
(most of it mounting and calibration).

---

## 1. What you need

| Item | Notes |
|---|---|
| Raspberry Pi 5 (2 GB) | 27 W USB-C PSU recommended (official one) |
| IMX219 camera module | e.g. Camera Module 2. **Pi 5 needs a 22-pin to 15-pin CSI cable** (the Pi 5 has mini FPC connectors — the classic 15-pin cable does not fit) |
| microSD card, 16 GB+ | A1/A2 class card; the app writes very little |
| Camera mount | rigid mount above the belt — vibration is your enemy |
| Optional: LED light | constant (non-PWM/flicker-free) light if ambient light varies |
| Optional: PLC/counter input | one GPIO pulse per box is provided |

No internet is required on the Pi after installation. If the Pi has no
network at all, you can pre-download the apt packages on another machine, or
prepare the SD card while temporarily connected to a phone hotspot — after
`scripts/install.sh` completes, the system never needs a connection again.

## 2. Physical installation

Full details with the math in [HARDWARE_SETUP.md](HARDWARE_SETUP.md). Summary:

1. Mount the camera **directly above the belt centerline, looking straight
   down**. The IMX219's field of view is 62.2° × 48.8°, so at height *H* it
   sees roughly **1.21 H across × 0.91 H along the belt**. Pick *H* so the
   belt width fills most of the image: e.g. a 60 cm belt is nicely covered
   from ~55–65 cm above.
2. Orient the camera so the belt travels **vertically in the image** (along
   the image's long or short axis — either works; the config's `counting.axis`
   matches it). Default config assumes travel top→bottom (`axis: y`,
   `direction: positive`).
3. Make the mount rigid. Route the CSI ribbon away from motors/VFDs.
4. Lighting: constant and diffuse. Avoid heavy direct sunlight patterns
   moving across the belt during the day (or use the `static` method + a
   shade). Shadow rejection is built in but good lighting always wins.

## 3. Operating system

1. Flash **Raspberry Pi OS Lite (64-bit), Bookworm or newer** with Raspberry
   Pi Imager. Lite is preferred on 2 GB — no desktop saves ~400 MB RAM.
   In the Imager's settings, set hostname (`boxcounter`), user, and enable SSH.
2. Boot, SSH in, and verify the camera is detected:

   ```bash
   rpicam-hello --list-cameras       # should list imx219
   ```

   If not detected, re-seat the ribbon (contacts facing the correct side at
   both ends) — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## 4. Get the code onto the Pi

Raspberry Pi OS **Lite does not ship with git**, so install it first no
matter how you get the code. If the repo is **public**, a plain clone then
works; if it is **private**, you must authenticate first (a bare clone of a
private repo fails with "repository not found"). Pick whichever path fits.

### Path A — clone on the Pi (Pi has internet during setup)

Install git:

```bash
sudo apt-get update
sudo apt-get install -y git
```

**If the repo is public**, just clone it — no credentials needed:

```bash
git clone https://github.com/JJgithu/box_count_rpi5.git
```

**If the repo is private**, authenticate with **one** of A1–A3 below.

**A1. HTTPS + Personal Access Token (simplest).**
Create a token on GitHub: *Settings → Developer settings → Personal access
tokens → Fine-grained tokens*, scope it to the `box_count_rpi5` repo with
**Contents: Read-only**, then:

```bash
git clone https://github.com/JJgithu/box_count_rpi5.git
# Username: your GitHub username (JJgithu)
# Password: paste the token (NOT your GitHub password)
```

To avoid re-typing it on future `git pull`s, cache it:
`git config --global credential.helper store` (writes the token to
`~/.git-credentials` in plain text — fine on a dedicated appliance).

**A2. SSH key** (nicer for long-lived machines):

```bash
ssh-keygen -t ed25519 -C "pi-boxcounter"     # press Enter through the prompts
cat ~/.ssh/id_ed25519.pub                     # add this to GitHub: Settings
                                              #  -> SSH and GPG keys -> New key
git clone git@github.com:JJgithu/box_count_rpi5.git
```

**A3. GitHub CLI:** `sudo apt-get install -y gh && gh auth login` (choose
HTTPS, authenticate in a browser/device code), then clone as in A1 without a
manual token.

### Path B — transfer the code (air-gapped Pi, recommended for a true offline box)

Since the counter is meant to run offline, the Pi never actually needs GitHub
access. On a computer that already has the repo, copy it over — no git,
tokens, or keys on the Pi at all:

```bash
# On your PC (has network + the repo). Either clone or download the ZIP from
# GitHub ("Code -> Download ZIP"), then:
scp -r box_count_rpi5 <pi-user>@boxcounter.local:~/     # or use a USB stick
```

(You can also `git clone` on the PC and `rsync -a box_count_rpi5/ pi@boxcounter:~/box_count_rpi5/`.)

## 4b. Install dependencies

```bash
cd box_count_rpi5
bash scripts/install.sh
```

The script installs everything from the Raspberry Pi OS repositories
(`python3-picamera2`, `python3-opencv`, `python3-numpy`, `python3-yaml`,
`python3-flask`, `python3-gpiozero`, `python3-lgpio`) — no pip, no
compilation — and installs (but does not yet enable) the systemd service.

> **`install.sh` needs apt access** (network, or a local mirror). For a
> fully air-gapped Pi, pre-download the `.deb`s on a networked Pi of the same
> OS version with
> `apt-get install --download-only -y python3-picamera2 python3-opencv python3-numpy python3-yaml python3-flask python3-gpiozero python3-lgpio rpicam-apps`,
> copy `/var/cache/apt/archives/*.deb` to the target, and
> `sudo dpkg -i *.deb`. After this one-time install the system runs with no
> network forever.

## 5. Calibration

All tuning lives in one file: **`config/config.yaml`** (heavily commented).
Details in [CALIBRATION.md](CALIBRATION.md). The short version:

1. **Take an annotated snapshot** (belt running, a box or two in view):

   ```bash
   python3 tools/calibrate.py
   ```

   Copy `data/calibration_snapshot.jpg` to your PC (`scp`) and look at it:
   a 10 % grid is drawn with fraction labels, plus the current ROI (blue)
   and counting line (yellow).

2. **Set the ROI** — `processing.roi: [x, y, w, h]` as fractions — so it
   covers *only belt surface*: no rails, no rollers, no walkway where people
   pass. Re-run the snapshot until it looks right.

3. **Set the counting line** — `counting.line_position` — around the middle
   of the box path, well away from where boxes enter/exit the image.
   `counting.axis: y` + `direction: positive` = boxes travel downward in the
   image; flip `direction` to `negative` if they travel upward (or use
   `camera.vflip: true` to rotate the world instead).

4. **Check blob sizes** with boxes flowing:

   ```bash
   python3 tools/calibrate.py --check --seconds 30
   ```

   It prints detected blob areas and suggests `min_area_frac` /
   `max_area_frac` values.

5. **Watch it live.** Start the counter (`python3 -m boxcounter`) and open
   `http://<pi-ip>:8080` from any browser on the same network. The dashboard
   shows the live count, rate, and the camera view with tracked boxes; the
   "show mask" toggle displays exactly what the detector sees — boxes should
   be solid white blobs, the empty belt black.

### The two detection methods

- `method: mog2` (default): adapts to gradual lighting change. The model
  learns **only while the belt is empty** (`freeze_learning_fg_fraction`),
  so continuous box traffic is never absorbed into the background.
- `method: static`: compares against a fixed photo of the empty belt —
  immune to absorption by design, ideal for uniform belts under constant
  (artificial) light. Capture the reference with the belt **running and
  empty**:

  ```bash
  python3 tools/capture_background.py
  # then set processing.method: static in config/config.yaml
  ```

### Open boxes

Open boxes are detected out of the box: their interior may resemble the belt,
fragmenting the raw mask into a rim, but the morphological close
(`close_kernel: 31`) plus blob merging (`merge_gap_px: 24`) fuse the pieces.
Rule of thumb: `close_kernel` ≈ ¼ of a box's width **in pixels**. If open
boxes show up as two tracked halves in the live view, raise these two values.

### Packing station (pieces per box + pack time)

If a packer fills the boxes under this camera, enable the packing monitor —
it counts the pieces placed into each box and times each pack from the same
camera. Set the `packing:` section of the config (`enabled: true`, and
`zone` to where boxes stop; the shipped config has it on). Layout, concept
and tuning live in [PACKING.md](PACKING.md); verify on your bench with
`tools/make_packing_video.py` before deploying.

## 6. Verify counting accuracy

Let a known batch through — e.g. 20 boxes, mixed open/closed, at production
speed and spacing — and compare with the dashboard total. Repeat after any
lighting or speed change. For a data-driven check, run
`python3 tools/benchmark.py` to confirm the processing rate comfortably
exceeds the camera frame rate (expect < 10 ms/frame total on a Pi 5 at
640×480 — well over 60 fps of headroom).

## 7. Run at boot

```bash
sudo systemctl enable --now boxcounter
systemctl status boxcounter          # should be active (running)
journalctl -u boxcounter -f          # live logs incl. every count + heartbeat
```

The service auto-restarts on failure, restarts the camera if it stops
delivering frames, resumes the running total from SQLite after reboot, and is
capped at 512 MB RAM (`MemoryMax`) so it can never destabilize the 2 GB system.

## 8. Getting the counts out (all offline)

| Channel | What you get | Where |
|---|---|---|
| Web dashboard | live total, boxes/min, fps, recent events, video | `http://<pi>:8080/` |
| JSON API | `{"total": N, "rate_per_min": ..}` for other systems on the LAN | `GET /api/stats`, `POST /api/reset` |
| SQLite | every event with timestamp and box size | `data/boxcount.db` |
| CSV | one file per day, spreadsheet-ready | `data/events_YYYY-MM-DD.csv` |
| GPIO pulse | 50 ms pulse per box for a PLC/totalizer input | set `gpio.enabled: true`, default BCM 17 (physical pin 11) |
| Logs | one line per count + periodic heartbeat | `journalctl -u boxcounter` |

GPIO is 3.3 V — never wire it directly to a 24 V PLC input; use an
optocoupler or relay module ([HARDWARE_SETUP.md](HARDWARE_SETUP.md) §GPIO).

## 9. How it works (for the maintainer)

Each frame from the camera goes through four stages (~6 ms total):

1. **Detect** (`detector.py`) — the frame (color, ROI-cropped, blurred) is
   compared against a background model of the empty belt (MOG2 mixture model
   or a static photo). Shadows are classified and discarded. Morphological
   open removes noise specks; a large close fuses open-box rims; contours
   become bounding boxes; near-adjacent boxes merge; size filters drop
   too-small/too-big blobs.
2. **Track** (`tracker.py`) — each detection is matched to existing tracks by
   predicted position (constant-velocity), keeping stable IDs. Unseen tracks
   coast a few frames so a momentary detection dropout doesn't lose the box.
3. **Count** (`counter.py`) — a track is counted once, when it moves from
   confidently before the line to confidently past it (hysteresis band), in
   the configured direction, after a minimum travel distance. Jitter,
   vibration, and boxes that enter already past the line can't create counts.
4. **Record** (`pipeline.py`) — event goes to SQLite + CSV, GPIO pulses,
   dashboard state updates, log line written.

Design decisions and their reasons live as comments in the code and in the
config file itself.

## 10. Performance & memory on the 2 GB Pi 5

Measured expectations at 640×480 @ 30 fps, color MOG2:

| Stage | per frame |
|---|---|
| capture (zero-copy from libcamera) | ~1 ms |
| detection (subtraction + morphology + contours) | 3–6 ms |
| tracking + counting | < 0.5 ms |
| **total** | **< 8 ms → 30 fps with 4× headroom** |

RSS is typically 120–180 MB. The web dashboard encodes JPEG only while
someone is actually watching. If you ever need more margin (e.g. very fast
belts needing 50+ fps), drop `camera.width/height` to 480×360 and scale the
pixel-unit parameters (`close_kernel`, `merge_gap_px`) proportionally.

## 11. Maintenance

- **Lens**: wipe periodically; dust looks like noise blobs (the `open_kernel`
  removes small specks, but keep it clean).
- **Data**: SQLite grows ~100 bytes/box — decades of headroom; prune old
  daily CSVs if you like (`find data -name 'events_*.csv' -mtime +90 -delete`).
- **Config changes**: edit `config/config.yaml`, then
  `sudo systemctl restart boxcounter`.
- **Watch remotely**: the dashboard and JSON API are all another machine on
  the LAN needs; nothing ever leaves the network.
