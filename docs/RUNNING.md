# Running the Box Counter

Everything here assumes you are in the project directory on the Pi:

```bash
cd ~/box_count_rpi5
```

There are two ways to run it: **foreground** (you watch it, Ctrl-C stops it —
use this while setting up) and **as a service** (starts at boot, restarts
itself, survives you logging out — use this in production).

---

## 1. Foreground: run it now and watch

```bash
python3 -m boxcounter
```

That's the whole command. It reads `config/config.yaml`, opens the camera,
and starts counting. You'll see:

```
15:04:21 INFO boxcounter.camera: Picamera2 started: 640x480 @ 30 fps
15:04:24 INFO boxcounter.camera: Locked exposure: 8993 us, gain 1.80
15:04:24 INFO boxcounter.detector: Detector initialized: frame 640x480, ROI (32, 0, 576, 480)
15:04:24 INFO boxcounter.webui: Web dashboard on http://0.0.0.0:8080/
15:04:24 INFO boxcounter.pipeline: Pipeline running (resumed total: 0)
15:04:31 INFO boxcounter.pipeline: BOX #1 (track 3, 142x118 px)
15:04:33 INFO boxcounter.pipeline: BOX #2 (track 4, 138x121 px)
15:04:54 INFO boxcounter.pipeline: heartbeat: total=2 rate=5.7/min fps=30.0 tracks=1
```

- **`BOX #n`** — one line every time a box is counted.
- **`heartbeat`** — a status line every 30 s, even when nothing passes. If
  heartbeats stop, something is wrong.

**Stop it with `Ctrl-C`.** It shuts down cleanly and the total is saved.

### Watch the live view

While it runs, open a browser on any machine on the same network:

```
http://<pi-ip>:8080
```

Find the Pi's address with `hostname -I`. The dashboard shows the running
total, boxes/min, camera fps, recent events, and the live camera view. The
**"show mask"** link toggles to the detector's view — this is the single most
useful screen for checking your setup: **boxes should be solid white blobs on
a black belt.**

### Useful foreground options

| Command | What it does |
|---|---|
| `python3 -m boxcounter` | normal run |
| `python3 -m boxcounter --no-web` | no dashboard (slightly lighter) |
| `python3 -m boxcounter --log-level DEBUG` | verbose: logs every counting decision |
| `python3 -m boxcounter -c /path/to/other.yaml` | use a different config file |
| `python3 -m boxcounter --source data/test.mp4` | run on a video file instead of the camera |
| `python3 -m boxcounter --max-frames 300` | process 300 frames then stop |
| `python3 -m boxcounter --display` | OpenCV preview windows (needs a desktop, not Lite) |

`bash scripts/run.sh` is a shortcut for the normal run and passes through any
of these flags.

---

## 2. Service: run it forever, starting at boot

Set it up once:

```bash
sudo systemctl enable --now boxcounter
```

`enable` = start at every boot. `--now` = also start it right now. That's it —
it will keep running across reboots, crashes, and camera glitches.

### Day-to-day service commands

```bash
sudo systemctl status boxcounter     # is it running? shows recent log lines
sudo systemctl restart boxcounter    # apply a config change
sudo systemctl stop boxcounter       # stop for now (comes back after reboot)
sudo systemctl start boxcounter      # start again
sudo systemctl disable --now boxcounter   # stop AND don't start at boot
```

A healthy `status` looks like:

```
● boxcounter.service - Conveyor box counter (Pi 5 + IMX219)
     Loaded: loaded (/etc/systemd/system/boxcounter.service; enabled)
     Active: active (running) since Mon 2026-08-05 09:12:03 UTC; 2h 14min ago
```

`Active: active (running)` and an uptime that keeps growing is what you want.
If uptime keeps resetting to a few seconds, it's crash-looping — see the logs.

### Watching the logs

```bash
journalctl -u boxcounter -f            # live tail (Ctrl-C to exit)
journalctl -u boxcounter -n 100        # last 100 lines
journalctl -u boxcounter --since today
journalctl -u boxcounter | grep BOX    # just the counts
```

### After changing config.yaml

The service reads the config **only at startup**, so:

```bash
nano config/config.yaml
sudo systemctl restart boxcounter
```

> **Don't run both at once.** If the service is running and you also start
> `python3 -m boxcounter` by hand, the second one fails — the camera and port
> 8080 are already taken. Run `sudo systemctl stop boxcounter` first when you
> want to experiment in the foreground.

---

## 3. Reading the counts

| Where | How |
|---|---|
| Dashboard | `http://<pi-ip>:8080` |
| Just the number | `curl -s http://<pi-ip>:8080/api/stats` |
| Reset to zero | click **Reset count** on the dashboard, or `curl -X POST http://<pi-ip>:8080/api/reset` |
| Today's CSV | `column -s, -t data/events_$(date +%F).csv` |
| Database | `sqlite3 data/boxcount.db "SELECT COUNT(*) FROM events;"` |
| Counts per hour | `sqlite3 data/boxcount.db "SELECT substr(iso,1,13) h, COUNT(*) FROM events GROUP BY h;"` |

The running total **survives restarts and reboots** — it resumes from the
database. "Reset count" starts a new tally without deleting history; the old
events stay in the database and CSVs.

To copy the data to a PC (run this *on the PC*):

```bash
scp -r <pi-user>@boxcounter.local:box_count_rpi5/data ./boxcount-data
```

---

## 4. Trying it without the camera

Useful for a demo, or to sanity-check the software before the hardware is
mounted. This works on the Pi or on any PC with the requirements installed.

```bash
python3 tools/make_test_video.py --out data/test.mp4 --boxes 12
python3 -m boxcounter --source data/test.mp4 --no-web
```

Expected last line: `frames=... counted_total=12 session=12 fps=...` —
12 boxes generated, 12 counted.

Run the automated tests the same way:

```bash
python3 -m pytest tests/
```

---

## 5. Is it working correctly?

Run a known batch — say 20 boxes at production speed and spacing — and compare
with the dashboard. If the number is off:

1. **Open the mask view** (dashboard → "show mask"). Boxes should be solid
   white, the belt black. Almost every miscount is visible here.
2. Look up the symptom in [CALIBRATION.md](CALIBRATION.md) — it has
   symptom → fix tables for both the mask and the count.
3. Check the processing headroom: `python3 tools/benchmark.py --frames 300`.
   On a Pi 5 at 640×480 expect detection under 6 ms/frame.

Quick checks when something seems off:

```bash
systemctl is-active boxcounter        # active?
journalctl -u boxcounter -n 30        # what did it last say?
vcgencmd get_throttled                # 0x0 = power/heat OK
rpicam-hello --list-cameras           # camera still detected? (stop the service first)
```

---

## 6. Quick reference

```bash
# run now, watch it
python3 -m boxcounter

# run forever, from boot
sudo systemctl enable --now boxcounter

# is it alive?
systemctl status boxcounter
journalctl -u boxcounter -f

# change settings
nano config/config.yaml && sudo systemctl restart boxcounter

# read the count
curl -s http://localhost:8080/api/stats
```
