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
and shows a live panel that refreshes in place:

```
 BOX COUNTER                                          15:31:19   30 fps

┌─ TOTALS ─────────────────────────────────────────────────────────────┐
│ Boxes counted                                                     42 │
│ Average pack time                                             11.4 s │
│ Average pads per box                                             3.0 │
│ Pads counted (total)                                             126 │
│ Boxes per minute                                                 5.2 │
└──────────────────────────────────────────────────────────────────────┘

┌─ LAST BOX ───────────────────────────────────────────────────────────┐
│ Box #42   at 15:31:19                                3 pads   10.9 s │
└──────────────────────────────────────────────────────────────────────┘

┌─ RECENT BOXES ───────────────────────────────────────────────────────┐
│   box   pads   pack time      time                                   │
│    42      3      10.9 s  15:31:19                                   │
│    41      4      14.2 s  15:30:59                                   │
│    40      2       9.8 s  15:30:39                                   │
└──────────────────────────────────────────────────────────────────────┘

 packing now: 2 pads, 8s  — hand in box

 CSV: /home/pi/box_count_rpi5/data/events_2026-08-12.csv
 Ctrl-C to stop
```

Totals and averages cover **every box since the last reset** and are read
from the database, so they survive restarts.

The panel appears only on a real terminal. Under systemd the same
information goes to the journal as log lines (a redrawing panel would be
unreadable there). To force log lines on a terminal — for debugging, or to
pipe output to a file:

```bash
python3 -m boxcounter --no-status
python3 -m boxcounter --log-level DEBUG     # implies --no-status
```

In log mode you get one `BOX #n ... — 3 pads, packed in 11.4 s` line per box
plus a `heartbeat` every 30 s. **If heartbeats stop, something is wrong.**

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

### The CSV

**The counter writes a CSV automatically — one row per counted box** — to:

```
~/box_count_rpi5/data/events_YYYY-MM-DD.csv
```

A new file per day, created on the first box of that day. Columns:

```
box,timestamp,pads,pack_seconds,track_id,x,y,w,h,area,direction
1,2026-08-12T15:31:00,3,4.67,1,240,326,157,128,20096,1
2,2026-08-12T15:31:02,2,3.58,5,164,323,169,134,22646,1
3,2026-08-12T15:31:04,4,5.71,8,210,325,152,129,19608,1
```

The first four columns are what you asked for — box number, when it was
counted, how many pads went in, how long the pack took. The rest is
detection detail kept for diagnostics; ignore or delete those columns.

Read it on the Pi, or copy it to a PC:

```bash
column -s, -t data/events_$(date +%F).csv       # on the Pi, readable
# from your PC:
scp <pi-user>@boxcounter.local:box_count_rpi5/data/events_*.csv .
```

### One file for a whole week or shift

The daily files are convenient for rotation but awkward for reporting, so
there's an exporter that pulls any date range out of the database into a
single clean CSV:

```bash
python3 tools/export_csv.py -o report.csv                     # everything
python3 tools/export_csv.py --today -o today.csv
python3 tools/export_csv.py --since 2026-08-01 --until 2026-08-08 -o week.csv
python3 tools/export_csv.py --summary                         # just the numbers
```

`--summary` prints the headline figures:

```
Boxes counted        : 412
Pads counted (total) : 1236
Average pads per box : 3.0
Average pack time    : 11.4 s
Fastest / slowest    : 8.2 s / 26.7 s
```

### Everything else

| Where | How |
|---|---|
| Dashboard | `http://<pi-ip>:8080` |
| Just the numbers | `curl -s http://<pi-ip>:8080/api/stats` |
| Reset to zero | click **Reset count** on the dashboard, or `curl -X POST http://<pi-ip>:8080/api/reset` |
| Database | `sqlite3 data/boxcount.db "SELECT COUNT(*) FROM events;"` |
| Counts per hour | `sqlite3 data/boxcount.db "SELECT substr(iso,1,13) h, COUNT(*) FROM events GROUP BY h;"` |
| Pads + pack time per box | `sqlite3 data/boxcount.db "SELECT iso, pieces, pack_seconds FROM events ORDER BY id DESC LIMIT 20;"` |

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
