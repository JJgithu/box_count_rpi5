# box_count_rpi5

Fully **offline** box counting for a conveyor belt, running on a
**Raspberry Pi 5 (2 GB)** with an **IMX219 camera** mounted above the belt.
Counts boxes — open or closed — as they pass a virtual line, with no internet,
no GPU and no neural network required.

```
   IMX219 (top view, CSI)
        │ 640x480 @ 30 fps (full-FoV binned sensor mode)
        ▼
  ┌─────────────┐   ┌──────────────┐   ┌──────────┐   ┌───────────────┐
  │ background  │──▶│ blob detect  │──▶│ centroid │──▶│ line-crossing │
  │ subtraction │   │ + open-box   │   │ tracking │   │ counter with  │
  │ (color MOG2)│   │ fusion       │   │          │   │ hysteresis    │
  └─────────────┘   └──────────────┘   └──────────┘   └──────┬────────┘
                                                             ▼
                                    SQLite + CSV logs · GPIO pulse (PLC)
                                    LAN web dashboard with live video
```

## Why classical CV instead of a neural network?

On a 2 GB Pi 5 there is no NPU and no spare RAM for a detector model — and no
pretrained model knows what *your* boxes look like anyway. With a fixed
top-down camera and a moving belt, adaptive background subtraction is the
textbook solution: it runs at the full 30 fps using ~15 % of one core and
under 150 MB of RAM, works for any box size/color without training data, and
handles open boxes via morphological fusion of the rim fragments. The
detector is a single class (`boxcounter/detector.py`) — if you ever outgrow
it, you can swap in a TFLite/NCNN model without touching tracking or counting.

## Key robustness features

- **Open boxes**: an open box's interior can look like the belt; a large
  morphological close + blob merging fuses the rim into one detection
  (covered by tests).
- **Color subtraction**: cardboard on a gray belt often has almost no
  *brightness* contrast; subtraction runs in color so chroma differences count.
- **No background absorption**: the model only learns while the belt is
  empty, so steady box traffic is never "learned away" (a failure mode our
  end-to-end tests reproduce and guard against).
- **Hysteresis counting**: a box is counted exactly once, only when it crosses
  the line in the travel direction — vibration and bbox jitter can't double-count.
- **Locked exposure**: auto-exposure is frozen after startup so the image
  stays stable for subtraction.
- **Restart-safe**: totals persist in SQLite; the systemd service auto-restarts.

## Quick start (on the Pi)

```bash
# Raspberry Pi OS Lite has no git preinstalled, and this repo is private:
sudo apt-get update && sudo apt-get install -y git
# then authenticate (Personal Access Token / SSH key / gh) — or just copy the
# repo over with scp. See docs/IMPLEMENTATION_GUIDE.md section 4 for all paths.
git clone https://github.com/JJgithu/box_count_rpi5.git
cd box_count_rpi5
bash scripts/install.sh                      # apt packages + systemd unit
python3 tools/calibrate.py                   # snapshot with grid/ROI/line overlay
python3 -m boxcounter                        # run; dashboard at http://<pi>:8080
sudo systemctl enable --now boxcounter       # run at every boot
```

> Private repo + offline target: the cleanest route is to **not clone on the
> Pi at all** — copy the code across with `scp`/USB from a machine that
> already has it. Full details (git install, tokens, SSH keys, air-gapped
> apt) are in [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) §4.

## Try it without any hardware

```bash
pip install -r requirements.txt
python3 tools/make_test_video.py --out data/test.mp4 --boxes 12
python3 -m boxcounter --source data/test.mp4 --no-web
python3 -m pytest tests/                     # 38 tests incl. exact-count e2e
```

## Documentation

| Document | Contents |
|---|---|
| [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | **Start here** — complete step-by-step deployment walkthrough |
| [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) | Mounting height/FoV math, lighting, wiring, GPIO to PLC |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | Every tunable parameter, with symptom → fix tables |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems and their causes |

## Repository layout

```
boxcounter/          the application package
  camera.py            Picamera2 / video / USB frame sources
  detector.py          color background subtraction + open-box fusion
  tracker.py           predictive centroid tracker
  counter.py           directional line counter with hysteresis
  pipeline.py          main loop wiring it all together
  storage.py           SQLite + daily CSV persistence
  gpio_out.py          one pulse per box (PLC / stack light)
  webui.py             offline LAN dashboard with live MJPEG view
config/config.yaml   all tunables, heavily commented
tools/               calibrate, capture background, synthetic video, benchmark
tests/               unit + exact-count end-to-end tests
scripts/, systemd/   installer and boot service
```
