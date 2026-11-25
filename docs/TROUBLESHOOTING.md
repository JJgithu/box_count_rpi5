# Troubleshooting

## Packing station mode (`counting.mode: station`)

First look at the two lines under the panel (`now:` and `last:`) and at
`tail -n 30 data/boxcounter.log`: they say what the counter sees and why it
decided what it decided. The full guide is [STATION.md](STATION.md).

**Hands are counted as boxes / boxes are not counted at all**
- `python3 -m boxcounter --check` must say `counting : station`. If it says
  `conveyor`, the config is an old one: see STATION.md §1 (update).
- Run `python3 tools/station_setup.py` so the counter knows your box size and
  where the packing spot is.

**`now:` stays on `waiting for a box` with a box on the spot**
- After a second it adds the reason in brackets, and the log gets a line
  `Something at the spot is not taken as a box: …`:
  - `too small to be a box (N% of the picture, boxes are at least M%)` —
    lower `station.min_box_frac` below N%, or run the setup with this box.
  - `too big to be a box` — raise `station.max_box_frac`, or run the setup.
  - `not cardboard-coloured` — white or printed boxes, or unusual lighting:
    set `station.box_color: any`.
  - `it reaches in from the picture edge like an arm` — only a narrow part
    of the box is in the picture. Raise or turn the camera so the whole box
    with its flaps is in view (the setup tool reports this).
- No reason at all: the box stands outside `station.zone`, or the camera has
  moved since the setup. Run `python3 tools/station_setup.py` again.

**Pads too low or too high**
- `python3 -m boxcounter --check` says how pads are counted: `at the
  machine` (setup step 3 done) or `per reach into the box`. See the tuning
  table in STATION.md §5 for each. The log has a line for every pad:
  `Outlet: a pad was taken (…)`, or `Box 4 reach 2: 23% of the box changed
  (pad needs 6%) -> +1 pad`.

**Picture shows moving dark bands (lights)**
- Set `camera.flicker_hz` to your mains frequency (60 in the US, 50 in most
  other countries) and restart.

**Two boxes counted as one**
- The next box was pushed onto the spot before the previous one had left it.
  Leave a short gap (about half a second of empty spot) between boxes.

## Camera

**`rpicam-hello --list-cameras` shows nothing**
- CSI ribbon backwards or not fully seated (Pi 5 end: contacts face inward;
  push the latch down firmly).
- Wrong cable: Pi 5 needs the 22-pin mini-FPC cable, not the classic 15-pin.
- Try the other camera connector (CAM0 vs CAM1).
- `dmesg | grep -i imx219` for driver messages.

**`Picamera2` import errors**
- Install via apt (`python3-picamera2`), never pip, and run the app with the
  system `/usr/bin/python3` (the systemd unit already does). If you created a
  venv, it must use `--system-site-packages`.

**Image is black / far too dark**
- Something (enclosure lid?) blocking the lens during startup got the
  exposure locked wrong: restart the service with the scene in its normal
  state, or set manual `exposure_time_us`.

**Image is soft/blurry**
- Motion blur: shorter manual exposure (`exposure_time_us: 4000`, raise
  `analogue_gain` to compensate).
- Focus: the M12/stock lens can be rotated to refocus after freeing the glue.

## Service

**`systemctl status boxcounter` shows restart loop**
- `journalctl -u boxcounter -n 50` tells you why. Usual suspects: config
  typo (YAML error message points at it), camera not detected, port 8080
  already in use (`web.port`).

**Web dashboard unreachable**
- Confirm service is running and listening: `ss -tlnp | grep 8080`.
- You're on the same LAN / VLAN? The Pi never exposes anything beyond it.
- `web.enabled: true` in the config.

**Counts lost after reboot**
- Totals live in `data/boxcount.db`; make sure `data_dir` is writable by the
  service user and `output.sqlite: true`. The journal shows
  `Pipeline running (resumed total: N)` at startup.

## Counting quality

See the symptom tables in [CALIBRATION.md](CALIBRATION.md) — mask view first,
always.

**Everything was fine, now it's not (weeks later)**
- Dusty lens (wipe it), moved light, new belt, camera bracket rotated.
  Re-run `tools/calibrate.py` and compare the snapshot with the old one.

**High CPU / low fps**
- `vcgencmd get_throttled` → non-zero means power or heat problems.
- Someone raised the resolution? 640×480 is the intended operating point.
- Several dashboard viewers streaming MJPEG: each costs a JPEG encode; use
  the JSON API for machine consumers instead.

## GPIO

**No pulses at the PLC**
- `gpio.enabled: true`?
- Verify with a multimeter/LED+resistor on BCM 17 (physical pin 11) vs GND.
- Pulse too short for the PLC input filter → raise `gpio.pulse_ms`.
- Remember: 3.3 V logic; 24 V inputs need an optocoupler board
  ([HARDWARE_SETUP.md](HARDWARE_SETUP.md)).

## Recovering from a corrupted SD card

Flash a new card, reinstall (§3–4 of the implementation guide), restore
`config/config.yaml` and optionally `data/` from backup. Consider
`rsync -a pi@boxcounter:box_count_rpi5/data/ ./backup/` as a periodic
LAN-side cron job — still fully offline.
