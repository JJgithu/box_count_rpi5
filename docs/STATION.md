# Packing station mode — boxes, pads and pack time at the paper machine

This is the default mode (`counting.mode: station` in `config/config.yaml`).
It is made for a packing station like this one: a roller conveyor in front of
a paper-pad machine (Storopack PAPERplus), the camera on an arm above the
machine looking down at the spot where the box is filled.

```
        paper machine — the pad hangs from the outlet
   ┌──────────────────────────────────────────────┐
   │                 [ outlet ]                   │
   └──────────────────────╥───────────────────────┘
   ║║║║║║║║║║║║║║║║┌──────╨─────┐║║║║║║║║║║║║║║║║    rollers
   ║║║║║ box ──▶ ║║│ the box is ││║║║ ──▶ box ║║║║    (either way)
   ║║║║║║║║║║║║║║║║│   packed   │║║║║║║║║║║║║║║║║
   ║║║║║║║║║║║║║║║║└────────────┘║║║║║║║║║║║║║║║║
                         ▲
                      packer
```

## What it counts, and how

| | How the camera decides |
|---|---|
| **A box** | Something that is not the empty rollers, **stands still**, is **box-sized**, looks like **cardboard**, and sits at the packing spot. Counted when it **leaves** the spot, either way along the rollers or lifted off. |
| **Pads in that box** | Every **reach**: a hand moving over the box, then the box going quiet again. If enough of the inside of the box looks different afterwards, a pad went in. |
| **Pack time** | From the moment the box was set down on the spot to the moment it left. |

Why hands are not counted as boxes any more: a hand or arm never stays
perfectly still for long, it is smaller than a box, and it reaches in from
the edge of the picture. The old counting-line method counted anything that
crossed a line, and a hand pulling a pad towards the packer does exactly
that. There is no counting line in this mode.

Things that do **not** count as a pad:
- a reach that changes nothing in the box (patting the paper down slightly),
- the machine pushing out the next pad (that paper belongs to the picture of
  the empty spot, not to the box),
- shoving the box along the rollers a bit (it is followed to its new place).

---

## 1. Update the Pi to this version

On the Pi:

```bash
cd ~/box_count_rpi5
cp config/config.yaml ~/config-before-station.yaml   # keep your old settings
git checkout -- config/config.yaml                   # undo local edits so the
git pull                                             #   update can come in
python3 -m boxcounter --check
```

`--check` must say `counting : station`. (The `git checkout` line is needed
because the old calibration wizard edited `config/config.yaml` on the Pi;
those settings were for the counting-line method and do not apply here.)

If the counter runs as a service, stop it while setting up — only one program
can use the camera at a time:

```bash
sudo systemctl stop boxcounter
```

## 2. Set up — two pictures, two key presses

```bash
python3 tools/station_setup.py
```

1. It asks you to **clear the spot**: no box on the rollers in front of the
   machine, hands away. Leave a pad hanging from the machine if one normally
   hangs there. Press Enter.
2. It asks you to put an **empty box exactly where it stands while it is
   packed**, flaps open as usual, then step back. Press Enter.

It then draws what it found in the terminal (the box as `@#`, the packing
zone as `:.:`) and tells you:

- how much of the picture the box fills (good: roughly 10–60 %),
- whether the whole box is in view (if it touches an edge, turn or raise the
  camera if you can),
- whether the box looks cardboard-coloured to the camera.

Answer **Y** to save. A backup of the previous config is kept next to it
(`config/config.backup-<date>.yaml`). The pictures are saved in `data/`
(`setup_empty.jpg`, `setup_box.jpg`, `setup_result.jpg`).

Run the setup again whenever the camera is moved or you switch to a very
different box size. If you pack several box sizes, set up with the
**smallest**: anything from about half its size up to 2.5× it is accepted.

## 3. Run it and watch

```bash
python3 -m boxcounter
```

The panel shows the totals, the last box and the recent boxes. Under them
are two lines that tell you what the camera is doing **right now**:

```
 now: box at the station: 2 pads, 14 s  (hand in box)
 last: reach: 23% of the box changed -> +1 pad
```

| `now:` says | Meaning |
|---|---|
| `learning the empty packing spot — keep it clear` | First second after start: it is taking the picture of the empty spot. |
| `waiting for a box` | Spot empty. |
| `something stopped at the spot — checking (0.4 s)` | A box arrived; it is making sure (well under a second). |
| `box at the station: 2 pads, 14 s` | Packing. `(hand in box)` during a reach. |
| `ignoring an object at the spot: …` | Something stopped there that is not counted, and why. |

`last:` says why the last reach did or did not count as a pad, for example
`reach: 3% of the box changed -> no pad`.

The CSV (one row per box: box number, time, pads, pack seconds) is in
`data/events_<date>.csv`, exactly as before.

**Every** box and reach, with the reason for each decision, is also written to
`data/boxcounter.log`:

```bash
tail -n 30 data/boxcounter.log
```

The web dashboard (if the Pi is on your network) shows the same thing on a
phone or PC: `http://<pi-address>:8080` — find the address with
`hostname -I`. The live picture shows the packing zone, the box, its pads and
time, and `HAND` during a reach; the second picture shows what the counter
"sees" (white = box, orange = something new, red = movement).

## 4. Check it against reality

Pack 3–5 boxes while counting the pads yourself, then compare with the
`RECENT BOXES` table. If something is off, the log says exactly what happened
for every reach.

## 5. Tuning — symptom → setting

All in the `station:` section of `config/config.yaml`; restart the counter
after editing.

| Symptom | Change |
|---|---|
| Pads counted **too low**: two quick pads become one | `quiet_seconds: 0.5` → `0.4` |
| Pads counted **too high**: one pad counts twice (hand rests in the box mid-reach) | `quiet_seconds` → `0.7` |
| Reaches that add no pad are counted (log shows small %, e.g. 7%) | `pad_min_change: 0.06` → `0.10` |
| Real pads missed (log shows e.g. `4% … -> no pad` for a real pad) | `pad_min_change` → `0.04` |
| Each reach places a fixed bundle of N pads | `pieces_per_visit: N` |
| A box is never recognised; `now:` says `ignoring … not cardboard-coloured` | `box_color: any` |
| A box is never recognised; nothing happens at all | Run `tools/station_setup.py` again; check `min_box_frac` |
| Boxes queued next to the spot get counted | Run the setup again (it narrows the `zone`) |
| A quick stop (not a real box) gets counted | `min_pack_seconds: 3.0` → higher |

## 6. Record a clip so the counting can be tuned to your station

The most useful thing for improving the counting is a short video of real
packing. While the counter runs:

```bash
python3 -m boxcounter --record
```

Pack 3–5 boxes as usual (count the pads yourself), then press Ctrl-C. The
clip is saved as `data/recordings/clip_<date>_<time>.mp4` (about 10 MB per
minute) and can be replayed any time, giving exactly the same counts:

```bash
python3 -m boxcounter --source data/recordings/clip_<date>_<time>.mp4 --no-web
```

To share it: copy it to a USB stick, or upload it on github.com to the
repository (Add file → Upload files; files up to 25 MB, about 2 minutes),
together with how many pads went into each box.

## 7. Known limits

- **Two pads in one reach count as one.** The camera counts reaches. If pads
  always go in as fixed bundles, set `pieces_per_visit`.
- **Boxes need a gap.** If the next box is pushed in so closely behind the
  packed one that the spot is never empty for about half a second, the two
  are taken as one box.
- **A box already standing on the spot when the counter starts** becomes part
  of the "empty" picture: it is not counted (the spot is repaired when it is
  taken away; the next boxes count normally).
- **The first moments of a box**: the box is recognised within about a
  second of being set down. A pad pushed in during that second is still
  counted (it is compared with how the box looked when it arrived).
- **If reach counting turns out not to be reliable enough at your station**,
  the alternative is to count pads where they come out of the machine (each
  pad torn off the outlet = one pad). That needs a recorded clip first to see
  how the outlet looks to the camera.
