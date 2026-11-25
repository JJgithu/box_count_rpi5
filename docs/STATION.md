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
| **A box** | Something in the **packing zone** that is not the empty rollers, **stands still**, is **box-sized** and looks like **cardboard**. Counted when it **leaves** the spot, either way along the rollers or lifted off. |
| **Pads in that box** | **At the machine** (after setup step 3): the paper just below the outlet is watched. The pad hanging there gone, or a *different* pad hanging there, means one pad was pulled off. Without step 3: per **reach** into the box (see section 7). |
| **Pack time** | From the moment the box was set down on the spot to the moment it left. |

What is deliberately ignored:
- **Everything outside the packing zone** — the packer standing right
  against the box, people walking by, boxes waiting beside the spot.
- **Hands and arms**: they do not stand still, they are smaller than a box,
  and an arm lying across a box is cut off its outline, not added to it.
- **The pad hanging from the machine** (after step 3): it hangs over part of
  the box and is replaced all the time; it is never part of a box.
- For pads counted at the machine: the same pad hanging a little differently
  (dragged along by a box, pushed aside by a hand), paper being re-arranged
  in the box, and the machine pushing out the next pad into an empty outlet.

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
because the setup tool edits `config/config.yaml` on the Pi and the update
changes that file too.) **Then run the setup again** (next section): the
settings are measured afresh, including where the pads come out.

If the counter runs as a service, stop it while setting up — only one program
can use the camera at a time:

```bash
sudo systemctl stop boxcounter
```

## 2. Set up — three pictures

```bash
python3 tools/station_setup.py
```

1. It asks you to **clear the spot**: no box on the rollers in front of the
   machine, hands away. Leave a pad hanging from the machine if one normally
   hangs there. Press Enter.
2. It asks you to put an **empty box exactly where it stands while it is
   packed**, flaps open as usual, then step back. The pad hanging from the
   machine may hang over the box — that is fine, as long as it is the **same
   pad, hanging the same way**, as in picture 1. Press Enter.
3. It asks you to **pull the hanging pad off and put it aside** (not in the
   box), and let the machine push out the next one. When the new pad hangs
   still and your hands are away, press Enter. From the difference it learns
   where pads hang out of the machine. That paper is then never mistaken
   for part of the box — and **pads are counted there**. The setup tries
   the counting on the two pads right away and tells you:
   `pads will be counted at the machine`, or why not (then pads are counted
   per reach into the box instead). Typing `s` skips this step: pads are
   then counted per reach, which is less exact at a station where the pad
   hangs into the box.

It then draws what it found in the terminal (the box as `@#`, the packing
zone as `:.:`) and tells you:

- how much of the picture the box fills (good: roughly 10–60 %),
- whether the whole box is in view (if it touches an edge, turn or raise the
  camera if you can),
- whether the box looks cardboard-coloured to the camera.

Answer **Y** to save. A backup of the previous config is kept next to it
(`config/config.backup-<date>.yaml`). The pictures are saved in `data/`
(`setup_empty.jpg`, `setup_box.jpg`, `setup_outlet.jpg`, and
`setup_result.jpg` with the box, the packing zone and the pad outlet drawn
on it).

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
| `waiting for a box (ignoring what is at the spot: …)` | Something stands there that is not taken as a box, and why: `too small to be a box (7% of the picture, boxes are at least 9%)`, `too big …`, `not cardboard-coloured`, `it reaches in from the picture edge like an arm`. If this shows while a real box stands there, that reason is what to fix (see section 5). |
| `something stopped at the spot — checking (0.4 s)` | A box arrived; it is making sure (well under a second). |
| `box at the station: 2 pads, 14 s` | Packing. `(hand in box)` during a reach. |
| `ignoring an object at the spot: …` | Something stopped there that is not counted, and why. |

`last:` says what was decided last, for example `a pad was taken from the
machine`, or (per reach) `reach: 3% of the box changed -> no pad`.

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
| A box is not recognised; `now:` says `… not cardboard-coloured` | `box_color: any` |
| … says `too small to be a box (N% …)` | `min_box_frac` → a little below N%, or run the setup again with this box |
| … says `too big to be a box` | `max_box_frac` → higher, or run the setup again |
| … says nothing at all (`waiting for a box`) | Run `tools/station_setup.py` again: the camera or the spot has moved |
| Boxes queued next to the spot get counted | Run the setup again (it narrows the `zone`) |
| A quick stop (not a real box) gets counted | `min_pack_seconds: 3.0` → higher |
| Box recognised late while you are already working on it | `confirm_busy_seconds: 1.2` → `0.8` |
| **Pads at the machine** counted too low: you pull each pad the instant it is out | `outlet_quiet_seconds: 0.4` → `0.25` |
| Pads at the machine counted too high or not at all | Run the setup again (step 3 measures where the paper hangs); or `pad_method: reach` |
| **Pads per reach** too low: two quick pads become one | `quiet_seconds: 0.5` → `0.4` |
| Pads per reach too high: one pad counts twice | `quiet_seconds` → `0.7` |
| Reaches that add no pad are counted (log shows small %, e.g. 7%) | `pad_min_change: 0.06` → `0.10` |
| Each pad taken is really a fixed bundle of N | `pieces_per_visit: N` |

## 6. Show what the counter saw (to tune it to your station)

When the counts are wrong, the run that went wrong is the most useful thing
there is: it can be replayed through the counter, frame by frame, as often
as needed. Three things, from quick to thorough:

**1. The report** — settings and the last decisions, as text to paste, plus
the setup pictures in one image:

```bash
python3 tools/station_report.py
```

Every box, every pad, and every time something at the spot was *not* taken
as a box (with the reason) is in there. The image is
`data/report/setup.jpg`.

**2. A snapshot** — while the counter runs, open
`http://<pi-address>:8080/snapshot.jpg` on a phone or PC on the same network
(`hostname -I` on the Pi shows the address): the camera picture with what
the counter drew on it, next to what the counter "sees". Save the picture.

**3. A recorded clip** — the real thing. While the counter runs:

```bash
python3 -m boxcounter --record
```

Pack 2–3 boxes as usual (count the pads yourself), then press Ctrl-C. The
clip is saved in `data/recordings/` (about 10 MB per minute, so keep it to
a minute or two) and replays any time with exactly the same counts:

```bash
python3 -m boxcounter --source "$(ls -t data/recordings/*.mp4 | head -1)" --no-web
```

To turn it into pictures (one moment per second, camera picture next to
what the counter sees, and every decision in `data/sheets/decisions.txt`):

```bash
python3 tools/station_sheet.py "$(ls -t data/recordings/*.mp4 | head -1)"
```

**Getting the files off the Pi.** While the counter runs, open
`http://<pi-address>:8080/files` in a browser on the same network and tap
the file: clips, sheets, setup pictures, the log. Or, from a Windows PC
(PowerShell):

```powershell
scp pi@countbox.local:box_count_rpi5/data/recordings/*.mp4 .
```

## 7. Known limits

- **Boxes need a gap.** If the next box is pushed in so closely behind the
  packed one that the spot is never empty for about half a second, the two
  are taken as one box.
- **One box at a time**: the spot packs one box; anything else that stops
  next to it waits until that box has left.
- **A box already standing on the spot when the counter starts** becomes part
  of the "empty" picture: it is not counted (the spot is repaired when it is
  taken away; the next boxes count normally).
- **Pads at the machine: each new pad has to hang still for a moment**
  (about a third of a second) before it is pulled off, so it can be told
  from the one before. Where the machine leaves the outlet bare for a
  moment between pads, even that is not needed.
- **Pads at the machine: kraft-brown paper.** The counter tells paper from
  the machine and the rollers by its colour and its creases. With white
  paper, use `pad_method: reach`.
- **A pad pulled off but not put in the box** (thrown away, put aside)
  still counts for the box standing there.
- **Pads per reach** (`pad_method: reach`, or setup step 3 skipped) is less
  exact where the pad hangs into the box: the paper sliding out of the
  machine changes the look of the box by itself, re-arranging paper counts
  as a pad, two pads in one reach count as one, and a reach only ends when
  no hand has moved over the box for `quiet_seconds`.

## 8. Would a different camera help?

Not by itself. The counter works on a picture reduced to 320 pixels wide,
compares it with the empty spot, and needs **colour** (cardboard and kraft
paper against grey rollers and dark clothes) and a **steady** picture far
more than speed or sharpness:

- A monochrome global-shutter camera (OV9281 and the like) would take away
  the colour, and its strengths — no distortion of fast-moving things,
  hundreds of frames per second — do not matter for a box that stands
  still while it is packed.
- What does matter: the camera **must not move** (a shaking arm shifts the
  whole picture), the **whole box with a margin** and the **machine's
  outlet** must be in view, and the light should be even. The counter
  already locks exposure and white balance after start-up and keeps the
  exposure to whole periods of the mains flicker (`camera.flicker_hz`).
- If the box does not fit in the picture from where the camera can be
  mounted, a **wider colour camera** helps (for example the Raspberry Pi
  Camera Module 3 Wide); it needs no change in the software.
