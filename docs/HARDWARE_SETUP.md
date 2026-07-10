# Hardware Setup

## Camera choice and cable

The IMX219 (Camera Module 2, 8 MP) is a good fit: cheap, well supported by
libcamera, and this application needs neither high resolution nor low-light
performance. **The Raspberry Pi 5 uses 22-pin mini-FPC camera connectors**,
not the 15-pin connector of earlier Pis — you need a *22-pin to 15-pin* CSI
ribbon ("Pi 5 camera cable"). Either of the Pi 5's two camera ports works;
the software auto-detects.

Seat the ribbon with the contacts facing the correct way at both ends (on the
Pi 5 connector, contacts face inward toward the board; on the camera, follow
the connector's markings), and verify:

```bash
rpicam-hello --list-cameras     # must list imx219
```

## Mounting geometry

Mount the camera looking **straight down** at the belt centerline. The IMX219
field of view is **62.2° horizontal × 48.8° vertical**, so at height *H*
above the belt it covers approximately:

| Height H | across-belt coverage (1.21·H) | along-belt coverage (0.91·H) |
|---|---|---|
| 40 cm | 48 cm | 36 cm |
| 60 cm | 73 cm | 55 cm |
| 80 cm | 97 cm | 73 cm |
| 100 cm | 121 cm | 91 cm |

Guidelines:

- The **whole belt width** should be inside the image with a small margin,
  and along the belt you want to see **at least 2× the longest box**, so a
  box is fully visible for several frames before it reaches the counting line.
- The default config maps the *wider* FoV axis across the image's width. If
  your belt runs along the image's horizontal axis instead, just set
  `counting.axis: x` — no need to physically rotate anything.
- Boxes must appear **smaller than ~60 %** of the region of interest
  (`max_area_frac`), and bigger than ~1 % (`min_area_frac`); both are tunable.
- **Rigidity matters more than precision.** A vibrating camera smears the
  background model. Use a proper bracket, not zip ties.
- Focus: the stock IMX219 is fixed-focus near ~50 cm and beyond — fine here.
  If the image is soft, the lens can be rotated (gently, it's threaded) after
  breaking the glue dot.

## Lighting

Background subtraction wants a **temporally stable** image:

- Best: diffuse, constant artificial light (LED panel **without PWM
  dimming** — PWM flicker beats against the shutter). Mount it beside the
  camera, angled, to minimize specular glare off shiny tape.
- Avoid: direct sunlight stripes that creep across the belt, people's
  shadows falling on the belt, and reflective belts with overhead point lights.
- The software locks the camera's exposure/white-balance after startup
  (`camera.lock_exposure`), rejects MOG2-detected shadows, and slowly adapts
  to lighting drift while the belt is empty — but it cannot fix a strobing
  light.
- Dark belts with dark boxes: add light at an angle. In desperation the
  `mog2_var_threshold` can go down to ~16, at the cost of noise sensitivity.

## Raspberry Pi placement

- Keep the Pi off the vibrating conveyor frame if possible; the CSI ribbon
  (usually ≤ 30 cm) sets the distance limit. Longer runs: mount Pi + camera
  on the same bracket.
- Use the official 27 W PSU; undervoltage throttles the CPU (check with
  `vcgencmd get_throttled` — should be `0x0`).
- The Pi 5 wants airflow; the official active cooler is cheap insurance in a
  warm factory. Thermal throttling starts at 80–85 °C.
- Industrial dust: put the Pi in a vented enclosure; a dusty camera lens is
  the #1 slow-degradation failure.

## GPIO output to a PLC / counter (optional)

One clean 3.3 V pulse (default 50 ms on BCM 17, physical pin 11) is emitted
per counted box when `gpio.enabled: true`.

**Never connect a Pi GPIO directly to 24 V PLC inputs.** Use one of:

- an **optocoupler module** (e.g. PC817 board): GPIO + GND on the input side,
  PLC voltage on the output side — recommended, galvanically isolated;
- a **relay module** rated for 3.3 V trigger (slower, audible, fine for
  totalizers);
- a **3.3 V-tolerant PLC high-speed counter input**, direct wiring
  (GPIO 17 → input, Pi GND → input common) only if the PLC side is
  specified for 3.3 V logic.

Pinout reminder (Pi 5, 40-pin header): BCM 17 = physical pin 11; ground =
pins 6/9/14/20/25/30/34/39. Set `gpio.pin`, `gpio.active_high` and
`gpio.pulse_ms` in the config to match the receiving device (PLC inputs
usually want ≥ 10 ms).

Pulses are serialized through a worker thread that guarantees a LOW gap
between them, so two boxes counted in the same frame still produce two
distinct rising edges (never one merged pulse). With the default 50 ms pulse
+ 10 ms gap the output tops out around **16 pulses/s** — well above realistic
box rates. If your belt genuinely exceeds that, lower `gpio.pulse_ms` to keep
the pulse period shorter than the minimum box-to-box interval. gpiozero on the
Pi 5 needs the lgpio backend (`python3-lgpio`, installed by `scripts/install.sh`);
without it, pulses are silently disabled and a warning is logged at startup.

## Bill of materials (reference)

| Qty | Item |
|---|---|
| 1 | Raspberry Pi 5, 2 GB |
| 1 | Official 27 W USB-C power supply |
| 1 | Official Pi 5 active cooler |
| 1 | IMX219 camera module (Camera Module 2) |
| 1 | 22-pin↔15-pin CSI cable, length to suit mount |
| 1 | 16 GB+ microSD (A1/A2) |
| 1 | Camera mounting bracket / arm above belt |
| 1 | LED work light, non-PWM (if ambient light varies) |
| 1 | Optocoupler board (only if pulsing a PLC) |
