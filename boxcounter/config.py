"""Configuration loading and validation.

All tunable parameters live in a single YAML file (see config/config.yaml).
Geometry values (ROI, line position, distances) are expressed as fractions
of the frame so the same config works at any capture resolution.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    source: str = "picamera"          # picamera | video | usb
    video_path: str = ""              # used when source == "video"
    loop_video: bool = False
    realtime_video: bool = False      # pace video playback at native fps
    usb_index: int = 0                # used when source == "usb"
    width: int = 640
    height: int = 480
    fps: int = 30
    # Raw sensor mode. [1640, 1232] = IMX219 2x2 binned full field of view.
    # Without this, libcamera may pick a cropped mode for small output sizes.
    sensor_size: Optional[List[int]] = field(default_factory=lambda: [1640, 1232])
    hflip: bool = False
    vflip: bool = False
    warmup_seconds: float = 3.0
    # Lock auto-exposure/white-balance after warmup. Strongly recommended:
    # background subtraction assumes stable brightness.
    lock_exposure: bool = True
    # Manual exposure overrides (0 = keep auto, then lock if lock_exposure).
    exposure_time_us: int = 0
    analogue_gain: float = 0.0


@dataclass
class ProcessingConfig:
    # Region of interest (x, y, w, h) as fractions of the frame. Restrict to
    # the belt surface so edges/rollers don't generate false blobs.
    roi: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    method: str = "mog2"              # mog2 | static
    # Process in color (recommended): cardboard on a gray belt often has
    # strong chroma contrast but almost no luminance contrast, which a
    # grayscale pipeline would miss entirely.
    use_color: bool = True
    background_image: str = "data/background.png"   # for method: static
    mog2_history: int = 400
    mog2_var_threshold: float = 32.0
    detect_shadows: bool = True
    learning_rate: float = -1.0       # -1 = OpenCV auto
    # Freeze background learning while more than this fraction of the ROI is
    # foreground, i.e. only learn from (nearly) empty belt. Without this,
    # steady box traffic gets absorbed into the background model and later
    # boxes are missed. Raise toward 1.0 only to disable the guard.
    freeze_learning_fg_fraction: float = 0.02
    # Escape hatch for the freeze above: if this fraction of the ROI stays
    # foreground for relearn_after_freeze_frames consecutive frames, the scene
    # changed permanently (lights, exposure re-lock, re-taped belt) and the
    # model is rebuilt from the current scene. Kept near 1.0 so discrete box
    # traffic — which only ever covers part of the ROI and has gaps — never
    # trips it; only a near-total, sustained change does.
    relearn_fg_fraction: float = 0.9
    relearn_after_freeze_frames: int = 150   # ~5 s at 30 fps; 0 disables
    static_diff_threshold: int = 35   # for method: static
    blur_kernel: int = 5              # gaussian blur, odd, 0/1 = off
    open_kernel: int = 5              # morphological open: removes speckle
    close_kernel: int = 31            # morphological close: fuses open-box rims
    dilate_kernel: int = 0            # extra dilation, 0 = off
    # Accepted blob area, as a fraction of the ROI area. Wide by default so
    # a mixed line of small and large boxes is all counted; narrow them if
    # debris gets counted (raise min) or a lighting change registers as a
    # giant box (lower max).
    min_area_frac: float = 0.003
    max_area_frac: float = 0.80
    merge_gap_px: int = 24            # merge blobs closer than this gap
    warmup_frames: int = 60           # ignore detections while model settles


@dataclass
class TrackingConfig:
    max_distance_frac: float = 0.15   # match gate, fraction of frame diagonal
    max_disappeared: int = 10         # frames a track may coast unseen
    min_hits: int = 3                 # detections required before counting


@dataclass
class CountingConfig:
    axis: str = "y"                   # travel axis in the image: x | y
    line_position: float = 0.55       # line coordinate as fraction of frame
    hysteresis_frac: float = 0.03     # dead band around the line
    direction: str = "positive"       # positive | negative | any
    min_travel_frac: float = 0.10     # required travel along axis before count


@dataclass
class PackingConfig:
    """Packing-station monitoring: pieces placed into each box + pack time.

    A box that stops inside the zone becomes the "active" box. Each time the
    packer's arm reaches into it (foreground appearing in a ring around the
    box) and real motion happens inside the box, one insertion is counted.
    The session ends when the box departs; its piece count and pack time are
    attached to the box's count event when it crosses the counting line.
    """
    enabled: bool = False
    # Zone [x, y, w, h] (fractions of frame) where boxes stop to be packed.
    # Keep it upstream of (not overlapping) the counting line.
    zone: List[float] = field(default_factory=lambda: [0.0, 0.05, 1.0, 0.55])
    # -- box dwell / departure --
    dwell_speed_px: float = 1.5       # slower than this = box has stopped
    dwell_frames: int = 5             # consecutive slow frames to activate
    # A box must have been seen ARRIVING (moved at least this many pixels
    # since first tracked) before it can start a session. Keeps stationary
    # ghost blobs (e.g. after a background rebuild) from capturing sessions.
    min_arrival_px: float = 30.0
    depart_frames: int = 5            # consecutive gone/out-of-zone frames to end
    # How long a box may go untracked before the session is abandoned. The
    # packer's hand can hide a small box completely for several seconds, and
    # a box that cannot be seen is not the same as a box that has left, so
    # this is deliberately generous (5 s). Departure is detected
    # geometrically and does not wait for this.
    track_grace_frames: int = 150
    max_session_s: float = 600.0      # abandon a session after this long
    # -- arm detection (ring around the box) --
    ring_px: int = 28                 # width of the band around the box bbox
    arm_enter_frac: float = 0.06      # ring foreground fraction = arm present
    arm_exit_frac: float = 0.03       # hysteresis: below this = arm gone
    enter_frames: int = 2             # debounce on entry
    exit_frames: int = 3              # debounce on exit
    min_visit_frames: int = 4         # shorter visits are ignored (flicker)
    # If the "hand present" state persists this long with no motion inside the
    # box, the occupant is static (e.g. the next box queued into the watch
    # band) — the visit is closed and the new scene adopted as baseline.
    # Generous on purpose: a packer resting a hand in the box for a second or
    # two is normal, and on a small box a hand can fill the whole interior so
    # little frame-to-frame motion is measurable. A queued box sits there for
    # minutes, so waiting 10 s to conclude "this is furniture" costs nothing.
    static_exit_frames: int = 300
    max_visit_frames: int = 300       # hard backstop for a stuck visit
    # -- insertion confirmation (inside the box) --
    interior_inset_frac: float = 0.12 # shrink bbox by this to get the interior
    motion_threshold: int = 18        # gray-level delta that counts as motion
    interior_motion_frac: float = 0.04  # interior motion needed during a visit
    appearance_check: bool = False    # also require the interior to LOOK
    appearance_delta: float = 4.0     # different after the visit (mean |diff|)
    # -- accounting --
    pieces_per_visit: int = 1         # pads placed per reach (fixed bundles)
    expected_pieces: int = 0          # warn when a box leaves with a different
                                      # count; 0 disables the check


@dataclass
class OutputConfig:
    data_dir: str = "data"
    sqlite: bool = True
    csv: bool = True
    log_level: str = "INFO"
    heartbeat_seconds: float = 30.0


@dataclass
class GpioConfig:
    enabled: bool = False
    pin: int = 17
    active_high: bool = True
    pulse_ms: int = 50


@dataclass
class WebConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 70


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gpio: GpioConfig = field(default_factory=GpioConfig)
    web: WebConfig = field(default_factory=WebConfig)


def _build(dc_cls, data: dict, path: str):
    """Build a dataclass from a dict, warning about unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"config section '{path}' must be a mapping, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(dc_cls)}
    unknown = set(data) - names
    for key in sorted(unknown):
        log.warning("Ignoring unknown config key '%s.%s'", path, key)
    kwargs = {k: v for k, v in data.items() if k in names}
    return dc_cls(**kwargs)


def _validate(cfg: AppConfig) -> None:
    errors = []
    if cfg.camera.source not in ("picamera", "video", "usb"):
        errors.append(f"camera.source must be picamera|video|usb, got '{cfg.camera.source}'")
    if cfg.camera.source == "video" and not cfg.camera.video_path:
        errors.append("camera.video_path is required when camera.source is 'video'")
    if cfg.camera.width <= 0 or cfg.camera.height <= 0:
        errors.append("camera.width/height must be positive")

    roi = cfg.processing.roi
    if len(roi) != 4 or not all(isinstance(v, (int, float)) for v in roi):
        errors.append("processing.roi must be [x, y, w, h] fractions")
    else:
        x, y, w, h = roi
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9):
            errors.append(f"processing.roi {roi} out of bounds (fractions of frame)")
    if cfg.processing.method not in ("mog2", "static"):
        errors.append(f"processing.method must be mog2|static, got '{cfg.processing.method}'")
    if not (0.0 < cfg.processing.min_area_frac < cfg.processing.max_area_frac <= 1.0):
        errors.append("processing: need 0 < min_area_frac < max_area_frac <= 1")

    if cfg.counting.axis not in ("x", "y"):
        errors.append(f"counting.axis must be x|y, got '{cfg.counting.axis}'")
    if not (0.0 < cfg.counting.line_position < 1.0):
        errors.append("counting.line_position must be between 0 and 1")
    if cfg.counting.direction not in ("positive", "negative", "any"):
        errors.append(f"counting.direction must be positive|negative|any, got '{cfg.counting.direction}'")

    if cfg.tracking.min_hits < 1:
        errors.append("tracking.min_hits must be >= 1")

    pk = cfg.packing
    if pk.enabled:
        zone = pk.zone
        if len(zone) != 4 or not all(isinstance(v, (int, float)) for v in zone):
            errors.append("packing.zone must be [x, y, w, h] fractions")
        else:
            x, y, w, h = zone
            if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                    and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9):
                errors.append(f"packing.zone {zone} out of bounds (fractions of frame)")
            elif cfg.counting.axis == "y" and y + h > cfg.counting.line_position:
                log.warning("packing.zone reaches past the counting line "
                            "(zone ends at %.2f, line at %.2f) — the box should "
                            "finish packing before it is counted",
                            y + h, cfg.counting.line_position)
            # The detector only sees inside processing.roi; a packing zone
            # outside it can never detect an arm.
            rx, ry, rw, rh = cfg.processing.roi
            if (x < rx - 1e-9 or y < ry - 1e-9
                    or x + w > rx + rw + 1e-9 or y + h > ry + rh + 1e-9):
                log.warning("packing.zone %s extends outside processing.roi %s "
                            "— arm detection is blind outside the ROI; align "
                            "the zone (and leave ring_px of ROI margin around "
                            "the parked box)", zone, cfg.processing.roi)
        if not (0.0 < pk.arm_exit_frac < pk.arm_enter_frac < 1.0):
            errors.append("packing: need 0 < arm_exit_frac < arm_enter_frac < 1")
        if pk.ring_px < 4:
            errors.append("packing.ring_px must be >= 4")
        if pk.pieces_per_visit < 1:
            errors.append("packing.pieces_per_visit must be >= 1")
        if pk.min_visit_frames < 1 or pk.max_visit_frames <= pk.min_visit_frames:
            errors.append("packing: need 1 <= min_visit_frames < max_visit_frames")

    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the YAML config file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of config must be a mapping")

    known = {f.name for f in dataclasses.fields(AppConfig)}
    for key in sorted(set(raw) - known):
        log.warning("Ignoring unknown config section '%s'", key)

    cfg = AppConfig(
        camera=_build(CameraConfig, raw.get("camera"), "camera"),
        processing=_build(ProcessingConfig, raw.get("processing"), "processing"),
        tracking=_build(TrackingConfig, raw.get("tracking"), "tracking"),
        counting=_build(CountingConfig, raw.get("counting"), "counting"),
        packing=_build(PackingConfig, raw.get("packing"), "packing"),
        output=_build(OutputConfig, raw.get("output"), "output"),
        gpio=_build(GpioConfig, raw.get("gpio"), "gpio"),
        web=_build(WebConfig, raw.get("web"), "web"),
    )
    _validate(cfg)
    return cfg
