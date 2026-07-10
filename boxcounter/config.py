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
    min_area_frac: float = 0.01       # blob area as fraction of ROI area
    max_area_frac: float = 0.60
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
        output=_build(OutputConfig, raw.get("output"), "output"),
        gpio=_build(GpioConfig, raw.get("gpio"), "gpio"),
        web=_build(WebConfig, raw.get("web"), "web"),
    )
    _validate(cfg)
    return cfg
