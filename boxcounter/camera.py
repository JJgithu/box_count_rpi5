"""Frame sources: Raspberry Pi CSI camera (Picamera2), video file, USB camera.

All sources return BGR numpy arrays compatible with OpenCV, or None when no
frame is available (end of video / camera failure).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


class FrameSource:
    """Interface for frame producers."""

    # A live source (camera) may return None transiently and should be
    # retried/restarted; a finite source (video file) returning None means EOF.
    is_live: bool = False

    def start(self) -> None:
        pass

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def media_time(self) -> Optional[float]:
        """Seconds of media position for finite sources (video files), so
        durations measured while processing faster/slower than real time stay
        correct. Live sources return None and the caller uses the wall clock."""
        return None

    def stop(self) -> None:
        pass


class PiCameraSource(FrameSource):
    """IMX219 (or any CSI camera) via Picamera2/libcamera.

    Requests the full-FoV binned sensor mode (1640x1232 on IMX219) and scales
    to the working resolution, so the whole belt stays visible. Picamera2's
    "RGB888" format stores pixels in B,G,R memory order, which is exactly
    what OpenCV expects.
    """

    is_live = True

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.picam2 = None

    def start(self) -> None:
        from picamera2 import Picamera2  # imported lazily: only exists on the Pi
        from libcamera import Transform

        cfg = self.cfg
        self.picam2 = Picamera2()
        kwargs = dict(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},
            transform=Transform(hflip=int(cfg.hflip), vflip=int(cfg.vflip)),
            controls={"FrameRate": float(cfg.fps)},
            buffer_count=4,
        )
        if cfg.sensor_size:
            kwargs["raw"] = {"size": tuple(cfg.sensor_size)}
        video_config = self.picam2.create_video_configuration(**kwargs)
        self.picam2.configure(video_config)
        self.picam2.start()
        log.info("Picamera2 started: %dx%d @ %d fps (sensor %s)",
                 cfg.width, cfg.height, cfg.fps, cfg.sensor_size)

        # Let auto exposure/white balance settle on the empty belt.
        time.sleep(max(0.0, cfg.warmup_seconds))

        if cfg.exposure_time_us > 0:
            controls = {
                "AeEnable": False,
                "ExposureTime": int(cfg.exposure_time_us),
            }
            if cfg.analogue_gain > 0:
                controls["AnalogueGain"] = float(cfg.analogue_gain)
            # Manual exposure must still freeze white balance, otherwise AWB
            # keeps retuning ColourGains as colored boxes fill the frame and
            # the color background model sees the belt shift. Lock to whatever
            # AWB converged to during warmup.
            if cfg.lock_exposure:
                md = self.picam2.capture_metadata()
                controls["AwbEnable"] = False
                if "ColourGains" in md:
                    controls["ColourGains"] = md["ColourGains"]
            self.picam2.set_controls(controls)
            log.info("Manual exposure: %d us, gain %s, awb %s",
                     cfg.exposure_time_us, cfg.analogue_gain or "auto",
                     "locked" if cfg.lock_exposure else "auto")
        elif cfg.lock_exposure:
            # Freeze whatever auto exposure converged to. A stable image is
            # essential for background subtraction.
            md = self.picam2.capture_metadata()
            controls = {"AeEnable": False, "AwbEnable": False}
            if "ExposureTime" in md:
                controls["ExposureTime"] = md["ExposureTime"]
            if "AnalogueGain" in md:
                controls["AnalogueGain"] = md["AnalogueGain"]
            if "ColourGains" in md:
                controls["ColourGains"] = md["ColourGains"]
            self.picam2.set_controls(controls)
            log.info("Locked exposure: %s us, gain %.2f",
                     controls.get("ExposureTime", "?"), controls.get("AnalogueGain", 0.0))

    def read(self) -> Optional[np.ndarray]:
        try:
            return self.picam2.capture_array("main")
        except Exception:
            log.exception("Camera capture failed")
            return None

    def stop(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                log.exception("Error stopping camera")
            self.picam2 = None


class VideoFileSource(FrameSource):
    """Read frames from a video file. Used for testing and tuning off-device."""

    def __init__(self, path: str, loop: bool = False, realtime: bool = False,
                 fps_hint: float = 30.0):
        self.path = path
        self.loop = loop
        self.realtime = realtime
        self.fps = fps_hint
        self.cap = None
        self._next_deadline = 0.0
        self._frames_read = 0

    def start(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.fps = fps
        self._next_deadline = time.monotonic()
        log.info("Video source: %s (%.1f fps)", self.path, self.fps)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(1, 0)  # CAP_PROP_POS_FRAMES
            ok, frame = self.cap.read()
            if not ok:
                return None
        self._frames_read += 1
        if self.realtime:
            now = time.monotonic()
            if now < self._next_deadline:
                time.sleep(self._next_deadline - now)
            self._next_deadline = max(now, self._next_deadline) + 1.0 / self.fps
        return frame

    def media_time(self) -> Optional[float]:
        return self._frames_read / self.fps if self.fps > 0 else None

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class UsbCameraSource(FrameSource):
    """USB webcam via V4L2. Fallback if a CSI camera is not available."""

    is_live = True

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap = None

    def start(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.cfg.usb_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera index {self.cfg.usb_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        log.info("USB camera %d started", self.cfg.usb_index)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def create_source(cfg: CameraConfig, video_override: Optional[str] = None) -> FrameSource:
    """Build the frame source described by the config.

    video_override forces a video file source regardless of config (used by
    `--source` on the command line and by the test suite).
    """
    if video_override:
        return VideoFileSource(video_override, loop=cfg.loop_video,
                               realtime=cfg.realtime_video, fps_hint=cfg.fps)
    if cfg.source == "picamera":
        return PiCameraSource(cfg)
    if cfg.source == "video":
        return VideoFileSource(cfg.video_path, loop=cfg.loop_video,
                               realtime=cfg.realtime_video, fps_hint=cfg.fps)
    if cfg.source == "usb":
        return UsbCameraSource(cfg)
    raise ValueError(f"Unknown camera source '{cfg.source}'")
