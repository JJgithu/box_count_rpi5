"""Optional GPIO pulse output — one pulse per counted box.

Useful for feeding a PLC counter input, a stack light, or an external totalizer.
Uses gpiozero, which supports the Pi 5's RP1 GPIO via the lgpio pin factory.
Degrades to a no-op when GPIO hardware/libraries are unavailable (e.g. during
development on a PC), so the same code runs everywhere.

Pulses are emitted from a dedicated worker thread that guarantees a clean
LOW gap between consecutive pulses. This matters because two boxes can be
counted in the same frame or on consecutive frames (33 ms at 30 fps < a
50 ms pulse): a naive "restart the pulse" approach would hold the line HIGH
continuously and the PLC would see one rising edge for two boxes. The queue
serialises them into distinct edges instead.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from .config import GpioConfig

log = logging.getLogger(__name__)

_MIN_GAP_S = 0.01          # guaranteed LOW time between pulses
_QUEUE_MAX = 1000          # backstop; box rates never approach this


class GpioPulse:
    def __init__(self, cfg: GpioConfig):
        self.cfg = cfg
        self.dev = None
        self._queue: "queue.Queue[int]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker = None
        self._stop = threading.Event()
        self._dropped = 0
        if not cfg.enabled:
            return
        try:
            from gpiozero import DigitalOutputDevice
            self.dev = DigitalOutputDevice(cfg.pin, active_high=cfg.active_high,
                                           initial_value=False)
            self._worker = threading.Thread(target=self._run, name="gpio",
                                            daemon=True)
            self._worker.start()
            log.info("GPIO pulse output on pin %d (%d ms, active_%s)",
                     cfg.pin, cfg.pulse_ms, "high" if cfg.active_high else "low")
        except Exception as e:
            # RPi.GPIO fallback on a Pi 5 (missing python3-lgpio), no hardware,
            # or a busy pin all land here — the counter keeps running headless.
            log.warning("GPIO unavailable, pulses disabled: %s", e)
            self.dev = None

    def _run(self) -> None:
        on_s = self.cfg.pulse_ms / 1000.0
        while not self._stop.is_set():
            try:
                self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.dev.on()
                time.sleep(on_s)
                self.dev.off()
                time.sleep(_MIN_GAP_S)
            except Exception:
                log.exception("GPIO pulse failed")

    def pulse(self) -> None:
        """Enqueue one pulse. Non-blocking; never stalls the pipeline."""
        if self.dev is None:
            return
        try:
            self._queue.put_nowait(1)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.error("GPIO pulse queue full; dropped %d pulses", self._dropped)

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        if self.dev is not None:
            try:
                self.dev.off()
                self.dev.close()
            except Exception:
                pass
            self.dev = None
