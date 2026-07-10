"""GPIO pulse serialization — no real hardware required.

Verifies the core guarantee that fixes the gpiozero blink() merge bug: N
counted boxes always produce N distinct on/off edge pairs, even when the
pulses are enqueued back-to-back faster than one pulse width.
"""

import threading
import time

from boxcounter.config import GpioConfig
from boxcounter.gpio_out import GpioPulse


class FakeDev:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def on(self):
        with self._lock:
            self.events.append("on")

    def off(self):
        with self._lock:
            self.events.append("off")

    def close(self):
        pass


def test_back_to_back_pulses_are_distinct_edges():
    cfg = GpioConfig(enabled=False, pulse_ms=5)   # disabled -> no real gpiozero
    g = GpioPulse(cfg)
    fake = FakeDev()
    g.dev = fake                                  # inject a fake output device
    g._worker = threading.Thread(target=g._run, daemon=True)
    g._worker.start()
    try:
        for _ in range(5):                        # 5 boxes counted at once
            g.pulse()
        deadline = time.monotonic() + 3.0
        while len([e for e in fake.events if e == "off"]) < 5:
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)
        pulse_events = list(fake.events)          # snapshot before close()
    finally:
        g.close()

    # Five boxes -> five distinct rising+falling edge pairs, strictly
    # alternating. (close() drives one extra off() to leave the pin LOW.)
    assert pulse_events.count("on") == 5
    assert pulse_events.count("off") == 5
    assert pulse_events == ["on", "off"] * 5


def test_pulse_is_noop_without_device():
    g = GpioPulse(GpioConfig(enabled=False))
    g.pulse()      # must not raise
    g.close()
