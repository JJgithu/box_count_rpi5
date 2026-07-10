"""Offline web dashboard: live count, rate, MJPEG preview, debug mask view.

Served on the local network only — no external assets, no internet needed.
JPEG encoding happens in the web server thread, per connected viewer, so the
counting loop pays nothing when nobody is watching.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from .config import WebConfig

log = logging.getLogger(__name__)


class SharedState:
    """Thread-safe hand-off point between the pipeline and the web server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._stats: dict = {}

    def update(self, frame: Optional[np.ndarray], mask: Optional[np.ndarray],
               stats: dict) -> None:
        with self._lock:
            if frame is not None:
                self._frame = frame
            if mask is not None:
                self._mask = mask
            self._stats = dict(stats)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_mask(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._mask is None else self._mask.copy()

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Box Counter</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #14171c; color: #e8eaed; }
  header { padding: 10px 16px; background: #1d232b; display: flex; align-items: baseline; gap: 16px; }
  h1 { font-size: 18px; margin: 0; }
  #status { color: #9aa4b0; font-size: 13px; }
  main { display: flex; flex-wrap: wrap; gap: 16px; padding: 16px; }
  .card { background: #1d232b; border-radius: 10px; padding: 14px 18px; }
  .big { font-size: 64px; font-weight: 700; line-height: 1.1; }
  .label { color: #9aa4b0; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
  .metrics { display: flex; gap: 28px; margin-top: 10px; }
  .metric b { font-size: 22px; display: block; }
  img { max-width: 100%; border-radius: 8px; display: block; }
  button { background: #2b3542; color: #e8eaed; border: 0; border-radius: 6px;
           padding: 8px 14px; cursor: pointer; margin-top: 12px; }
  button:hover { background: #38455a; }
  table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  td, th { padding: 3px 10px; text-align: left; color: #c4ccd6; }
  th { color: #9aa4b0; font-weight: 500; }
  a { color: #7ab3ef; }
</style>
</head>
<body>
<header><h1>&#128230; Box Counter</h1><span id="status">connecting&hellip;</span></header>
<main>
  <div class="card">
    <div class="label">Total count</div>
    <div class="big" id="total">&ndash;</div>
    <div class="metrics">
      <div class="metric"><span class="label">boxes/min</span><b id="rate">&ndash;</b></div>
      <div class="metric"><span class="label">camera fps</span><b id="fps">&ndash;</b></div>
      <div class="metric"><span class="label">uptime</span><b id="uptime">&ndash;</b></div>
    </div>
    <button onclick="resetCount()">Reset count</button>
    <div class="label" style="margin-top:16px">Recent events</div>
    <table id="events"><tr><th>time</th><th>track</th><th>size</th></tr></table>
  </div>
  <div class="card">
    <div class="label">Live view (<a href="#" onclick="return swap()" id="swap">show mask</a>)</div>
    <img id="stream" src="/stream.mjpg" width="640">
  </div>
</main>
<script>
let mask = false;
function swap() {
  mask = !mask;
  document.getElementById('stream').src = mask ? '/mask.mjpg' : '/stream.mjpg';
  document.getElementById('swap').textContent = mask ? 'show camera' : 'show mask';
  return false;
}
function fmtUp(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h + 'h ' + m + 'm';
}
async function tick() {
  try {
    const r = await fetch('/api/stats');
    const s = await r.json();
    document.getElementById('total').textContent = s.total;
    document.getElementById('rate').textContent = s.rate_per_min.toFixed(1);
    document.getElementById('fps').textContent = s.fps.toFixed(1);
    document.getElementById('uptime').textContent = fmtUp(s.uptime_s);
    document.getElementById('status').textContent = 'live';
    const rows = ['<tr><th>time</th><th>track</th><th>size</th></tr>'];
    for (const e of s.recent || []) {
      rows.push(`<tr><td>${e.time.replace('T',' ')}</td><td>#${e.track_id}</td><td>${e.w}&times;${e.h}</td></tr>`);
    }
    document.getElementById('events').innerHTML = rows.join('');
  } catch (err) {
    document.getElementById('status').textContent = 'disconnected';
  }
}
async function resetCount() {
  if (confirm('Reset the running total to zero?'))
    await fetch('/api/reset', {method: 'POST'});
  tick();
}
setInterval(tick, 1000);
tick();
</script>
</body>
</html>
"""


class WebUI:
    def __init__(self, cfg: WebConfig, state: SharedState,
                 reset_cb: Optional[Callable[[], None]] = None,
                 recent_cb: Optional[Callable[[int], list]] = None):
        self.cfg = cfg
        self.state = state
        self.reset_cb = reset_cb
        self.recent_cb = recent_cb
        self._thread: Optional[threading.Thread] = None

    def _mjpeg_generator(self, getter):
        import cv2
        import numpy as np
        boundary = b"--frame\r\n"
        # Placeholder shown before the first frame (camera still starting) or
        # during an outage. Yielding it — rather than silently sleeping — lets
        # werkzeug notice a closed connection and reap the thread instead of
        # leaking one per disconnected viewer.
        placeholder = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(placeholder, "no signal", (90, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        while True:
            img = getter()
            if img is None:
                img = placeholder
            ok, jpg = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if ok:
                yield (boundary + b"Content-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                       + jpg.tobytes() + b"\r\n")
            time.sleep(0.1)  # ~10 fps per viewer is plenty for monitoring

    def _build_app(self):
        from flask import Flask, Response, jsonify

        app = Flask(__name__)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        @app.route("/")
        def index():
            return _PAGE

        @app.route("/api/stats")
        def stats():
            s = self.state.get_stats()
            if self.recent_cb is not None:
                s["recent"] = self.recent_cb(10)
            return jsonify(s)

        @app.route("/api/reset", methods=["POST"])
        def reset():
            if self.reset_cb is not None:
                self.reset_cb()
            return jsonify({"ok": True})

        @app.route("/stream.mjpg")
        def stream():
            return Response(self._mjpeg_generator(self.state.get_frame),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/mask.mjpg")
        def mask():
            return Response(self._mjpeg_generator(self.state.get_mask),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        return app

    def start(self) -> None:
        app = self._build_app()

        def serve():
            try:
                app.run(host=self.cfg.host, port=self.cfg.port,
                        threaded=True, use_reloader=False, debug=False)
            except Exception:
                log.exception("Web UI stopped")

        self._thread = threading.Thread(target=serve, name="webui", daemon=True)
        self._thread.start()
        log.info("Web dashboard on http://%s:%d/", self.cfg.host, self.cfg.port)
