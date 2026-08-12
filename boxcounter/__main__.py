"""Command-line entry point: python -m boxcounter --config config/config.yaml

Dependencies are checked before anything heavy is imported, so a fresh
install reports every missing package at once with the exact command to fix
it, instead of failing on one cryptic ImportError at a time.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# module name -> Raspberry Pi OS package that provides it
_CORE_DEPS = [
    ("cv2", "python3-opencv"),
    ("numpy", "python3-numpy"),
    ("yaml", "python3-yaml"),
]
_OPTIONAL_DEPS = [
    ("picamera2", "python3-picamera2", "camera.source: picamera"),
    ("flask", "python3-flask", "web.enabled: true"),
    ("gpiozero", "python3-gpiozero", "gpio.enabled: true"),
]


def _missing(modules):
    """Return the entries whose module cannot be imported.

    Uses find_spec so nothing is actually executed — importing picamera2
    on a non-Pi can be slow and noisy.
    """
    out = []
    for entry in modules:
        name = entry[0]
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(entry)
    return out


def _report_missing(missing, *, fatal: bool) -> None:
    pkgs = " ".join(sorted({m[1] for m in missing}))
    where = sys.stderr if fatal else sys.stdout
    print("", file=where)
    print("Missing Python packages:", file=where)
    for entry in missing:
        note = f"   (needed for {entry[2]})" if len(entry) > 2 else ""
        print(f"  - {entry[0]:<12} provided by {entry[1]}{note}", file=where)
    print("\nInstall them with:\n", file=where)
    print(f"  sudo apt-get install -y {pkgs}\n", file=where)
    print("Or install everything the project needs at once:\n", file=where)
    print("  bash scripts/install.sh\n", file=where)
    print("On Raspberry Pi OS always use apt, not pip: pip builds OpenCV from\n"
          "source (slow, often fails on 2 GB) and can break picamera2.\n",
          file=where)


def _check_core_deps() -> int:
    missing = _missing(_CORE_DEPS)
    if missing:
        _report_missing(missing, fatal=True)
        return 2
    return 0


def _check_optional_deps(cfg, args) -> int:
    """Check only what this configuration actually needs."""
    needed = []
    for name, pkg, reason in _OPTIONAL_DEPS:
        if name == "picamera2":
            if args.source or cfg.camera.source != "picamera":
                continue
        elif name == "flask":
            if args.no_web or not cfg.web.enabled:
                continue
        elif name == "gpiozero":
            continue    # gpio_out degrades to a no-op and logs a warning
        needed.append((name, pkg, reason))
    missing = _missing(needed)
    if missing:
        _report_missing(missing, fatal=True)
        return 2
    return 0


def _live_reset(port: int) -> bool:
    """Ask a running counter to reset, so its on-screen total updates now.

    Returns True if a running instance handled it. Resetting the database
    underneath a running counter would leave its in-memory total stale, so
    the running instance is always given first refusal.
    """
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/reset", method="POST")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _reset_count(cfg) -> int:
    from .storage import CountStore

    if _live_reset(cfg.web.port):
        print("Count reset. The running counter picked it up immediately.")
        print("History is kept — the database and CSV files are untouched.")
        return 0

    store = CountStore(cfg.output.data_dir, use_sqlite=cfg.output.sqlite,
                       use_csv=False)
    try:
        before = store.total()
        store.reset()
    finally:
        store.close()
    print(f"Count reset: {before} -> 0")
    print("History is kept — the database and CSV files are untouched.")
    print("\nIf the counter is running right now, restart it so its screen")
    print("catches up:  sudo systemctl restart boxcounter")
    return 0


def _print_totals(cfg) -> int:
    from .storage import CountStore

    store = CountStore(cfg.output.data_dir, use_sqlite=cfg.output.sqlite,
                       use_csv=False)
    try:
        s = store.summary()
        recent = store.recent(5)
    finally:
        store.close()
    print(f"Boxes counted        : {s['total']}")
    if cfg.packing.enabled:
        print(f"Pads counted (total) : {s['pieces_total']}")
        avg_p = s["avg_pieces"]
        avg_t = s["avg_pack_seconds"]
        print(f"Average pads per box : "
              f"{avg_p:.1f}" if avg_p is not None else
              "Average pads per box : --")
        print(f"Average pack time    : "
              f"{avg_t:.1f} s" if avg_t is not None else
              "Average pack time    : --")
    if recent:
        print("\nMost recent boxes:")
        for r in recent:
            pads = "--" if r["pieces"] is None else f"{r['pieces']} pads"
            secs = "--" if r["pack_seconds"] is None else f"{r['pack_seconds']:.1f} s"
            print(f"  {r['time'].replace('T', ' ')}   {pads:>8}   {secs:>8}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="boxcounter",
        description="Offline conveyor box counter for Raspberry Pi 5 + IMX219")
    parser.add_argument("--config", "-c", default="config/config.yaml",
                        help="path to YAML config (default: config/config.yaml)")
    parser.add_argument("--source", help="override input with a video file (testing)")
    parser.add_argument("--display", action="store_true",
                        help="show OpenCV preview windows (needs a desktop)")
    parser.add_argument("--no-web", action="store_true",
                        help="disable the web dashboard")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after N frames (benchmarks/tests)")
    parser.add_argument("--log-level", default=None,
                        help="override output.log_level (DEBUG, INFO, ...)")
    parser.add_argument("--check", action="store_true",
                        help="verify dependencies and config, then exit")
    parser.add_argument("--no-status", action="store_true",
                        help="plain log lines instead of the live status panel")
    parser.add_argument("--reset", action="store_true",
                        help="set the running total back to zero, then exit "
                             "(history is kept in the database and CSVs)")
    parser.add_argument("--total", action="store_true",
                        help="print the current totals, then exit")
    args = parser.parse_args(argv)

    rc = _check_core_deps()
    if rc:
        return rc

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 2

    from .config import load_config          # needs yaml, checked above
    cfg = load_config(cfg_path)

    if args.reset:
        return _reset_count(cfg)
    if args.total:
        return _print_totals(cfg)

    rc = _check_optional_deps(cfg, args)
    if rc:
        return rc

    if args.check:
        print("Dependencies OK.")
        print(f"Config OK: {cfg_path}")
        print(f"  camera   : {args.source or cfg.camera.source} "
              f"{cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.fps} fps")
        print(f"  counting : axis={cfg.counting.axis} "
              f"line={cfg.counting.line_position} dir={cfg.counting.direction}")
        print(f"  packing  : {'on' if cfg.packing.enabled else 'off'}"
              + (f", zone={cfg.packing.zone}" if cfg.packing.enabled else ""))
        print(f"  web      : {'off' if args.no_web or not cfg.web.enabled else 'http://0.0.0.0:%d/' % cfg.web.port}")
        print(f"  data dir : {cfg.output.data_dir}")
        return 0

    from .status import supported as status_supported   # light import
    use_status = (not args.no_status) and status_supported() \
        and not args.log_level

    level = (args.log_level or cfg.output.log_level).upper()
    logging.basicConfig(
        # The panel owns the screen; routine INFO chatter would corrupt it.
        level=logging.WARNING if use_status else getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    from .pipeline import Pipeline           # needs cv2, checked above
    pipeline = Pipeline(cfg,
                        video_override=args.source,
                        display=args.display,
                        enable_web=False if args.no_web else None,
                        max_frames=args.max_frames,
                        status=use_status)
    pipeline.install_signal_handlers()
    summary = pipeline.run()
    print(f"frames={summary['frames']} counted_total={summary['total']} "
          f"session={summary['session']} fps={summary['fps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
