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

    level = (args.log_level or cfg.output.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    from .pipeline import Pipeline           # needs cv2, checked above
    pipeline = Pipeline(cfg,
                        video_override=args.source,
                        display=args.display,
                        enable_web=False if args.no_web else None,
                        max_frames=args.max_frames)
    pipeline.install_signal_handlers()
    summary = pipeline.run()
    print(f"frames={summary['frames']} counted_total={summary['total']} "
          f"session={summary['session']} fps={summary['fps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
