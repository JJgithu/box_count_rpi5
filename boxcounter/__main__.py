"""Command-line entry point: python -m boxcounter --config config/config.yaml"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline


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
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(cfg_path)

    level = (args.log_level or cfg.output.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

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
