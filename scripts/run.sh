#!/usr/bin/env bash
# Run the box counter in the foreground (Ctrl-C to stop).
cd "$(dirname "$0")/.."
exec python3 -m boxcounter --config config/config.yaml "$@"
