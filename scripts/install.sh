#!/usr/bin/env bash
# Install the box counter on Raspberry Pi OS (Bookworm or newer), Pi 5.
# Run from the repository root:   bash scripts/install.sh
set -euo pipefail

APPDIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"

echo "==> Installing system packages (all from Raspberry Pi OS repos, offline-friendly)"
sudo apt-get update
sudo apt-get install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-flask \
    python3-gpiozero \
    python3-lgpio \
    rpicam-apps

echo "==> Creating data directory"
mkdir -p "$APPDIR/data"

echo "==> Quick camera check (5 s preview-less capture)"
if rpicam-hello -t 2000 -n >/dev/null 2>&1; then
    echo "    camera OK"
else
    echo "    WARNING: rpicam-hello failed — check the CSI ribbon cable and"
    echo "    that the camera is detected:  rpicam-hello --list-cameras"
fi

echo "==> Installing systemd service (boxcounter.service)"
sed -e "s|__USER__|$RUN_USER|g" -e "s|__APPDIR__|$APPDIR|g" \
    "$APPDIR/systemd/boxcounter.service" | sudo tee /etc/systemd/system/boxcounter.service >/dev/null
sudo systemctl daemon-reload

echo
echo "Done. Next steps:"
echo "  1. Calibrate:      python3 tools/calibrate.py          (see docs/CALIBRATION.md)"
echo "  2. Test run:       python3 -m boxcounter --config config/config.yaml"
echo "     Dashboard:      http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080/"
echo "  3. Enable service: sudo systemctl enable --now boxcounter"
