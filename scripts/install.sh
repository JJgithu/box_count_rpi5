#!/usr/bin/env bash
# Install the box counter on Raspberry Pi OS (Bookworm or newer), Pi 5.
#
#   bash scripts/install.sh                 packages + service
#   bash scripts/install.sh --service-only  just the systemd service
#                                           (when the packages are already in)
set -euo pipefail

APPDIR="$(cd "$(dirname "$0")/.." && pwd)"
# $USER is not set in every non-interactive shell, and `set -u` would abort.
RUN_USER="${SUDO_USER:-${USER:-$(id -un)}}"
SERVICE_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --service-only) SERVICE_ONLY=1 ;;
        -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$SERVICE_ONLY" -eq 0 ]; then
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
else
    echo "==> Skipping apt (--service-only)"
fi

echo "==> Verifying every Python dependency imports"
MISSING=""
for m in cv2 numpy yaml flask picamera2 gpiozero; do
    if python3 -c "import $m" >/dev/null 2>&1; then
        echo "    OK      $m"
    else
        echo "    MISSING $m"
        MISSING="$MISSING $m"
    fi
done
if [ -n "$MISSING" ]; then
    echo
    echo "ERROR: these modules still do not import:$MISSING"
    echo "The apt install did not fully succeed. Common causes:"
    echo "  - stale package index      -> sudo apt-get update, then re-run"
    echo "  - no network/DNS for apt   -> check: ping -c1 deb.debian.org"
    echo "  - full SD card             -> check: df -h /"
    echo "  - running inside a venv    -> deactivate, or recreate it with"
    echo "                                python3 -m venv --system-site-packages"
    exit 1
fi

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
if systemctl list-unit-files boxcounter.service >/dev/null 2>&1; then
    echo "    service installed (not started yet)"
else
    echo "    WARNING: the service does not appear in systemctl"
fi

echo
echo "Done. Next steps:"
echo "  1. Calibrate:      python3 tools/wizard.py             (see docs/CALIBRATION.md)"
echo "  2. Test run:       python3 -m boxcounter"
echo "     Dashboard:      http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080/"
echo "  3. Enable service: sudo systemctl enable --now boxcounter"
