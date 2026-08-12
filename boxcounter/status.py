"""Live terminal status panel.

Shows the numbers an operator actually watches — total boxes, average pack
time, average pads per box, and the last box — redrawn in place instead of
scrolling log lines past.

Only used when stdout is a real terminal. Under systemd the output goes to
the journal, where an in-place redraw would be unreadable, so the pipeline
falls back to ordinary log lines automatically.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from typing import Deque, Optional, Tuple

_CSI = "\x1b["
_HIDE_CURSOR = f"{_CSI}?25l"
_SHOW_CURSOR = f"{_CSI}?25h"
_CLEAR = f"{_CSI}2J{_CSI}H"      # clear screen, home
_HOME = f"{_CSI}H"
_CLEAR_LINE = f"{_CSI}K"

_DIM = f"{_CSI}2m"
_BOLD = f"{_CSI}1m"
_AMBER = f"{_CSI}33m"
_GREEN = f"{_CSI}32m"
_CYAN = f"{_CSI}36m"
_RESET = f"{_CSI}0m"

_RECENT_ROWS = 8


def supported(stream=None) -> bool:
    """True when an in-place panel makes sense on this stream."""
    stream = stream or sys.stdout
    if os.environ.get("TERM", "") == "dumb":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _fmt_secs(v: Optional[float]) -> str:
    if v is None:
        return "--"
    if v >= 60:
        return f"{int(v // 60)}m {v % 60:04.1f}s"
    return f"{v:.1f} s"


def _fmt_num(v: Optional[float], digits: int = 1) -> str:
    return "--" if v is None else f"{v:.{digits}f}"


class StatusDisplay:
    """Redraws a compact panel of the headline numbers."""

    def __init__(self, csv_path: str = "", packing: bool = True,
                 refresh_s: float = 0.5, stream=None):
        self.stream = stream or sys.stdout
        self.csv_path = csv_path
        self.packing = packing
        self.refresh_s = refresh_s
        self._recent: Deque[Tuple[int, Optional[int], Optional[float], str]] = \
            deque(maxlen=_RECENT_ROWS)
        self._last_draw = 0.0
        self._started = False

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self._started = True
        self.stream.write(_HIDE_CURSOR + _CLEAR)
        self.stream.flush()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self.stream.write(_SHOW_CURSOR + "\n")
        self.stream.flush()

    # -- data ------------------------------------------------------------

    def add_box(self, box_number: int, pieces: Optional[int],
                pack_seconds: Optional[float], ts: float) -> None:
        self._recent.appendleft(
            (box_number, pieces, pack_seconds,
             datetime.fromtimestamp(ts).strftime("%H:%M:%S")))

    # -- drawing ---------------------------------------------------------

    def draw(self, stats: dict, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < self.refresh_s:
            return
        self._last_draw = now
        width = min(72, max(52, shutil.get_terminal_size((80, 24)).columns - 2))
        inner = width - 2

        out = [_HOME]

        def line(text: str = "", raw_len: Optional[int] = None) -> None:
            out.append(text + _CLEAR_LINE + "\n")

        def box_top(title: str) -> None:
            label = f"─ {title} "
            line(f"{_DIM}┌{label}{'─' * max(0, inner - len(label))}┐{_RESET}")

        def box_bottom() -> None:
            line(f"{_DIM}└{'─' * inner}┘{_RESET}")

        def row(left: str, right: str = "", colour: str = "") -> None:
            pad = inner - 2 - len(left) - len(right)
            line(f"{_DIM}│{_RESET} {left}{' ' * max(1, pad)}"
                 f"{colour}{right}{_RESET} {_DIM}│{_RESET}")

        clock = datetime.now().strftime("%H:%M:%S")
        fps = stats.get("fps", 0.0)
        head = f"{_BOLD}BOX COUNTER{_RESET}"
        tail = f"{_DIM}{clock}   {fps:.0f} fps{_RESET}"
        pad = inner - len("BOX COUNTER") - len(f"{clock}   {fps:.0f} fps")
        line(f" {head}{' ' * max(1, pad)}{tail}")
        line()

        # -- headline numbers --
        box_top("TOTALS")
        row("Boxes counted", f"{stats.get('total', 0)}", _BOLD + _GREEN)
        if self.packing:
            row("Average pack time", _fmt_secs(stats.get("avg_pack_seconds")), _BOLD)
            row("Average pads per box", _fmt_num(stats.get("avg_pieces")), _BOLD)
            row("Pads counted (total)", f"{stats.get('pieces_total', 0)}", _DIM)
        row("Boxes per minute", _fmt_num(stats.get("rate_per_min")), _DIM)
        box_bottom()
        line()

        # -- last box --
        if self.packing:
            box_top("LAST BOX")
            if self._recent:
                n, pieces, secs, when = self._recent[0]
                pads = "--" if pieces is None else f"{pieces} pads"
                row(f"Box #{n}   at {when}", f"{pads}   {_fmt_secs(secs)}",
                    _BOLD + _AMBER)
            else:
                row("waiting for the first box...", "", _DIM)
            box_bottom()
            line()

        # -- recent table --
        box_top("RECENT BOXES")
        row(f"{'box':>5}  {'pads':>5}  {'pack time':>10}  {'time':>8}", "", _DIM)
        if not self._recent:
            row("none yet", "", _DIM)
        for n, pieces, secs, when in self._recent:
            pads = "--" if pieces is None else str(pieces)
            row(f"{n:>5}  {pads:>5}  {_fmt_secs(secs):>10}  {when:>8}")
        for _ in range(_RECENT_ROWS - len(self._recent)):
            row("")
        box_bottom()
        line()

        # -- live packing state --
        if self.packing:
            if stats.get("packing_now"):
                hand = "  — hand in box" if stats.get("hand_in") else ""
                line(f" {_CYAN}packing now: {stats.get('current_pieces', 0)} pads, "
                     f"{stats.get('current_elapsed_s', 0):.0f}s{hand}{_RESET}")
            else:
                line(f" {_DIM}no box at the packing station{_RESET}")
            line()

        if self.csv_path:
            line(f" {_DIM}CSV: {self.csv_path}{_RESET}")
        line(f" {_DIM}Ctrl-C to stop     reset the total: "
             f"python3 -m boxcounter --reset{_RESET}")
        out.append(f"{_CSI}J")     # clear anything below

        self.stream.write("".join(out))
        self.stream.flush()
