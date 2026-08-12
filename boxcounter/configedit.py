"""Edit single values in config.yaml while preserving comments and layout.

The config file is documentation as much as configuration — every key has an
explanatory comment. A round-trip through PyYAML would discard all of it, so
the calibration wizard rewrites individual lines instead.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(#.*)?$")
_KEY_RE = re.compile(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*):(\s*)([^#]*?)(\s*#.*)?$")


def format_value(value) -> str:
    """Render a Python value the way the config file writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def set_values(text: str, updates: Dict[str, object]) -> Tuple[str, List[str]]:
    """Apply {"section.key": value} to the config text.

    Returns the new text and a list of human-readable change descriptions.
    Keys that do not exist are reported as changes with a "not found" note
    rather than silently ignored.
    """
    lines = text.splitlines(keepends=True)
    applied: List[str] = []

    by_section: Dict[str, Dict[str, object]] = {}
    for dotted, value in updates.items():
        section, _, key = dotted.partition(".")
        if not key:
            raise ValueError(f"update key must be 'section.key', got '{dotted}'")
        by_section.setdefault(section, {})[key] = value

    current = None
    seen: set = set()
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        m_sec = _SECTION_RE.match(stripped)
        if m_sec:
            current = m_sec.group(1)
            continue
        if current is None or current not in by_section:
            continue
        m_key = _KEY_RE.match(stripped)
        if not m_key:
            continue
        indent, key, gap, old, comment = m_key.groups()
        if key not in by_section[current]:
            continue
        new_value = format_value(by_section[current][key])
        old_value = (old or "").strip()
        if old_value != new_value:
            applied.append(f"{current}.{key}: {old_value} -> {new_value}")

        prefix = f"{indent}{key}:{gap or ' '}"
        tail = comment or ""
        if tail:
            # Keep the comment in its original column so repeated edits do
            # not progressively ragged this heavily aligned file. The value
            # group is non-greedy, so the padding spaces are part of the
            # comment group — take the column from the original line.
            body = tail.lstrip()
            comment_col = len(stripped) - len(body)
            pad = comment_col - (len(prefix) + len(new_value))
            tail = (" " * pad if pad > 0 else " ") + body
        lines[i] = f"{prefix}{new_value}{tail}\n"
        seen.add(f"{current}.{key}")

    for dotted in updates:
        if dotted not in seen:
            applied.append(f"{dotted}: NOT FOUND in the config file")
    return "".join(lines), applied


def backup(path: Path) -> Path:
    """Copy the config aside before editing; returns the backup path."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    shutil.copy2(path, dest)
    return dest


def apply_to_file(path: Path, updates: Dict[str, object]) -> Tuple[Path, List[str]]:
    """Back up, then apply updates in place. Returns (backup_path, changes)."""
    original = path.read_text()
    new_text, changes = set_values(original, updates)
    bak = backup(path)
    path.write_text(new_text)
    return bak, changes
