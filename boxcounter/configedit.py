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


def add_missing(text: str, updates: Dict[str, object]) -> Tuple[str, List[str]]:
    """Append keys that set_values could not find, so an older config file
    (missing a whole section, or a newer key) still receives every update.
    Keys go at the end of their section, or into a new section at the end."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    present: Dict[str, set] = {}
    section_end: Dict[str, int] = {}
    current = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        m_sec = _SECTION_RE.match(stripped)
        if m_sec:
            current = m_sec.group(1)
            present.setdefault(current, set())
            section_end[current] = i + 1
            continue
        m_key = _KEY_RE.match(stripped)
        if current is not None and m_key:
            present[current].add(m_key.group(2))
            section_end[current] = i + 1
    added: List[str] = []
    inserts: Dict[int, List[str]] = {}
    new_sections: Dict[str, List[str]] = {}
    for dotted, value in updates.items():
        section, _, key = dotted.partition(".")
        if key in present.get(section, set()):
            continue
        entry = f"  {key}: {format_value(value)}\n"
        if section in section_end:
            inserts.setdefault(section_end[section], []).append(entry)
        else:
            new_sections.setdefault(section, []).append(entry)
        added.append(f"{dotted}: (added) {format_value(value)}")
    for at in sorted(inserts, reverse=True):
        lines[at:at] = inserts[at]
    for section, entries in new_sections.items():
        lines.append(f"\n{section}:\n")
        lines.extend(entries)
    return "".join(lines), added


def backup(path: Path) -> Path:
    """Copy the config aside before editing; returns the backup path."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    shutil.copy2(path, dest)
    return dest


def apply_to_file(path: Path, updates: Dict[str, object],
                  add: bool = False) -> Tuple[Path, List[str]]:
    """Back up, then apply updates in place. Returns (backup_path, changes).
    With add=True, keys missing from the file are appended instead of being
    reported as NOT FOUND."""
    original = path.read_text()
    new_text, changes = set_values(original, updates)
    if add:
        changes = [c for c in changes if "NOT FOUND" not in c]
        new_text, added = add_missing(new_text, updates)
        changes += added
    bak = backup(path)
    path.write_text(new_text)
    return bak, changes
