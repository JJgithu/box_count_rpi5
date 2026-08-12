"""The config editor writes to the user's live config — it must never corrupt it."""

import numpy as np
import pytest

from boxcounter.configedit import apply_to_file, format_value, set_values

SAMPLE = """\
# Header comment
camera:
  source: picamera            # picamera | video | usb
  width: 640
  fps: 30

processing:
  roi: [0.05, 0.0, 0.90, 1.0]
  min_area_frac: 0.01         # smallest accepted blob
  use_color: true

counting:
  line_position: 0.78         # counting line
  direction: positive

packing:
  enabled: true               # count pieces
  zone: [0.05, 0.05, 0.90, 0.45]

gpio:
  enabled: false              # pulse a pin
  pin: 17

web:
  enabled: true
  port: 8080
"""


def test_value_is_replaced_and_comment_kept():
    out, changes = set_values(SAMPLE, {"counting.line_position": 0.62})
    assert "  line_position: 0.62         # counting line\n" in out
    assert changes == ["counting.line_position: 0.78 -> 0.62"]


def test_everything_else_is_byte_identical():
    out, _ = set_values(SAMPLE, {"counting.line_position": 0.62})
    a = SAMPLE.splitlines()
    b = out.splitlines()
    assert len(a) == len(b)
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(differing) == 1, "only the edited line may change"


def test_same_key_in_different_sections_is_not_confused():
    """'enabled' exists in packing, gpio and web — the wrong one must not move."""
    out, changes = set_values(SAMPLE, {"gpio.enabled": True})
    assert changes == ["gpio.enabled: false -> true"]
    assert "  enabled: true               # count pieces\n" in out   # packing intact
    assert out.count("enabled: true") == 3       # packing + gpio + web
    assert "enabled: false" not in out
    # the web one is untouched and still its own line
    assert out.rstrip().endswith("port: 8080")


def test_comment_column_is_preserved_across_edits():
    """Repeated wizard runs must not ragged the aligned config file."""
    out, _ = set_values(SAMPLE, {"gpio.enabled": True})
    before = [ln for ln in SAMPLE.splitlines() if "pulse a pin" in ln][0]
    after = [ln for ln in out.splitlines() if "pulse a pin" in ln][0]
    assert before.index("#") == after.index("#")

    out2, _ = set_values(SAMPLE, {"processing.min_area_frac": 0.0301})
    b = [ln for ln in SAMPLE.splitlines() if "smallest accepted" in ln][0]
    a = [ln for ln in out2.splitlines() if "smallest accepted" in ln][0]
    assert b.index("#") == a.index("#")


def test_list_and_bool_formatting():
    out, _ = set_values(SAMPLE, {"packing.zone": [0.29, 0.2, 0.31, 0.38],
                                 "processing.use_color": False})
    assert "  zone: [0.29, 0.2, 0.31, 0.38]\n" in out
    assert "  use_color: false\n" in out


def test_multiple_updates_at_once():
    out, changes = set_values(SAMPLE, {
        "processing.min_area_frac": 0.03,
        "counting.direction": "negative",
        "camera.fps": 24,
    })
    assert "min_area_frac: 0.03         # smallest accepted blob" in out
    assert "direction: negative" in out
    assert "fps: 24" in out
    assert len(changes) == 3


def test_unknown_key_is_reported_not_silently_dropped():
    out, changes = set_values(SAMPLE, {"counting.nonexistent": 1})
    assert out == SAMPLE
    assert any("NOT FOUND" in c for c in changes)


def test_unchanged_value_is_not_reported_as_a_change():
    out, changes = set_values(SAMPLE, {"counting.direction": "positive"})
    assert changes == []
    assert out == SAMPLE


def test_bad_update_key_rejected():
    with pytest.raises(ValueError):
        set_values(SAMPLE, {"noSection": 1})


def test_apply_to_file_backs_up_first(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE)
    bak, changes = apply_to_file(cfg, {"counting.line_position": 0.5})
    assert bak.exists()
    assert bak.read_text() == SAMPLE, "backup must hold the original"
    assert "line_position: 0.5" in cfg.read_text()
    assert changes


def test_edited_file_still_loads_as_valid_config(tmp_path):
    """The whole point: the result must still parse as the real config."""
    import shutil
    from pathlib import Path

    from boxcounter.config import load_config
    src = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    dst = tmp_path / "config.yaml"
    shutil.copy2(src, dst)
    apply_to_file(dst, {
        "processing.min_area_frac": 0.0301,
        "processing.max_area_frac": 0.544,
        "counting.line_position": 0.62,
        "counting.direction": "positive",
        "packing.zone": [0.29, 0.2, 0.31, 0.38],
    })
    cfg = load_config(dst)
    assert cfg.processing.min_area_frac == 0.0301
    assert cfg.counting.line_position == 0.62
    assert cfg.counting.direction == "positive"
    assert cfg.packing.zone == [0.29, 0.2, 0.31, 0.38]
    # comments survived
    assert "# smallest accepted blob" in dst.read_text()


def test_format_value():
    assert format_value(True) == "true"
    assert format_value(False) == "false"
    assert format_value(0.5) == "0.5"
    assert format_value(3) == "3"
    assert format_value([0.1, 0.2]) == "[0.1, 0.2]"
    assert format_value("positive") == "positive"
