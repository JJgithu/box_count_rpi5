"""The wizard writes to the live config — it must never produce an unloadable one.

Turning "counts nothing" into "the service will not start" would be far worse
than doing nothing, so these are invariants rather than examples.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import wizard                                          # noqa: E402

from boxcounter.config import load_config              # noqa: E402
from boxcounter.configedit import set_values           # noqa: E402

REAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def area_limits(sizes):
    """Mirror of the wizard's area-limit rule, for invariant testing."""
    lo, hi = wizard._pct(sizes, 0.10), wizard._pct(sizes, 0.90)
    lo_v = round(max(wizard._NOISE_FRAC, lo * 0.5), 4)
    hi_v = round(min(0.9, max(hi * 2.5, lo_v * 4)), 3)
    return lo_v, hi_v


@pytest.mark.parametrize("sizes", [
    [0.0003] * 50,                    # all specks (the config-bricking case)
    [0.0001] * 50,                    # smaller still
    [0.0011, 0.0013] * 25,            # tiny boxes
    [0.01, 0.02, 0.03] * 20,          # normal
    [0.05, 0.2] * 25,                 # large
    [0.4, 0.5, 0.6] * 20,             # very large
    [0.0002] + [0.8] * 49,            # one speck among big boxes
    [0.9] * 50,                       # saturated
])
def test_area_limits_are_always_ordered_and_valid(sizes):
    lo_v, hi_v = area_limits(sizes)
    assert 0 < lo_v < hi_v <= 1.0, f"invalid pair for {sizes[:3]}: {lo_v}, {hi_v}"


@pytest.mark.parametrize("sizes", [
    [0.0003] * 50, [0.0011, 0.0013] * 25, [0.01, 0.02, 0.03] * 20, [0.9] * 50,
])
def test_proposed_area_limits_produce_a_loadable_config(tmp_path, sizes):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(REAL_CONFIG, cfg_path)
    lo_v, hi_v = area_limits(sizes)
    text, _ = set_values(cfg_path.read_text(), {
        "processing.min_area_frac": lo_v,
        "processing.max_area_frac": hi_v,
    })
    cfg_path.write_text(text)
    cfg = load_config(cfg_path)          # must not raise
    assert cfg.processing.min_area_frac == lo_v


@pytest.mark.parametrize("line,direction,axis", [
    (0.1, "positive", "y"), (0.9, "positive", "y"),
    (0.1, "negative", "y"), (0.9, "negative", "y"),
    (0.1, "positive", "x"), (0.9, "negative", "x"),
    (0.5, "positive", "y"), (0.5, "negative", "x"),
])
def test_zone_clamping_keeps_it_inside_the_frame(line, direction, axis):
    """Mirror of the wizard's zone clamp; the zone must stay in bounds for
    every combination, or the written config fails validation."""
    for zone in ([0.0, 0.0, 1.0, 1.0], [0.8, 0.8, 0.2, 0.2],
                 [0.3, 0.85, 0.4, 0.15], [0.0, 0.5, 0.1, 0.5]):
        zx, zy, zw, zh = zone
        gap = 0.04
        if axis == "y":
            if direction != "negative":
                zh = min(zh, max(0.05, line - gap - zy))
            else:
                new_zy = min(max(zy, line + gap), 0.94)
                zh = max(0.05, zy + zh - new_zy)
                zy = new_zy
        else:
            if direction != "negative":
                zw = min(zw, max(0.05, line - gap - zx))
            else:
                new_zx = min(max(zx, line + gap), 0.94)
                zw = max(0.05, zx + zw - new_zx)
                zx = new_zx
        zx = min(max(0.0, zx), 0.94)
        zy = min(max(0.0, zy), 0.94)
        zw = min(max(0.05, zw), 1.0 - zx)
        zh = min(max(0.05, zh), 1.0 - zy)
        assert 0.0 <= zx < 1.0 and 0.0 <= zy < 1.0
        assert 0.0 < zw <= 1.0 and 0.0 < zh <= 1.0
        assert zx + zw <= 1.0 + 1e-9, f"zone runs off the right: {zx},{zw}"
        assert zy + zh <= 1.0 + 1e-9, f"zone runs off the bottom: {zy},{zh}"


def test_validate_proposal_accepts_good_values(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(REAL_CONFIG, cfg_path)
    problems = wizard._validate_proposal(cfg_path, {
        "processing.min_area_frac": 0.03,
        "processing.max_area_frac": 0.5,
        "counting.line_position": 0.62,
        "counting.direction": "positive",
    })
    assert problems == []


def test_validate_proposal_rejects_bad_values_without_touching_the_file(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(REAL_CONFIG, cfg_path)
    before = cfg_path.read_text()
    problems = wizard._validate_proposal(cfg_path, {
        "processing.min_area_frac": 0.5,     # min >= max
        "processing.max_area_frac": 0.2,
    })
    assert problems, "an invalid pair must be reported"
    assert any("min_area_frac" in p for p in problems)
    assert cfg_path.read_text() == before, "validation must not write anything"


def test_validate_proposal_rejects_out_of_bounds_zone(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy2(REAL_CONFIG, cfg_path)
    problems = wizard._validate_proposal(cfg_path, {
        "packing.zone": [0.5, 0.9, 0.8, 0.5],   # runs off both edges
    })
    assert problems and any("zone" in p for p in problems)


def test_noise_threshold_is_the_min_area_floor():
    """The 'that is noise' cutoff and the min_area floor must agree, or the
    wizard can suggest a minimum that still rejects what it just measured."""
    assert wizard._NOISE_FRAC == 0.002
    lo_v, _ = area_limits([wizard._NOISE_FRAC] * 20)
    assert lo_v <= wizard._NOISE_FRAC
