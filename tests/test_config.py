import pytest

from boxcounter.config import AppConfig, load_config


def write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_load_repo_default_config():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    assert cfg.camera.source == "picamera"
    assert cfg.counting.axis == "y"


def test_empty_config_gives_defaults(tmp_path):
    cfg = load_config(write(tmp_path, ""))
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.width == 640


def test_partial_override(tmp_path):
    cfg = load_config(write(tmp_path, "counting:\n  line_position: 0.7\n"))
    assert cfg.counting.line_position == 0.7
    assert cfg.counting.axis == "y"


def test_invalid_roi_rejected(tmp_path):
    with pytest.raises(ValueError, match="roi"):
        load_config(write(tmp_path, "processing:\n  roi: [0.5, 0.5, 0.8, 0.8]\n"))


def test_invalid_direction_rejected(tmp_path):
    with pytest.raises(ValueError, match="direction"):
        load_config(write(tmp_path, "counting:\n  direction: sideways\n"))


def test_video_source_requires_path(tmp_path):
    with pytest.raises(ValueError, match="video_path"):
        load_config(write(tmp_path, "camera:\n  source: video\n"))


def test_unknown_keys_warn_but_load(tmp_path, caplog):
    cfg = load_config(write(tmp_path, "camera:\n  bogus_key: 1\nnot_a_section:\n  a: 1\n"))
    assert cfg.camera.width == 640
