"""The terminal view is how calibration is done without a browser."""

import re

import numpy as np

from boxcounter import asciiview


def strip(lines):
    return [re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", ln) for ln in lines]


def test_grid_dimensions_are_respected():
    mask = np.zeros((480, 640), np.uint8)
    art = strip(asciiview.render(mask, (480, 640), 40, 12))
    assert len(art) == 12
    assert all(len(line) == 40 for line in art)


def test_empty_mask_is_blank():
    mask = np.zeros((480, 640), np.uint8)
    art = strip(asciiview.render(mask, (480, 640), 40, 12, colour=False))
    assert all(line.strip() == "" for line in art)


def test_foreground_shows_as_density_characters():
    mask = np.zeros((480, 640), np.uint8)
    mask[200:280, 300:400] = 255
    art = strip(asciiview.render(mask, (480, 640), 40, 12, colour=False))
    body = "".join(art)
    assert "@" in body or "#" in body
    # the blob is in the middle, so the top row stays empty
    assert art[0].strip() == ""


def test_counting_line_lands_on_the_right_row():
    mask = np.zeros((480, 640), np.uint8)
    art = strip(asciiview.render(mask, (480, 640), 40, 20,
                                 line=("y", 240.0), colour=False))
    rows_with_line = [i for i, ln in enumerate(art) if "=" in ln]
    assert len(rows_with_line) == 1
    assert 8 <= rows_with_line[0] <= 11, "line at y=0.5 should be mid-grid"


def test_vertical_line_for_x_axis():
    mask = np.zeros((480, 640), np.uint8)
    art = strip(asciiview.render(mask, (480, 640), 40, 12,
                                 line=("x", 320.0), colour=False))
    assert all("!" in ln for ln in art)


def test_detected_boxes_are_outlined():
    mask = np.zeros((480, 640), np.uint8)
    mask[100:200, 100:250] = 255
    art = strip(asciiview.render(mask, (480, 640), 60, 20,
                                 boxes=[(100, 100, 150, 100)], colour=False))
    body = "".join(art)
    assert "|" in body and "-" in body


def test_overlays_do_not_change_dimensions():
    mask = np.zeros((480, 640), np.uint8)
    mask[100:300, 200:400] = 255
    art = strip(asciiview.render(
        mask, (480, 640), 50, 16,
        roi_px=(32, 0, 576, 480), line=("y", 300.0),
        zone_px=(64, 24, 448, 216), boxes=[(200, 100, 200, 200)], colour=False))
    assert len(art) == 16
    assert all(len(line) == 50 for line in art)


def test_handles_missing_image():
    assert asciiview.render(None, (480, 640), 40, 12) == ["(no image)"]


def test_colour_output_contains_escapes_and_resets():
    mask = np.zeros((480, 640), np.uint8)
    art = asciiview.render(mask, (480, 640), 20, 6, line=("y", 240.0), colour=True)
    assert any("\x1b[" in ln for ln in art)
    assert all(ln.endswith("\x1b[0m") for ln in art)


def test_grid_size_stays_within_the_terminal():
    cols, rows = asciiview.grid_size()
    assert 24 <= cols <= 78
    assert 8 <= rows <= 22
