"""Make Masks can detect on the image as drawn, and the gap's hover is a line.

2026-09-21, the maintainer: "in make masks there should be an option for the
cellpose/otsu object detection to work on the normalized image or not ... if
used for object detection, the intensity values in the top left corner should
also reflect the normalized image", and "when hovered jsut a thin blue line in
between the two hould appear, not the entire space between them become blue".
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt import mask_engine as engine  # noqa: E402
from spacr.qt.screens import make_masks as mm  # noqa: E402


@pytest.fixture
def screen(qtbot):
    s = mm.MakeMasksScreen()
    qtbot.addWidget(s)
    img = np.full((120, 120), 200, np.uint16)
    img[30:60, 30:60] = 3000
    s._canvas.image = img
    s._canvas.mask = np.zeros(img.shape, np.int32)
    s._canvas.refresh()
    return s


def test_off_by_default_and_detectors_read_the_loaded_pixels(screen):
    assert not screen._detect_normalized.isChecked()
    assert screen._detector_image() is screen._canvas.image


def test_on_detectors_read_exactly_what_is_drawn(screen):
    screen._detect_normalized.setChecked(True)
    c = screen._canvas
    expected = engine.normalize_uint16(c.image, c.norm_lo, c.norm_hi)
    np.testing.assert_array_equal(screen._detector_image(), expected)


def test_the_magnifier_box_cuts_from_the_same_field(screen):
    screen._detect_normalized.setChecked(True)
    crop = screen._magnifier.region_for((20, 20, 70, 70), invert=False)
    np.testing.assert_array_equal(
        crop, screen._canvas.detection_source()[20:70, 20:70])


def test_invert_and_normalize_compose_like_the_picture(screen):
    screen._cp_invert.setChecked(True)
    screen._detect_normalized.setChecked(True)
    c = screen._canvas
    expected = engine.normalize_uint16(engine.invert_normalized(c.image),
                                       c.norm_lo, c.norm_hi)
    np.testing.assert_array_equal(screen._detector_image(), expected)


def test_a_new_percentile_is_a_new_field(screen):
    screen._detect_normalized.setChecked(True)
    first = screen._detector_image()
    assert screen._detector_image() is first
    screen._norm_hi.setValue(90.0)
    assert screen._detector_image() is not first


def test_the_corner_readout_reports_the_normalized_value(screen):
    c = screen._canvas
    c.readout = engine.PixelReadout(40, 40, 3000.0)
    screen._detect_normalized.setChecked(True)
    source = c.detection_source()
    assert c._value_at(source, (40, 40)) == float(source[40, 40])
    assert "normalized" in c.readout_text()


def test_a_float_field_normalizes_onto_zero_one():
    img = np.linspace(0, 10, 100, dtype=np.float32).reshape(10, 10)
    out = engine.normalize_for_detection(img, 0.0, 100.0)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0


def test_the_hover_is_a_two_pixel_line_in_a_twelve_pixel_gap(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter, QWidget

    splitter = QSplitter(Qt.Horizontal)
    qtbot.addWidget(splitter)
    splitter.setHandleWidth(mm.SETTINGS_GAP)
    splitter.setStyleSheet(mm.thin_hover_line_sheet().replace(":hover", ""))
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    splitter.resize(300, 100)
    splitter.show()
    qtbot.wait(10)
    image = splitter.handle(1).grab().toImage()
    colours = [image.pixelColor(x, 50).name() for x in range(image.width())]
    accent = [c for c in colours if c != colours[0]]
    assert image.width() == mm.SETTINGS_GAP
    assert len(accent) == 2
