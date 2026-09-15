"""Item 417, the magnifier's second round: its settings and its models.

The maintainer's words, after using item 407's magnifier: the box size's
maximum of 512 "should be able to be as high as the image is high/wide";
"holding shift and scrolling should increase or decrease the size setting";
the settings categories fold "like in the core applications"; with Cellpose
on, "all the cellpose models in the model zoo" and "the model zoo button";
the flow and cell-probability thresholds, and Otsu's threshold correction,
set in the EXISTING Cellpose-SAM category, are what the magnifier's detection
uses; and "other computer vision models like a live YOLO or DINOCell".

The canvas geometry and the coded-field stub are item 407's, imported from
its test module, so a pixel here means what it means there.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from spacr.qt.screens import make_masks as mm
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CANVAS_H,
    CANVAS_W,
    IMG_N,
    PIXMAP_N,
    SIZE,
    CodedStub,
    canvas_xy,
    coded_field,
    fields,  # noqa: F401 - a fixture, used by name
    hover,
    screen,  # noqa: F401 - a fixture, used by name
    switch_on,
    wait_for_result,
)


def shift_wheel(screen, img_x, img_y, delta=120, *, modifiers=Qt.ShiftModifier,
                sideways=False):
    """Turn the wheel one notch over image pixel (img_x, img_y)."""
    pos = QPointF(*canvas_xy(img_x, img_y))
    angle = QPoint(delta, 0) if sideways else QPoint(0, delta)
    screen._canvas.wheelEvent(QWheelEvent(
        pos, pos, QPoint(0, 0), angle, Qt.NoButton, modifiers,
        Qt.NoScrollPhase, False))


# ---------------------------------------------------------------------------
# 1. The box can be as large as the image
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_then_square(tmp_path: Path) -> Path:
    """Two fields: 40 x 700 (wider than the old 512 cap), then 64 x 64."""
    folder = tmp_path / "shapes"
    folder.mkdir()
    imageio.imwrite(folder / "a_wide.tif", np.ones((40, 700), np.uint16))
    imageio.imwrite(folder / "b_square.tif", coded_field())
    return folder


def test_the_size_reaches_the_open_images_longer_side_and_stops_there(
        qtbot, qt_theme_applied, wide_then_square):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        size, magnifier = made._mag_size, made._magnifier
        assert (size.minimum(), size.maximum()) == mm._MAGNIFIER_SIZE_RANGE, (
            "with no field open the range is the old one")

        assert made._open_folder(str(wide_then_square))
        assert made._canvas.image.shape == (40, 700)
        assert (size.minimum(), size.maximum()) == (32, 700)
        size.setValue(700)
        assert size.value() == 700 and magnifier.size == 700, (
            "past the old 512 cap, up to the field's own width")
        size.setValue(5000)
        assert size.value() == 700 and magnifier.size == 700
        magnifier.set_size(9999)
        assert magnifier.size == 700, "the magnifier clamps on its own too"

        made._on_next()
        qtbot.waitUntil(lambda: made._canvas.image is not None
                        and made._canvas.image.shape == (IMG_N, IMG_N),
                        timeout=5_000)
        assert size.maximum() == IMG_N, "the range follows the field that opened"
        assert size.value() == IMG_N and magnifier.size == IMG_N, (
            "a size past the new field's side is clamped to it")
    finally:
        made._magnifier.close()
        made.close_folded()


def test_the_range_is_never_empty_on_a_field_smaller_than_the_floor():
    assert mm._magnifier_size_range((20, 12)) == (20, 20)
    assert mm._magnifier_size_range((1, 1)) == (1, 1)
    assert mm._magnifier_size_range((300, 2000, 3)) == (32, 2000)
    assert mm._magnifier_size_range(None) == mm._MAGNIFIER_SIZE_RANGE


# ---------------------------------------------------------------------------
# 8. Shift + wheel changes the size, never the view
# ---------------------------------------------------------------------------

def test_shift_wheel_changes_the_size_and_stops_at_the_images_size(
        qtbot, screen):
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    magnifier = screen._magnifier
    zoom = magnifier.zoom
    assert magnifier.size == SIZE == 32

    shift_wheel(screen, 30, 30, 120)
    grown = magnifier.size
    assert grown > SIZE
    assert screen._mag_size.value() == grown, "the Size box follows the wheel"
    assert magnifier.zoom == zoom, "Shift + wheel is not the magnifier's zoom"
    assert not screen._canvas.is_zoomed(), "and it is not the view's zoom"

    for _ in range(40):
        shift_wheel(screen, 30, 30, 120)
    assert magnifier.size == IMG_N == screen._mag_size.value(), (
        "it stops at the image's own size")
    assert not screen._canvas.is_zoomed()

    shift_wheel(screen, 30, 30, -120)
    assert magnifier.size < IMG_N
    for _ in range(40):
        shift_wheel(screen, 30, 30, -120)
    assert magnifier.size == 32 == screen._mag_size.value(), (
        "and at the bottom of the range the other way")


def test_a_sideways_shift_notch_counts_and_plain_wheel_is_unchanged(
        qtbot, screen):
    """Some platforms send Shift + wheel as a horizontal scroll."""
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    magnifier = screen._magnifier

    shift_wheel(screen, 30, 30, 120, sideways=True)
    assert magnifier.size > SIZE

    size, zoom = magnifier.size, magnifier.zoom
    shift_wheel(screen, 30, 30, 120, modifiers=Qt.NoModifier)
    assert magnifier.size == size, "a plain notch leaves the size alone"
    assert magnifier.zoom > zoom, "and still zooms the box, as in item 407"

    screen._btn_magnifier.setChecked(False)
    shift_wheel(screen, 30, 30, 120)
    assert magnifier.size == size
    assert screen._canvas.is_zoomed(), (
        "with the magnifier off the wheel is the view's, Shift or not")


def test_a_shift_wheel_step_is_proportional_and_at_least_a_pixel(
        qtbot, screen):
    magnifier = screen._magnifier
    screen._canvas.zoom_speed = 1.5
    magnifier.set_size(40)
    assert magnifier.wheel_size(True) == 60
    assert magnifier.wheel_size(False) == 32
    screen._canvas.zoom_speed = 1.001
    assert magnifier.wheel_size(True) == 33, "never a notch that does nothing"


# ---------------------------------------------------------------------------
# 2. Every settings category folds like a core application's
# ---------------------------------------------------------------------------

CATEGORIES = ("Brush", "Magic wand", "Display", "Auto-filter objects",
              "Object operations", "Cellpose-SAM", "Live magnifier")


def _categories(made):
    from spacr.qt.widgets.section import Section

    found = [w for w in made._settings_scroll.findChildren(Section)
             if w.parentWidget() is not None
             and not isinstance(w.parentWidget().parentWidget(), Section)]
    return {title: section for title, section in made._settings_categories
            if section in found}


def test_every_category_is_the_core_applications_folding_section(
        qtbot, qt_theme_applied):
    """The same widget, header and card a core module's settings use."""
    from spacr.qt.widgets import Card
    from spacr.qt.widgets.section import Section

    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        categories = _categories(made)
        assert tuple(categories) == CATEGORIES
        assert not made._settings_scroll.findChildren(Card), (
            "no unfoldable card is left on the panel")
        owners = {
            "Brush": made._brush_slider, "Magic wand": made._wand_pct,
            "Display": made._norm_hi, "Auto-filter objects":
                made._filter_min_area,
            "Object operations": made._btn_otsu,
            "Cellpose-SAM": made._cp_flow, "Live magnifier": made._mag_size,
        }
        for title, section in categories.items():
            assert type(section) is Section
            assert section.objectName() == "SectionCard"
            assert section.header().objectName() == "SectionHeader"
            assert section.header().text() == title.upper()
            assert section.is_expanded(), f"{title} starts open"
            assert section.isAncestorOf(owners[title])
            assert owners[title].isVisibleTo(made._settings_scroll)

            section.header().click()
            assert not section.is_expanded()
            assert not owners[title].isVisibleTo(made._settings_scroll), (
                f"folding {title} hides what is in it")
            section.header().click()
            assert section.is_expanded()
            assert owners[title].isVisibleTo(made._settings_scroll)
    finally:
        made._magnifier.close()
        made.close_folded()


def test_what_is_folded_is_remembered_for_the_next_visit(
        qtbot, qt_theme_applied):
    first = mm.MakeMasksScreen()
    qtbot.addWidget(first)
    try:
        cats = _categories(first)
        cats["Magic wand"].header().click()
        cats["Live magnifier"].header().click()
    finally:
        first._magnifier.close()
        first.close_folded()

    second = mm.MakeMasksScreen()
    qtbot.addWidget(second)
    try:
        cats = _categories(second)
        shut = {title for title, section in cats.items()
                if not section.is_expanded()}
        assert shut == {"Magic wand", "Live magnifier"}
        assert not second._wand_pct.isVisibleTo(second._settings_scroll)

        cats["Magic wand"].set_expanded(True)
    finally:
        second._magnifier.close()
        second.close_folded()

    third = mm.MakeMasksScreen()
    qtbot.addWidget(third)
    try:
        shut = {title for title, section in _categories(third).items()
                if not section.is_expanded()}
        assert shut == {"Live magnifier"}, "unfolding is remembered as well"
    finally:
        third._magnifier.close()
        third.close_folded()
