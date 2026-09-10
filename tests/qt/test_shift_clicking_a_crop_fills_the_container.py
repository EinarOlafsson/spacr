"""Shift + left click blows one crop up over the grid, and folds it back.

An EXTRA gesture. Annotating is the primary action on this screen and it
happens with a plain left click on the crop, so the zoom had to be reachable
without taking that click away -- which is the first thing measured here.

Everything is driven with real mouse events on the real widgets:
``QTest.mouseClick`` on the tile with ``ShiftModifier`` held, and on the
overlay at a point inside and a point outside the picture. Calling the
handlers directly would prove the handlers work and say nothing about
whether the gesture reaches them.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest


@pytest.fixture
def annotate_source(tmp_path: Path) -> Path:
    """A folder of eight crops with a ``png_list`` that points at them."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    rng = np.random.default_rng(3)
    paths = []
    for i in range(8):
        array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        png = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(array).save(png)
        paths.append(str(png))
    conn = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
        conn.commit()
    finally:
        conn.close()
    return src


@pytest.fixture
def open_annotate(qtbot, qt_theme_applied, annotate_source, monkeypatch):
    """Open the screen on the synthetic source, with its page decoded.

    ``size`` fixes the window, which is what decides how many crops fit on
    a page: a small one leaves more than one page to turn.
    """
    from spacr.qt import ai as ai_module
    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    from spacr.qt.screens.annotate import AnnotateScreen

    opened = []

    def _open(size=None):
        screen = AnnotateScreen()
        qtbot.addWidget(screen)
        if size is not None:
            screen.setFixedSize(*size)
        screen.show()
        qtbot.waitExposed(screen)
        screen._open_source(str(annotate_source))
        qtbot.waitUntil(lambda: len(screen._page_paths) >= 1, timeout=10000)
        qtbot.waitUntil(
            lambda: screen._thumb_pixmaps[0] is not None
            and not screen._thumb_pixmaps[0].isNull(), timeout=10000)
        opened.append(screen)
        return screen

    yield _open
    for screen in opened:
        if screen._worker:
            screen._worker.stop(wait=True)


def _centre(widget) -> QPoint:
    return QPoint(widget.width() // 2, widget.height() // 2)


def test_shift_click_fills_the_container_in_front_of_the_grid(open_annotate):
    """The crop takes over the container, and it is drawn over the others."""
    from spacr.qt.theme import SPACING

    screen = open_annotate()
    tile = screen._thumbs[0]
    overlay = screen._zoom_overlay
    viewport = screen._grid_scroll.viewport()
    tiles_before = len(screen._thumbs)
    assert not overlay.isVisible()

    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))

    assert overlay.isVisible(), "shift + left click did not open the crop"
    assert overlay.slot == 0
    assert overlay.geometry() == viewport.rect(), (
        "the zoomed crop does not fill the grid's container")

    # FILLS IT, at the crop's own aspect ratio: one of the two axes is as
    # large as the container allows.
    margin = float(SPACING["md"])
    picture = overlay.picture_rect()
    fills = (abs(picture.width() - (overlay.width() - 2 * margin)) < 1.0
             or abs(picture.height() - (overlay.height() - 2 * margin)) < 1.0)
    assert fills, (
        f"the crop was drawn {picture.width():.0f}x{picture.height():.0f} in "
        f"a {overlay.width()}x{overlay.height()} container")
    assert picture.width() > tile.width(), (
        f"the crop was drawn at {picture.width():.0f}px, no bigger than the "
        f"{tile.width()}px tile it came from")

    # IN FRONT, not instead of: the grid keeps its tiles and its geometry,
    # and the overlay is what the container hands a click at its centre to.
    assert len(screen._thumbs) == tiles_before
    assert viewport.childAt(viewport.rect().center()) is overlay


def test_shift_click_does_not_also_label_the_crop(open_annotate):
    """The gesture that opens a crop must not annotate it on the way in."""
    screen = open_annotate()
    tile = screen._thumbs[0]
    before = screen._page_paths[0][1]

    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))

    assert screen._page_paths[0][1] == before, (
        "shift-clicking a crop labelled it as well as opening it")


def test_plain_left_click_still_annotates_and_opens_nothing(open_annotate):
    """The primary action is untouched: this is an extra gesture."""
    screen = open_annotate()
    tile = screen._thumbs[0]

    QTest.mouseClick(tile, Qt.LeftButton, Qt.NoModifier, _centre(tile))

    assert screen._page_paths[0][1] == 1, (
        "a plain left click stopped assigning class 1")
    assert not screen._zoom_overlay.isVisible()


def test_a_click_beside_the_picture_folds_it_back(open_annotate):
    """Clicking outside is the way out; clicking the picture is not."""
    screen = open_annotate()
    tile = screen._thumbs[0]
    overlay = screen._zoom_overlay
    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))
    assert overlay.isVisible()

    picture = overlay.picture_rect()
    inside = QPoint(int(picture.center().x()), int(picture.center().y()))
    QTest.mouseClick(overlay, Qt.LeftButton, Qt.NoModifier, inside)
    assert overlay.isVisible(), (
        "a click on the picture itself folded it back")

    QTest.mouseClick(overlay, Qt.LeftButton, Qt.NoModifier, QPoint(1, 1))
    assert not overlay.isVisible(), (
        "a click beside the picture did not fold it back")


def test_escape_folds_the_crop_back(open_annotate):
    """The keyboard way out, on the widget the gesture left focused."""
    screen = open_annotate()
    tile = screen._thumbs[0]
    overlay = screen._zoom_overlay
    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))
    assert overlay.isVisible()

    QTest.keyClick(overlay, Qt.Key_Escape)
    assert not overlay.isVisible()


def test_turning_the_page_folds_the_crop_back(open_annotate, qtbot):
    """A blown-up crop belongs to the page it came from."""
    screen = open_annotate(size=(460, 420))
    assert screen._settings.page_size < 8, (
        "the whole population fits on one page, so there is nothing to turn")
    tile = screen._thumbs[0]
    overlay = screen._zoom_overlay
    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))
    assert overlay.isVisible()

    qtbot.waitUntil(lambda: not screen.is_busy() and screen.active_jobs() == 0,
                    timeout=20000)
    start = screen._offset
    screen._btn_next.click()
    assert screen._offset > start, "the Next control did not turn the page"
    assert not overlay.isVisible(), (
        "the crop from the previous page is still drawn over the new one")


def test_a_cell_with_no_crop_in_it_opens_nothing(open_annotate):
    """An empty overlay a user then has to dismiss is worse than no gesture."""
    screen = open_annotate()
    empty = len(screen._thumbs) - 1
    screen._set_slot_image(empty, None)
    tile = screen._thumbs[empty]

    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))

    assert not screen._zoom_overlay.isVisible()


def test_the_keyboard_still_labels_the_crop_that_is_open(open_annotate):
    """Looking closely is not a mode: the class keys still land.

    The overlay takes focus so ``Escape`` reaches it, and passes on every
    other key -- so a crop opened because the annotator could not tell what
    it was can be labelled the moment they can.
    """
    screen = open_annotate()
    tile = screen._thumbs[0]
    overlay = screen._zoom_overlay
    QTest.mouseClick(tile, Qt.LeftButton, Qt.ShiftModifier, _centre(tile))
    assert overlay.isVisible()

    QTest.keyClick(overlay, Qt.Key_1)

    assert screen._page_paths[0][1] == 1, (
        "the class keys stopped working while a crop was open")
    assert overlay.isVisible(), "labelling closed the crop"
