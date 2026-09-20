"""Item 419, points 7 to 9: the Filter category, Ctrl+click, and Invert.

The maintainer's words, 2026-09-16:

    "7. add a filter button and add this as a settings category with Minimum
    Area, Maximum Area, Minimum intensity, Maximum Intensity. these should
    default to off but if on when the user clicks the filter buton they are
    applied and red text in a text box in the Filter settings category says
    something like object 22 with area x and intensity y was removed by
    minimum intensity. with one row per removed object.

    8. if the user holds down ctrl and left clicks then the object should be
    split, if the user holds ctrl and right clicks the object that is
    hovered should be removed."

and the same day, for point 9:

    "in the cellpose-sam settings category ... there should be a new boolean
    slider (also change all booleans in this modual to the boolean sliders
    used throughout the package) invert for object detection. if this is on
    then the magnifyer shows an inverted image in the magnefication ... if
    this is on and and the cellpose-SMA detect button is pressed the masks
    are generated from the inverted image. in this case there should als obe
    a warning somewhere reminding the user that masks are generated from the
    inverted image."

EVERY GESTURE IS DRIVEN ON THE CANVAS rather than called as a method: point
8 is two mouse chords and nothing else, so a test that called
``_ctrl_edit_at`` directly would pass with the chord unreachable -- which is
instruction 52's failure, where 97 green tests sat over controls nobody
could press.

THE INVERSION IS CHECKED WHERE THE PIXELS ARE, not where the flag is. The
request's crop, the box's painted pixels and the array the detect buttons
are handed are each measured against the field, because "masks are generated
from the inverted image" is a claim about arrays; a test that asserted only
that a boolean reached a dict would hold with every one of them still
reading the field the right way up.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QCheckBox

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.theme import active_palette
from spacr.qt.widgets.toggle import Toggle
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CANVAS_H,
    CANVAS_W,
    IMG_N,
    PIXMAP_N,
    SIZE,
    CodedStub,
    canvas_xy,
    switch_on,
    wait_for_result,
)

#: A merged pair under one id, a round object, and a speck. The pair has a
#: waist to split, the round one has none, and the speck is small enough and
#: dim enough for either filter bound to take it.
PAIR, ROUND, SPECK = 4, 9, 3


def field_and_mask():
    """A field whose objects have known areas, means and shapes."""
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    image = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    mask[((xx - 16) ** 2 + (yy - 20) ** 2) < 49] = PAIR
    mask[((xx - 26) ** 2 + (yy - 20) ** 2) < 49] = PAIR
    mask[((xx - 45) ** 2 + (yy - 45) ** 2) < 36] = ROUND
    mask[2:5, 2:5] = SPECK
    image[mask == PAIR] = 800
    image[mask == ROUND] = 4000
    image[mask == SPECK] = 200
    return image, mask


@pytest.fixture
def one_field(tmp_path: Path) -> Path:
    folder = tmp_path / "field"
    (folder / "masks").mkdir(parents=True)
    image, mask = field_and_mask()
    imageio.imwrite(folder / "a.tif", image)
    imageio.imwrite(folder / "masks" / "a.tif", mask)
    return folder


@pytest.fixture
def dark_field(tmp_path: Path) -> Path:
    """Two DARK discs on a bright ground: what Invert exists for."""
    folder = tmp_path / "dark"
    (folder / "masks").mkdir(parents=True)
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    image = np.full((IMG_N, IMG_N), 50_000, dtype=np.uint16)
    image[((xx - 20) ** 2 + (yy - 20) ** 2) < 64] = 2_000
    image[((xx - 44) ** 2 + (yy - 44) ** 2) < 64] = 2_000
    imageio.imwrite(folder / "a.tif", image)
    imageio.imwrite(folder / "masks" / "a.tif",
                    np.zeros((IMG_N, IMG_N), dtype=np.uint16))
    return folder


def _built(qtbot, folder: Path):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(folder))
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    assert made._canvas.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    made._mag_size.setValue(SIZE)
    return made


@pytest.fixture
def screen(qtbot, qt_theme_applied, one_field: Path):
    made = _built(qtbot, one_field)
    yield made
    made._magnifier.close()
    made.close_folded()


@pytest.fixture
def dark_screen(qtbot, qt_theme_applied, dark_field: Path):
    made = _built(qtbot, dark_field)
    yield made
    made._magnifier.close()
    made.close_folded()


def _mouse(kind, x, y, button=Qt.NoButton, buttons=Qt.NoButton,
           modifiers=Qt.NoModifier):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def ctrl_click(screen, img_x, img_y, button=Qt.LeftButton):
    """Press and release ``button`` with Ctrl held, on image pixel (x, y)."""
    x, y = canvas_xy(img_x, img_y)
    screen._canvas.mousePressEvent(_mouse(
        QEvent.Type.MouseButtonPress, x, y, button, button,
        Qt.ControlModifier))
    screen._canvas.mouseReleaseEvent(_mouse(
        QEvent.Type.MouseButtonRelease, x, y, button, Qt.NoButton,
        Qt.ControlModifier))


def ids_of(mask) -> list:
    return sorted(int(v) for v in np.unique(np.asarray(mask)) if v)


def log_rows(screen) -> list:
    text = screen._filter_log.toPlainText()
    return [row for row in text.splitlines() if row.strip()]


def category(screen, title: str):
    """The settings category called ``title``, as the panel built it."""
    found = dict(screen._settings_categories)
    assert title in found, f"no settings category called {title!r}: {list(found)}"
    return found[title]


# ---------------------------------------------------------------------------
# 7. The Filter category
# ---------------------------------------------------------------------------

def test_the_category_is_called_filter_and_holds_the_four_bounds(screen):
    card = category(screen, "Filter")
    for box in (screen._filter_min_area, screen._filter_max_area,
                screen._filter_min_int, screen._filter_max_int,
                screen._btn_filter, screen._filter_log):
        assert card.isAncestorOf(box), f"{box} is not in the Filter category"


def test_the_filter_button_is_called_filter(screen):
    assert screen._btn_filter.text() == "Filter"


def test_every_bound_starts_off(screen):
    """"these should default to off" -- and 0 is what off is."""
    assert screen._filter_min_area.value() == 0
    assert screen._filter_max_area.value() == 0
    assert screen._filter_min_int.value() == 0.0
    assert screen._filter_max_int.value() == 0.0
    before = ids_of(screen._canvas.mask)
    screen._btn_filter.click()
    assert ids_of(screen._canvas.mask) == before
    assert log_rows(screen) == []


def test_a_layout_that_folded_the_old_category_folds_the_new_one(
        qtbot, qt_theme_applied, monkeypatch):
    """A user who folded Auto-filter objects away keeps it folded."""
    from spacr.qt import preferences

    monkeypatch.setattr(
        preferences, "get_section_layout",
        lambda panel: {"folded": ["Auto-filter objects"]})
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        folded = {title for title, section in made._settings_categories
                  if not section.is_expanded()}
        assert folded == {"Filter"}
    finally:
        made._magnifier.close()
        made.close_folded()


def test_only_the_bounds_that_are_on_are_applied(screen):
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    assert ids_of(screen._canvas.mask) == [PAIR, ROUND]


def test_every_removed_object_gets_its_own_row(screen):
    screen._filter_min_int.setValue(1_000.0)
    screen._btn_filter.click()
    rows = log_rows(screen)
    assert len(rows) == 2
    assert [row.split()[1] for row in rows] == [str(SPECK), str(PAIR)]


def test_a_row_names_the_object_its_area_its_intensity_and_the_bound(screen):
    lookup = engine.ObjectLookup(screen._canvas.mask, screen._canvas.image)
    area, mean = lookup.measure(SPECK)
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    row = log_rows(screen)[0]
    assert row == (f"Object {SPECK} with area {area} px and intensity "
                   f"{mean:.2f} was removed by minimum area 20")


def test_a_row_names_both_bounds_when_an_object_misses_on_two(screen):
    screen._filter_min_area.setValue(20)
    screen._filter_min_int.setValue(500.0)
    screen._btn_filter.click()
    row = next(r for r in log_rows(screen) if r.startswith(f"Object {SPECK} "))
    assert row.endswith(
        "was removed by minimum area 20 and minimum intensity 500.00")


def test_the_rows_are_the_ids_the_readout_shows(screen):
    """Point 1's readout and point 7's ledger name the same object."""
    screen._canvas.update_readout(QPointF(*canvas_xy(3, 3)), measure=True)
    readout = screen._canvas.readout
    assert readout is not None and readout.label == SPECK
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    assert log_rows(screen)[0].startswith(f"Object {readout.label} ")
    assert f"area {readout.area} px" in log_rows(screen)[0]
    assert f"intensity {readout.mean_intensity:.2f}" in log_rows(screen)[0]


def test_a_run_that_removes_nothing_clears_the_last_run_s_rows(screen):
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    assert log_rows(screen)
    screen._on_undo()
    screen._filter_min_area.setValue(1)
    screen._btn_filter.click()
    assert log_rows(screen) == []
    assert "nothing outside the bounds" in screen._status_label.text()


def test_the_rows_go_when_the_next_field_arrives(qtbot, screen, one_field):
    """They name objects in the mask ON SCREEN, so they cannot outlive it."""
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    assert log_rows(screen)
    screen._filter_min_area.setValue(0)
    image, mask = field_and_mask()
    imageio.imwrite(one_field / "b.tif", image)
    imageio.imwrite(one_field / "masks" / "b.tif", mask)
    assert screen._open_folder(str(one_field))
    qtbot.waitUntil(lambda: screen._canvas.mask is not None, timeout=5_000)
    assert log_rows(screen) == []


def test_the_rows_of_the_field_on_screen_are_that_fields_own(qtbot, screen,
                                                              one_field):
    """A bound left on re-runs on the new field and lists ITS objects."""
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    first = log_rows(screen)
    image, mask = field_and_mask()
    imageio.imwrite(one_field / "b.tif", image)
    imageio.imwrite(one_field / "masks" / "b.tif", mask)
    assert screen._open_folder(str(one_field))
    qtbot.waitUntil(lambda: screen._canvas.mask is not None, timeout=5_000)
    assert log_rows(screen) == first, "the rows accumulated instead of being remade"


def test_the_rows_are_the_themes_error_colour(qtbot, screen):
    """"red text", and the theme's red rather than a literal one."""
    screen.show()
    qtbot.waitExposed(screen)
    ink = screen._filter_log.palette().color(
        screen._filter_log.foregroundRole()).name()
    assert ink.lower() == active_palette()["error"].lower()


def test_the_filter_is_one_undo_step_and_one_ledger_entry(screen):
    before = ids_of(screen._canvas.mask)
    written = len(screen._log.edits)
    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    assert len(screen._log.edits) == written + 1
    assert screen._log.edits[-1].kind == "filter"
    screen._on_undo()
    assert ids_of(screen._canvas.mask) == before


# ---------------------------------------------------------------------------
# 8. Ctrl + left click splits, Ctrl + right click removes
# ---------------------------------------------------------------------------

def test_ctrl_left_click_splits_the_object_under_the_cursor(screen):
    ctrl_click(screen, 16, 20)
    assert ids_of(screen._canvas.mask) == [SPECK, PAIR, ROUND, 10]
    assert int(screen._canvas.mask[20, 16]) != int(screen._canvas.mask[20, 26])


def test_the_split_keeps_every_pixel_of_the_object(screen):
    before = screen._canvas.mask.copy()
    ctrl_click(screen, 16, 20)
    np.testing.assert_array_equal(screen._canvas.mask > 0, before > 0)


def test_ctrl_right_click_removes_the_object_under_the_cursor(screen):
    ctrl_click(screen, 45, 45, button=Qt.RightButton)
    assert ids_of(screen._canvas.mask) == [SPECK, PAIR]


def test_each_is_one_undo_step(screen):
    before = ids_of(screen._canvas.mask)
    ctrl_click(screen, 16, 20)
    assert ids_of(screen._canvas.mask) != before
    screen._on_undo()
    assert ids_of(screen._canvas.mask) == before
    ctrl_click(screen, 45, 45, button=Qt.RightButton)
    assert ids_of(screen._canvas.mask) != before
    screen._on_undo()
    assert ids_of(screen._canvas.mask) == before


def test_each_writes_one_ledger_entry_naming_the_object(screen):
    written = len(screen._log.edits)
    ctrl_click(screen, 16, 20)
    split = screen._log.edits[-1]
    assert (split.kind, split.target) == ("split", PAIR)
    assert split.detail["into"] == [10]
    ctrl_click(screen, 45, 45, button=Qt.RightButton)
    gone = screen._log.edits[-1]
    assert (gone.kind, gone.target) == ("delete", ROUND)
    assert len(screen._log.edits) == written + 2


def test_a_ctrl_click_on_background_changes_nothing_and_says_so(screen):
    before = screen._canvas.mask.copy()
    written = len(screen._log.edits)
    ctrl_click(screen, 60, 8)
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert len(screen._log.edits) == written
    assert "there is none under the cursor" in screen._status_label.text()


def test_an_object_with_one_centre_is_not_split_and_the_line_says_why(screen):
    before = screen._canvas.mask.copy()
    ctrl_click(screen, 45, 45)
    np.testing.assert_array_equal(screen._canvas.mask, before)
    text = screen._status_label.text()
    assert f"Object {ROUND} has one centre" in text
    assert "Divide" in text


def test_min_area_reaches_the_split(screen):
    """The box that says what debris is says what is too small to be two."""
    screen._min_area.setValue(4_000)
    assert screen._canvas.split_min_area == 4_000
    before = screen._canvas.mask.copy()
    ctrl_click(screen, 16, 20)
    np.testing.assert_array_equal(screen._canvas.mask, before)
    screen._min_area.setValue(0)
    ctrl_click(screen, 16, 20)
    assert ids_of(screen._canvas.mask) == [SPECK, PAIR, ROUND, 10]


def test_both_gestures_work_with_the_magnifier_on(qtbot, screen):
    """They are checked before the magnifier, which takes every click."""
    stub = CodedStub({77: (0, 0, 4, 4)})
    switch_on(screen, stub)
    screen._magnifier.hover(QPointF(*canvas_xy(16, 20)))
    ctrl_click(screen, 16, 20)
    assert ids_of(screen._canvas.mask) == [SPECK, PAIR, ROUND, 10]
    assert 77 not in ids_of(screen._canvas.mask), \
        "the magnifier committed its own objects on the way back up"
    ctrl_click(screen, 45, 45, button=Qt.RightButton)
    assert ROUND not in ids_of(screen._canvas.mask)


def test_the_two_gestures_are_on_the_shortcut_panel(screen):
    keys = dict(mm.SHORTCUT_HINTS)
    assert "Split" in keys["Ctrl + left click"]
    assert "Remove" in keys["Ctrl + right click"]
    for line in ("Ctrl + left click", "Ctrl + right click"):
        assert line in screen._shortcut_rows
        assert screen._shortcut_rows[line][0].text() == line


# ---------------------------------------------------------------------------
# 9a. Every boolean is the package's slider
# ---------------------------------------------------------------------------

def test_every_boolean_on_this_screen_is_the_packages_slider(screen):
    plain = [box for box in screen.findChildren(QCheckBox)
             if not isinstance(box, Toggle)]
    assert plain == [], [box.text() for box in plain]


def test_the_sliders_still_read_and_write_as_checkboxes(screen):
    """A Toggle is a QCheckBox subclass, so nothing wired to one moved."""
    for box in (screen._otsu_bright, screen._cp_normalize,
                screen._otsu_fill_holes, screen._otsu_split,
                screen._cp_invert):
        was = box.isChecked()
        box.setChecked(not was)
        assert box.isChecked() is (not was)
        box.setChecked(was)


# ---------------------------------------------------------------------------
# 9b, 9e. Invert, and the warning it puts on the screen
# ---------------------------------------------------------------------------

def test_invert_is_in_the_object_detection_category(screen):
    assert screen._cp_invert.text() == "Invert"
    assert category(screen, "Object detection").isAncestorOf(screen._cp_invert)


def test_invert_starts_off_and_the_warning_with_it(screen):
    assert screen._cp_invert.isChecked() is False
    assert screen._invert_warning.isHidden()


def test_the_warning_is_up_while_invert_is_on(screen):
    screen._cp_invert.setChecked(True)
    assert not screen._invert_warning.isHidden()
    assert "INVERTED" in screen._invert_warning.text()
    screen._cp_invert.setChecked(False)
    assert screen._invert_warning.isHidden()


def test_the_warning_is_not_inside_the_settings_panel(screen):
    """The Settings toggle hides that panel; the magnifier goes on inverting."""
    screen._cp_invert.setChecked(True)
    assert not screen._settings_scroll.isAncestorOf(screen._invert_warning)
    screen._btn_settings.setChecked(False)
    assert not screen._invert_warning.isHidden()


def test_the_warning_wears_the_themes_warning_colour(qtbot, screen):
    screen._cp_invert.setChecked(True)
    screen.show()
    qtbot.waitExposed(screen)
    ink = screen._invert_warning.palette().color(
        screen._invert_warning.foregroundRole()).name()
    assert ink.lower() == active_palette()["warning"].lower()


# ---------------------------------------------------------------------------
# 9c, 9d. The inversion is what the detector sees
# ---------------------------------------------------------------------------

def test_the_magnifier_is_handed_the_inverted_region(qtbot, screen):
    stub = CodedStub({})
    switch_on(screen, stub)
    screen._magnifier.hover(QPointF(*canvas_xy(30, 30)))
    wait_for_result(qtbot, screen)
    plain = stub.calls[-1]
    x0, y0, x1, y1 = plain.box
    np.testing.assert_array_equal(
        plain.crop, screen._canvas.image[y0:y1, x0:x1])
    assert plain.invert is False

    screen._cp_invert.setChecked(True)
    screen._magnifier.hover(QPointF(*canvas_xy(30, 30)))
    qtbot.waitUntil(lambda: stub.calls[-1].invert, timeout=10_000)
    wait_for_result(qtbot, screen)
    inverted = stub.calls[-1]
    assert inverted.box == plain.box
    field = screen._canvas.image
    np.testing.assert_array_equal(
        inverted.crop, engine.invert_intensity(field)[y0:y1, x0:x1])


def test_the_region_is_inverted_about_the_whole_fields_range(screen):
    """So the box previews the button instead of disagreeing with it."""
    screen._cp_invert.setChecked(True)
    field = screen._canvas.image
    x0, y0, x1, y1 = 9, 13, 25, 29
    crop = field[y0:y1, x0:x1]
    assert int(crop.max()) < int(field.max()), \
        "the box must not hold the field's brightest pixel, or the two " \
        "rules would agree by accident"
    region = screen._magnifier.region_for((x0, y0, x1, y1), invert=True)
    np.testing.assert_array_equal(
        region, engine.invert_intensity(field)[y0:y1, x0:x1])
    assert not np.array_equal(region, engine.invert_intensity(crop)), \
        "the region was inverted about its own extremes, not the field's"


def test_the_request_key_carries_invert(screen):
    screen._magnifier.set_enabled(True)
    screen._magnifier.hover(QPointF(*canvas_xy(30, 30)))
    off = screen._magnifier.build_request()
    screen._cp_invert.setChecked(True)
    on = screen._magnifier.build_request()
    assert off is not None and on is not None
    assert off.key != on.key
    assert on.key[-2] is True and off.key[-2] is False


def test_the_box_paints_the_inverted_region(qtbot, screen):
    """Point 9c: what is inside the magnification, in pixels."""
    from PySide6.QtGui import QImage, QPainter

    screen._magnifier.set_enabled(True)
    screen._magnifier.hover(QPointF(*canvas_xy(32, 32)))

    def painted() -> QImage:
        picture = QImage(CANVAS_W, CANVAS_H, QImage.Format_RGB32)
        picture.fill(0)
        painter = QPainter(picture)
        screen._magnifier.paint(painter)
        painter.end()
        return picture

    plain = painted()
    screen._cp_invert.setChecked(True)
    inverted = painted()
    assert plain != inverted, "the box shows the same pixels either way"

    lens = screen._magnifier.lens_geometry()[1]
    middle = lens.center().toPoint()
    assert (QImage(plain).pixelColor(middle).value()
            != QImage(inverted).pixelColor(middle).value())


def test_otsu_detect_reads_the_inverted_image(dark_screen):
    """Point 9d, on the detector Invert was asked for."""
    dark_screen._min_area.setValue(20)
    dark_screen._btn_otsu.click()
    plain = dark_screen._canvas.mask.copy()
    dark_screen._on_undo()

    dark_screen._cp_invert.setChecked(True)
    dark_screen._btn_otsu.click()
    inverted = dark_screen._canvas.mask
    assert int(inverted[20, 20]) and int(inverted[44, 44])
    assert int(inverted[20, 20]) != int(inverted[44, 44]), \
        "the two dark discs came back as one object"
    assert len(ids_of(inverted)) > len(ids_of(plain))
    assert "INVERTED" in dark_screen._status_label.text()


def test_otsu_detect_records_which_way_up_the_image_was(dark_screen):
    dark_screen._min_area.setValue(20)
    dark_screen._cp_invert.setChecked(True)
    dark_screen._btn_otsu.click()
    entry = dark_screen._log.edits[-1]
    assert entry.kind == "detect" and entry.detail["invert"] is True


def test_object_detection_is_handed_the_inverted_image(screen, monkeypatch):
    """The array the model is given, not the flag that was set."""
    seen = []

    def record(image, model, **kwargs):
        seen.append(np.array(image, copy=True))
        return np.zeros(image.shape, dtype=np.int32), None, None

    monkeypatch.setattr(screen, "_cellpose_model", lambda name: object())
    monkeypatch.setattr(mm, "cellpose_detect", record)
    screen.run_cellpose()
    np.testing.assert_array_equal(seen[-1], screen._canvas.image)

    screen._cp_invert.setChecked(True)
    screen.run_cellpose()
    np.testing.assert_array_equal(
        seen[-1], engine.invert_intensity(screen._canvas.image))


def test_the_readout_and_the_filter_keep_the_fields_real_values(screen):
    """Point 9's own note: a user filtering by intensity judges real numbers."""
    screen._canvas.update_readout(QPointF(*canvas_xy(3, 3)), measure=True)
    plain = screen._canvas.readout

    screen._cp_invert.setChecked(True)
    screen._canvas.update_readout(QPointF(*canvas_xy(3, 3)), measure=True)
    assert screen._canvas.readout == plain

    screen._filter_min_area.setValue(20)
    screen._btn_filter.click()
    assert f"intensity {plain.mean_intensity:.2f}" in log_rows(screen)[0]


def test_the_canvas_array_is_never_inverted_in_place(screen):
    before = screen._canvas.image.copy()
    screen._cp_invert.setChecked(True)
    screen._magnifier.set_enabled(True)
    screen._magnifier.hover(QPointF(*canvas_xy(30, 30)))
    screen._magnifier.build_request()
    screen._detector_image()
    np.testing.assert_array_equal(screen._canvas.image, before)
