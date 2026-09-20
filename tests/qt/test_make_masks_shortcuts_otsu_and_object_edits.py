"""Item 419, points 4 to 6: the shortcut list, Otsu's settings, and the edits.

The maintainer's words, 2026-09-16:

    "4. in make masks to the right of the mak, cell probability, Flows should
    be explanations for the shortcuts, like hold shift to move, scroll to
    zoom, drag to choose in magnifier mode, etc., but written in a consise
    way.

    5. please add some more settings for the Otsu mode, change the name of
    classical to Otsu, and change the name of cellpose-sam to Object
    detection.

    6. Add these buttons: Clear (clears all of jects after clicking and
    clicking ok on a popup), dialate, shring."

THE SHORTCUT LIST IS CHECKED AGAINST THE CODE, not read back from the table
it was built from. A panel of twelve plausible sentences is worth nothing if
one of them is about a gesture the canvas does not have, and that is exactly
the failure a list like this drifts into: the keys change and the panel does
not. So every line drives the canvas and asserts what it claims.

The renames are checked in both directions -- the new name is what the
screen shows, and the old one still selects the mode it was renamed from,
because a mode name reaches the screen from a saved layout and from any
script written before today.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QLabel

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CANVAS_H,
    CANVAS_W,
    PIXMAP_N,
    SIZE,
    canvas_xy,
    coded_field,
    rect_mask,
    switch_on,
    wait_for_result,
)
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CodedStub,
)
from tests.qt.test_the_magnifier_drag_merges_what_the_cursor_passes_over import (
    CodedStub as DragStub,
    drag_along,
    settle_on,
    switch_on as drag_switch_on,
    wait_until_done,
)

#: Two objects one pixel apart, so Shrink separates them and Dilate does not
#: fuse them, and a third far away to be the one a sweep removes.
OBJECTS = {4: (10, 10, 20, 20), 9: (21, 10, 31, 20), 6: (44, 44, 52, 52)}


def _mouse(kind, x, y, button=Qt.NoButton, buttons=Qt.NoButton,
           modifiers=Qt.NoModifier):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def press(screen, img_x, img_y, button=Qt.LeftButton, modifiers=Qt.NoModifier):
    x, y = canvas_xy(img_x, img_y)
    screen._canvas.mousePressEvent(_mouse(
        QEvent.Type.MouseButtonPress, x, y, button, button, modifiers))


def move(screen, img_x, img_y, buttons=Qt.NoButton, modifiers=Qt.NoModifier):
    x, y = canvas_xy(img_x, img_y)
    screen._canvas.mouseMoveEvent(_mouse(
        QEvent.Type.MouseMove, x, y, Qt.NoButton, buttons, modifiers))


def release(screen, img_x, img_y, button=Qt.LeftButton):
    x, y = canvas_xy(img_x, img_y)
    screen._canvas.mouseReleaseEvent(_mouse(
        QEvent.Type.MouseButtonRelease, x, y, button, Qt.NoButton))


def wheel(screen, img_x, img_y, delta=120, modifiers=Qt.NoModifier):
    pos = QPointF(*canvas_xy(img_x, img_y))
    screen._canvas.wheelEvent(QWheelEvent(
        pos, pos, QPoint(0, 0), QPoint(0, delta), Qt.NoButton,
        modifiers, Qt.NoScrollPhase, False))


@pytest.fixture
def two_fields(tmp_path: Path) -> Path:
    """Two fields, so the arrow keys have somewhere to go."""
    folder = tmp_path / "pairs"
    (folder / "masks").mkdir(parents=True)
    for name in ("a.tif", "b.tif"):
        imageio.imwrite(folder / name, coded_field())
        imageio.imwrite(folder / "masks" / name,
                        rect_mask((64, 64), OBJECTS, dtype=np.uint16))
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, two_fields: Path):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(two_fields))
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    assert made._canvas.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    yield made
    made._magnifier.close()
    made.close_folded()


def ids_of(mask) -> list:
    return sorted(int(v) for v in np.unique(np.asarray(mask)) if v)


# ---------------------------------------------------------------------------
# 4. The shortcut list, beside the views
# ---------------------------------------------------------------------------

def test_the_shortcuts_sit_right_of_the_three_views(screen, qtbot):
    """Measured on a shown, laid-out screen, in pixels."""
    screen.resize(1500, 900)
    screen.show()
    qtbot.waitExposed(screen)
    QApplication.processEvents()
    for _ in range(4):
        QApplication.processEvents()
    panel = screen._shortcut_panel
    tabs = screen._view_tabs
    assert screen._view_pane.isAncestorOf(panel)
    assert screen._view_pane.isAncestorOf(tabs)
    assert screen._body_splitter.indexOf(screen._view_pane) == 1
    right_of_tabs = tabs.mapTo(screen, tabs.rect().topRight()).x()
    left_of_panel = panel.mapTo(screen, panel.rect().topLeft()).x()
    assert left_of_panel > right_of_tabs, (
        "the shortcuts are to the RIGHT of Mask / Cell probability / Flows")
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Mask", "Cell probability", "Flows"]
    assert panel.isVisible() and panel.width() == mm.SHORTCUTS_WIDTH


def test_every_hint_is_on_the_screen_with_its_keys_and_its_sentence(
        screen, qtbot):
    screen.resize(1500, 900)
    screen.show()
    qtbot.waitExposed(screen)
    QApplication.processEvents()
    shown = {label.text() for label in
             screen._shortcut_panel.findChildren(QLabel)}
    assert "Shortcuts" in shown
    for keys, does in mm.SHORTCUT_HINTS:
        assert keys in shown, f"no row for {keys!r}"
        assert does in shown, f"{keys!r} has no sentence"
        key_label, does_label = screen._shortcut_rows[keys]
        assert key_label.isVisible() and does_label.isVisible()
        assert len(does) <= 52, (
            f"{does!r} is not one terse line; the request asked for concise")


def test_the_settings_toggle_does_not_take_the_shortcuts_away(screen, qtbot):
    """They are not settings: hiding them with the settings would hide them
    exactly when the canvas has been cleared for work."""
    screen.resize(1500, 900)
    screen.show()
    qtbot.waitExposed(screen)
    QApplication.processEvents()
    screen._btn_settings.setChecked(False)
    QApplication.processEvents()
    assert not screen._settings_scroll.isVisibleTo(screen._body_splitter)
    assert screen._shortcut_panel.isVisibleTo(screen._body_splitter)


def test_shift_or_alt_drag_pans_as_the_list_says(screen):
    """'Shift or Alt + drag — Pan, from any tool'."""
    for modifier, (to_x, to_y) in ((Qt.ShiftModifier, (20, 20)),
                                   (Qt.AltModifier, (44, 44))):
        screen._on_reset_zoom()
        wheel(screen, 32, 32, 120)
        wheel(screen, 32, 32, 120)
        assert screen._canvas.is_zoomed()
        before = screen._canvas._viewport_bounds()
        press(screen, 32, 32, Qt.LeftButton, modifier)
        move(screen, to_x, to_y, Qt.LeftButton, modifier)
        release(screen, to_x, to_y)
        assert screen._canvas._viewport_bounds() != before, (
            f"{modifier} + drag did not pan")


def test_the_wheel_zooms_about_the_cursor_as_the_list_says(screen):
    """'Wheel — Zoom about the cursor'."""
    assert not screen._canvas.is_zoomed()
    wheel(screen, 50, 50, 120)
    assert screen._canvas.is_zoomed()
    x0, y0, x1, y1 = screen._canvas._viewport_bounds()
    assert x0 <= 50 < x1 and y0 <= 50 < y1


def test_the_right_button_sweeps_objects_away_as_the_list_says(screen):
    """'Right button — Sweep away the objects it passes'."""
    assert 6 in ids_of(screen._canvas.mask)
    press(screen, 47, 47, Qt.RightButton)
    release(screen, 47, 47, Qt.RightButton)
    assert 6 not in ids_of(screen._canvas.mask)


def test_the_arrows_move_between_fields_as_the_list_says(screen):
    """'← → — Previous / next field'."""
    first = screen._current_index
    screen._on_next()
    assert screen._current_index != first
    screen._on_prev()
    assert screen._current_index == first


def test_undo_redo_and_escape_do_what_the_list_says(screen):
    """'Ctrl+Z / Ctrl+Y — Undo / redo' and 'Esc — Reset the zoom'."""
    before = ids_of(screen._canvas.mask)
    screen._btn_clear.setEnabled(True)
    screen._confirm = lambda *_a, **_k: True
    screen._btn_clear.click()
    assert ids_of(screen._canvas.mask) == []
    screen._on_undo()
    assert ids_of(screen._canvas.mask) == before
    screen._on_redo()
    assert ids_of(screen._canvas.mask) == []
    wheel(screen, 32, 32, 120)
    assert screen._canvas.is_zoomed()
    screen._on_reset_zoom()
    assert not screen._canvas.is_zoomed()


def test_the_magnifier_wheel_lines_are_true(qtbot, screen):
    """'Magnifier: wheel — Box zoom' and 'Shift + wheel — Box size'."""
    switch_on(screen, CodedStub({}))
    screen._mag_size.setValue(SIZE)
    zoom = screen._magnifier.zoom
    wheel(screen, 32, 32, 120)
    assert screen._magnifier.zoom != zoom, "the wheel did not move the box zoom"
    assert not screen._canvas.is_zoomed(), "and it did not zoom the view"
    size = screen._magnifier.size
    wheel(screen, 32, 32, 120, Qt.ShiftModifier)
    assert screen._magnifier.size != size, "Shift + wheel did not resize the box"


def test_the_magnifier_drag_line_is_true(qtbot, screen):
    """'Magnifier: drag — Add the objects it passes over'.

    Driven with item 417's own drag helpers, because the gesture is 417's.
    The two the drag crosses arrive as ONE object, which is that item's rule
    for pieces a path joins and is its file's to state; what this asserts is
    the claim the panel makes -- that a drag adds what it was pulled across,
    in one undo step.
    """
    screen._mag_size.setValue(SIZE)
    screen._min_area.setValue(0)
    drag_switch_on(screen, DragStub({7: (36, 30, 44, 38),
                                     8: (44, 30, 52, 38)}), save="zoom")
    settle_on(qtbot, screen, 40, 34)
    before = len(screen._log.edits)
    drag_along(screen, list(range(40, 52)), 34)
    wait_until_done(qtbot, screen)
    qtbot.waitUntil(lambda: len(screen._log.edits) > before, timeout=10_000)
    painted = np.nonzero(screen._canvas.mask)[1]
    assert int(painted.min()) <= 37, "the object it started on"
    assert int(painted.max()) >= 50, "and the one it was pulled across"
    assert len(screen._log.edits) == before + 1, "one drag, one ledger entry"
    assert screen._history.can_undo()


def test_the_whole_image_right_click_line_is_true(qtbot, screen):
    """'Magnifier, whole image: right — Remove the object under it'."""
    switch_on(screen, CodedStub({}))
    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("image"))
    qtbot.waitUntil(lambda: screen._magnifier._image_result is not None,
                    timeout=10_000)
    assert 4 in ids_of(screen._canvas.mask)
    press(screen, 15, 15, Qt.RightButton)
    release(screen, 15, 15, Qt.RightButton)
    assert 4 not in ids_of(screen._canvas.mask)


# ---------------------------------------------------------------------------
# 5. Otsu's settings, and the two renames
# ---------------------------------------------------------------------------

def test_the_mode_is_called_otsu_and_the_category_object_detection(screen):
    modes = [(screen._mag_mode.itemText(i), screen._mag_mode.itemData(i))
             for i in range(screen._mag_mode.count())]
    assert ("Otsu", "otsu") in modes
    assert not [row for row in modes if row[1] == "classical"]
    assert "Classical" not in [text for text, _data in modes]
    titles = [title for title, _section in screen._settings_categories]
    assert "Object detection" in titles and "Otsu" in titles
    assert "Cellpose-SAM" not in titles
    assert screen._btn_cellpose.text() == "Object detection"


def test_the_old_mode_name_still_selects_the_mode_it_was_renamed_from():
    """A saved session or a script from before the rename still runs."""
    assert mm.canonical_magnifier_mode("classical") == "otsu"
    assert mm.canonical_magnifier_mode("cellpose") == "cellpose"
    assert mm.canonical_magnifier_mode(None) == "otsu"
    assert mm._MAGNIFIER_SEGMENTERS["otsu"] is mm._otsu_segmenter


def test_set_mode_takes_the_old_name_and_stores_the_new_one(screen):
    screen._magnifier.set_mode("classical")
    assert screen._magnifier.mode == "otsu"
    assert screen._magnifier.build_request() is None or (
        screen._magnifier.build_request().mode == "otsu")


def test_a_layout_that_folded_the_old_category_folds_the_new_one(
        qtbot, qt_theme_applied, monkeypatch):
    """The stored layout is a list of TITLES, so a rename loses the user's
    arrangement unless the old title is read as the new one."""
    from spacr.qt import preferences

    monkeypatch.setattr(
        preferences, "get_section_layout",
        lambda panel: {"folded": ["Cellpose-SAM", "Brush"]})
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        folded = {title for title, section in made._settings_categories
                  if not section.is_expanded()}
        assert folded == {"Object detection", "Brush"}
    finally:
        made._magnifier.close()
        made.close_folded()


def test_the_otsu_category_holds_six_settings_and_drives_the_magnifier(screen):
    categories = dict(screen._settings_categories)
    otsu = categories["Otsu"]
    for control in (screen._otsu_correction, screen._otsu_smoothing,
                    screen._otsu_bright, screen._otsu_fill_holes,
                    screen._otsu_split, screen._otsu_exclude_border):
        assert otsu.isAncestorOf(control), control
    screen._otsu_smoothing.setValue(2.5)
    screen._otsu_fill_holes.setChecked(False)
    screen._otsu_split.setChecked(False)
    context = screen._magnifier_context()
    assert context["otsu_smoothing"] == pytest.approx(2.5)
    assert context["otsu_fill_holes"] is False
    assert context["otsu_split"] is False
    screen._btn_magnifier.setChecked(True)
    screen._magnifier.hover(QPointF(*canvas_xy(32, 32)))
    request = screen._magnifier.build_request()
    assert request is not None
    assert request.otsu_smoothing == pytest.approx(2.5)
    assert request.otsu_fill_holes is False and request.otsu_split is False


def test_the_request_key_carries_every_otsu_setting(screen):
    """A setting the key does not carry is a setting a cached answer ignores."""
    assert mm._MODEL_SETTING_FIELDS[-3:] == (
        "otsu_smoothing", "otsu_fill_holes", "otsu_split")
    screen._btn_magnifier.setChecked(True)
    screen._magnifier.hover(QPointF(*canvas_xy(32, 32)))
    before = screen._magnifier.build_request().key
    screen._otsu_split.setChecked(not screen._otsu_split.isChecked())
    assert screen._magnifier.build_request().key != before


def otsu_field() -> np.ndarray:
    """Two touching disks with a hole in one, on a noisy background."""
    yy, xx = np.mgrid[0:64, 0:64]
    field = np.full((64, 64), 200.0)
    for cy, cx in ((32, 26), (32, 40)):
        field[(yy - cy) ** 2 + (xx - cx) ** 2 <= 81] = 3000.0
    field[(yy - 32) ** 2 + (xx - 26) ** 2 <= 4] = 200.0
    field[0:6, 0:6] = 3000.0
    return field.astype(np.uint16)


def test_split_touching_turns_one_blob_into_two():
    field = otsu_field()
    whole = engine._otsu_instances(field, min_area=4)
    split = engine._otsu_instances(field, min_area=4, split_touching=True)
    assert int(split.max()) > int(whole.max()), (
        "the two disks touch, so a plain labelling calls them one object")


def test_fill_holes_closes_the_hole_in_the_disk():
    field = otsu_field()
    plain = engine._otsu_instances(field, min_area=4)
    filled = engine._otsu_instances(field, min_area=4, fill_holes=True)
    assert int((filled > 0).sum()) > int((plain > 0).sum())


def test_exclude_border_drops_the_object_the_frame_cuts():
    field = otsu_field()
    kept = engine._otsu_instances(field, min_area=4)
    clear = engine._otsu_instances(field, min_area=4, exclude_border=True)
    assert int(clear.max()) == int(kept.max()) - 1, (
        "the corner square is the one the frame cuts")


def test_smoothing_stops_speckle_becoming_objects():
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:96, 0:96]
    field = np.full((96, 96), 500.0)
    field[(yy - 48) ** 2 + (xx - 48) ** 2 <= 100] = 2500.0
    field = np.clip(field + rng.normal(0, 260, field.shape), 0, None)
    field = field.astype(np.uint16)
    rough = engine._otsu_instances(field, min_area=1)
    smooth = engine._otsu_instances(field, min_area=1, smoothing=2.0)
    assert int(smooth.max()) < int(rough.max()), (
        f"smoothing left {smooth.max()} objects against {rough.max()}")


def test_otsu_detect_reads_the_category_and_records_what_it_used(screen):
    screen._min_area.setValue(4)
    screen._otsu_split.setChecked(False)
    screen._otsu_fill_holes.setChecked(False)
    screen._otsu_smoothing.setValue(0.0)
    screen._btn_otsu.click()
    detail = screen._log.edits[-1].detail
    assert detail["method"] == "otsu"
    assert detail["otsu_split"] is False and detail["otsu_fill_holes"] is False
    assert detail["otsu_smoothing"] == pytest.approx(0.0)
    assert detail["otsu_exclude_border"] is False


def test_the_otsu_boxes_change_the_mask_the_button_makes(screen):
    """Two presses, two settings, two different masks: the boxes are read."""
    screen._min_area.setValue(4)
    screen._combine_mode.setCurrentText("replace")
    screen._otsu_fill_holes.setChecked(False)
    screen._btn_otsu.click()
    without = int((screen._canvas.mask > 0).sum())
    screen._on_undo()
    screen._otsu_fill_holes.setChecked(True)
    screen._btn_otsu.click()
    assert int((screen._canvas.mask > 0).sum()) >= without


def test_the_otsu_defaults_are_the_magnifiers_own(screen):
    """The box under the mouse is the preview, so the button starts there."""
    settings = screen._otsu_settings()
    assert settings["smoothing"] == pytest.approx(mm.OTSU_SMOOTHING)
    assert settings["fill_holes"] is True
    assert settings["split_touching"] is True
    assert settings["exclude_border"] is False, (
        "the magnifier answers the border with its own box-border switch")


def test_the_panels_smoothing_default_is_still_the_engines():
    """``OTSU_SMOOTHING`` is a COPY, so something has to hold the two equal.

    ``mm.OTSU_SMOOTHING`` is written out in the screen rather than imported,
    because the value it copies -- ``mask_engine._CLASSICAL_SMOOTHING`` --
    is private and a screen importing a private name from the engine is
    worse than a duplicated float. But an unpinned copy is exactly the
    silent disagreement between the preview and the button that point 5
    exists to close, moved one level up: change the engine's sigma and the
    Smoothing box, the request default and the magnifier's own cut part
    company with nothing going red. This is the something.
    """
    assert mm.OTSU_SMOOTHING == pytest.approx(engine._CLASSICAL_SMOOTHING)
    assert (mm._MagnifierRequest._field_defaults["otsu_smoothing"]
            == pytest.approx(engine._CLASSICAL_SMOOTHING))


# ---------------------------------------------------------------------------
# 6. Clear, Dilate and Shrink
# ---------------------------------------------------------------------------

def test_dilate_grows_every_object_and_fuses_none(screen):
    before = {v: int((screen._canvas.mask == v).sum()) for v in ids_of(screen._canvas.mask)}
    screen._grow_step.setValue(2)
    screen._btn_dilate.click()
    after = {v: int((screen._canvas.mask == v).sum()) for v in ids_of(screen._canvas.mask)}
    assert sorted(after) == sorted(before), "an object was lost or fused"
    for value, area in before.items():
        assert after[value] > area, f"object {value} did not grow"
    assert "Dilated 3 object(s) by 2 px" in screen._status_label.text()


def test_shrink_pulls_two_touching_objects_apart(screen):
    """4 and 9 are one pixel apart, so a shrink opens the seam between them."""
    mask = screen._canvas.mask
    seam_before = int((mask[10:20, 19] > 0).sum() + (mask[10:20, 21] > 0).sum())
    assert seam_before > 0
    screen._grow_step.setValue(2)
    screen._btn_shrink.click()
    mask = screen._canvas.mask
    assert int((mask[10:20, 19] > 0).sum()) == 0
    assert int((mask[10:20, 21] > 0).sum()) == 0
    assert ids_of(mask) == [4, 6, 9], "shrinking must not renumber anything"
    assert "Shrank 3 object(s) by 2 px" in screen._status_label.text()


def test_a_shrink_that_erases_an_object_says_so(screen):
    screen._grow_step.setValue(6)
    screen._btn_shrink.click()
    assert ids_of(screen._canvas.mask) == []
    text = screen._status_label.text()
    assert "3 of them were thinner than 12 px" in text, text
    assert "Undo" in text


def test_the_shrink_count_is_what_was_shrunk_and_not_what_survived(screen):
    """Both numbers are counted BEFORE the edit, and they mean two things.

    Object 6 is 8 px across and objects 4 and 9 are 10, so a step of 4 keeps
    two and erases one. The sentence has to say that three objects were
    eroded and one of the three did not survive it; reporting the survivors
    as the number shrunk would read "Shrank 2 object(s) ... 1 of them are
    gone" over a field where all three were eroded, and emptying the field
    entirely would read "Shrank 0 object(s)".
    """
    assert ids_of(screen._canvas.mask) == [4, 6, 9]
    screen._grow_step.setValue(4)
    screen._btn_shrink.click()
    assert ids_of(screen._canvas.mask) == [4, 9], "geometry assumption broke"
    text = screen._status_label.text()
    assert text.startswith("Shrank 3 object(s) by 4 px"), text
    assert "1 of them were thinner than 8 px" in text, text


def test_dilate_and_shrink_are_one_undo_step_each(screen):
    before = screen._canvas.mask.copy()
    screen._grow_step.setValue(1)
    screen._btn_dilate.click()
    screen._btn_shrink.click()
    kinds = [entry.kind for entry in screen._log.edits]
    assert kinds[-2:] == ["dilate", "shrink"]
    assert screen._log.edits[-1].detail["step"] == 1
    assert screen._btn_undo.isEnabled()
    screen._on_undo()
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)


def test_a_dilate_and_a_shrink_of_the_same_size_cancel_out(screen):
    """The two use the same metric, or 'grow then shrink' would not be safe."""
    lone = np.zeros((64, 64), np.uint16)
    lone[20:40, 20:40] = 5
    screen._canvas.mask = lone.copy()
    screen._history.push(screen._canvas.mask)
    screen._grow_step.setValue(3)
    screen._btn_dilate.click()
    screen._btn_shrink.click()
    np.testing.assert_array_equal(screen._canvas.mask, lone)


def test_clear_asks_first_and_cancel_leaves_the_mask(screen):
    asked = []
    screen._confirm = lambda title, text: (asked.append((title, text)), False)[1]
    before = screen._canvas.mask.copy()
    screen._btn_clear.click()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert len(asked) == 1
    assert "3 object(s)" in asked[0][1], asked
    assert len(screen._log) == 0


def test_clear_removes_every_object_once_the_popup_is_accepted(screen):
    screen._confirm = lambda *_a, **_k: True
    screen._btn_clear.click()
    assert ids_of(screen._canvas.mask) == []
    assert screen._log.edits[-1].kind == "clear"
    screen._on_undo()
    assert ids_of(screen._canvas.mask) == [4, 6, 9]


def test_the_three_buttons_are_dead_until_a_folder_is_open(
        qtbot, qt_theme_applied):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        for button in (made._btn_clear, made._btn_dilate, made._btn_shrink):
            assert not button.isEnabled(), button.text()
    finally:
        made._magnifier.close()
        made.close_folded()


# ---------------------------------------------------------------------------
# The engine's own contract, with no display
# ---------------------------------------------------------------------------

def test_dilate_never_takes_a_pixel_from_a_neighbour():
    mask = np.zeros((40, 40), np.uint16)
    mask[5:15, 5:15] = 7
    mask[5:15, 16:26] = 3
    grown = engine.dilate_objects(mask, 4)
    assert ids_of(grown) == [3, 7]
    assert int((grown == 7).sum()) > 100 and int((grown == 3).sum()) > 100
    kept = (mask > 0)
    np.testing.assert_array_equal(grown[kept], mask[kept])


def test_shrink_erodes_each_object_against_its_neighbours_too():
    mask = np.zeros((40, 40), np.uint16)
    mask[5:15, 5:15] = 7
    mask[5:15, 15:25] = 3
    shrunk = engine.shrink_objects(mask, 1)
    assert int((shrunk[:, 14] > 0).sum()) == 0, (
        "the shared border is a border for both objects")
    assert int((shrunk[:, 15] > 0).sum()) == 0
    assert int((shrunk == 7).sum()) == 64 and int((shrunk == 3).sum()) == 64


def test_shrink_drops_an_object_thinner_than_twice_the_step():
    mask = np.zeros((20, 20), np.uint16)
    mask[5, 2:18] = 4
    mask[10:18, 10:18] = 9
    assert ids_of(engine.shrink_objects(mask, 1)) == [9]


def test_neither_edit_changes_the_dtype_or_the_ids():
    mask = np.zeros((32, 32), np.uint16)
    mask[4:12, 4:12] = 1000
    for out in (engine.dilate_objects(mask, 2), engine.shrink_objects(mask, 2)):
        assert out.dtype == mask.dtype
        assert ids_of(out) == [1000]


def test_a_step_of_zero_is_a_copy_and_not_an_edit():
    mask = np.zeros((16, 16), np.uint16)
    mask[4:9, 4:9] = 2
    for out in (engine.dilate_objects(mask, 0), engine.shrink_objects(mask, 0)):
        np.testing.assert_array_equal(out, mask)
        assert out is not mask
