"""Item 697: each disconnected brush stroke is a new object label.

A stroke that touches nothing paints the next free label; one that starts on
or touches an object (8-connected in 2-D, 26-connected through the adjacent
sections of a volume) extends the first object it met and never merges two;
Shift keeps the last label across a gap; Erase never adds a label; undo and
redo give back the exact array, ids and dtype included.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (MODE_BRUSH, MODE_ERASE, MakeMasksScreen,
                                         _MaskCanvas)

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2


def canvas_xy(x, y):
    return (MARGIN_X + (x + .5) * PIXMAP_N / IMG_N, (y + .5) * PIXMAP_N / IMG_N)


def _event(kind, x, y, buttons, button, modifiers):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def stroke(canvas, points, modifiers=Qt.NoModifier, to_widget=canvas_xy):
    """One brush gesture through image pixels ``points``."""
    first = to_widget(*points[0])
    canvas.mousePressEvent(_event(QEvent.Type.MouseButtonPress, *first,
                                  Qt.LeftButton, Qt.LeftButton, modifiers))
    for point in points[1:]:
        canvas.mouseMoveEvent(_event(QEvent.Type.MouseMove, *to_widget(*point),
                                     Qt.LeftButton, Qt.NoButton, modifiers))
    last = to_widget(*points[-1])
    canvas.mouseReleaseEvent(_event(QEvent.Type.MouseButtonRelease, *last,
                                    Qt.NoButton, Qt.LeftButton, modifiers))


def ids(mask):
    return {int(v) for v in np.unique(mask) if v}


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    c = _MaskCanvas()
    qtbot.addWidget(c)
    c.resize(CANVAS_W, CANVAS_H)
    image = np.zeros((IMG_N, IMG_N), np.uint16)
    c.set_image_and_mask(image, np.zeros((IMG_N, IMG_N), np.uint8))
    assert c.pixmap().deviceIndependentSize().width() == PIXMAP_N
    c.mode = MODE_BRUSH
    c.brush_radius = 10
    assert c._mask_radius_for_brush() == 1
    return c


def test_three_separate_strokes_on_empty_ground_are_three_objects(canvas):
    for row in (10, 30, 50):
        stroke(canvas, [(5, row), (20, row)])
    assert ids(canvas.mask) == {1, 2, 3}
    assert int(canvas.mask[10, 12]) == 1
    assert int(canvas.mask[30, 12]) == 2
    assert int(canvas.mask[50, 12]) == 3
    assert canvas.last_edit["target"] == 3
    assert not canvas.manual_ids


def test_a_stroke_from_inside_an_object_outward_stays_that_object(canvas):
    canvas.mask[20:30, 20:30] = 5
    stroke(canvas, [(25, 25), (45, 25)])
    assert ids(canvas.mask) == {5}
    assert (canvas.mask[24:26, 25:46] == 5).all()


def test_stationary_brush_events_keep_one_continuous_object(canvas):
    stroke(canvas, [(5, 10), (5, 10), (20, 10)])
    repeated = canvas.mask.copy()
    assert ids(repeated) == {1}
    assert int(repeated[10, 12]) == 1
    assert not canvas.manual_ids
    canvas.set_image_and_mask(np.zeros((IMG_N, IMG_N), np.uint16),
                              np.zeros((IMG_N, IMG_N), np.uint8))
    stroke(canvas, [(5, 10), (20, 10)])
    np.testing.assert_array_equal(canvas.mask, repeated)


def test_a_stroke_on_background_that_touches_an_object_extends_it(canvas):
    canvas.mask[20:30, 20:30] = 5
    stroke(canvas, [(45, 25), (31, 25)])
    assert ids(canvas.mask) == {5}
    assert int(canvas.mask[25, 44]) == 5, "the part painted before the touch is relabelled"


def test_touching_is_eight_connected_a_corner_counts_and_a_gap_does_not(canvas):
    canvas.mask[20:30, 20:30] = 5
    stroke(canvas, [(40, 40), (31, 31)])
    assert int(canvas.mask[30, 30]) == 5 and int(canvas.mask[29, 30]) == 0
    assert ids(canvas.mask) == {5}, "a corner-to-corner contact is a touch"
    canvas.mask[30:, 30:] = 0
    stroke(canvas, [(40, 40), (32, 32)])
    assert ids(canvas.mask) == {5, 6}, "a one-pixel gap is not a touch"


def test_a_stroke_touching_two_objects_extends_the_first_and_merges_nothing(canvas):
    canvas.mask[20:30, 5:15] = 3
    canvas.mask[20:30, 40:50] = 7
    seven = canvas.mask == 7
    stroke(canvas, [(25, 25), (16, 25)])
    stroke(canvas, [(16, 27), (39, 27)])
    assert ids(canvas.mask) == {3, 7}
    assert (canvas.mask[seven] == 7).sum() >= seven.sum() - 4
    stroke(canvas, [(10, 22), (45, 22)])
    assert ids(canvas.mask) == {3, 7}
    assert int(canvas.mask[22, 45]) == 3, "a stroke started on 3 stays 3"
    assert (canvas.mask[25:30, 40:50] == 7).all(), "7 is still its own object"


def test_shift_keeps_the_last_label_across_a_gap_and_groups_it(canvas):
    stroke(canvas, [(5, 10), (20, 10)])
    stroke(canvas, [(40, 50), (55, 50)], Qt.ShiftModifier)
    assert ids(canvas.mask) == {1}
    assert int(canvas.mask[50, 45]) == 1
    assert canvas.manual_ids, "the two pieces stay one object when saved"
    stroke(canvas, [(5, 30), (20, 30)])
    assert ids(canvas.mask) == {1, 2}


def test_shift_with_nothing_drawn_yet_starts_a_new_object(canvas):
    stroke(canvas, [(5, 10), (20, 10)], Qt.ShiftModifier)
    assert ids(canvas.mask) == {1}
    assert not canvas.manual_ids


def test_a_shift_stroke_that_reaches_its_object_does_not_group(canvas):
    stroke(canvas, [(5, 10), (20, 10)])
    stroke(canvas, [(20, 11), (30, 20)], Qt.ShiftModifier)
    assert ids(canvas.mask) == {1}
    assert not canvas.manual_ids


def test_shift_with_the_brush_paints_and_alt_still_pans(canvas):
    canvas.zoom_at(32, 32, 4.0)
    view = canvas._viewport_bounds()
    x0, y0 = view[0], view[1]

    def zoomed(x, y):
        point = canvas._image_to_canvas(x + .5, y + .5)
        return point.x(), point.y()

    stroke(canvas, [(x0 + 2, y0 + 2), (x0 + 6, y0 + 2)], Qt.ShiftModifier, zoomed)
    assert canvas._viewport_bounds() == view and canvas.mask.any()
    before = canvas.mask.copy()
    canvas.mousePressEvent(_event(QEvent.Type.MouseButtonPress, 300, 200,
                                  Qt.LeftButton, Qt.LeftButton, Qt.AltModifier))
    canvas.mouseMoveEvent(_event(QEvent.Type.MouseMove, 200, 150,
                                 Qt.LeftButton, Qt.NoButton, Qt.AltModifier))
    canvas.mouseReleaseEvent(_event(QEvent.Type.MouseButtonRelease, 200, 150,
                                    Qt.NoButton, Qt.LeftButton, Qt.AltModifier))
    assert canvas._viewport_bounds() != view
    np.testing.assert_array_equal(canvas.mask, before)


def test_erase_never_creates_labels(canvas):
    canvas.mask[20:30, 20:30] = 5
    canvas.mask[40:50, 40:50] = 9
    canvas.mode = MODE_ERASE
    stroke(canvas, [(10, 25), (55, 45)])
    stroke(canvas, [(25, 10), (25, 60)], Qt.ShiftModifier)
    assert ids(canvas.mask) <= {5, 9}
    assert canvas._stroke_label is None


def test_a_binary_mask_is_numbered_so_a_bridge_merges_nothing(canvas):
    canvas.mask[20:30, 5:15] = 255
    canvas.mask[20:30, 40:50] = 255
    stroke(canvas, [(16, 25), (39, 25)])
    assert len(ids(canvas.mask)) == 2, "the stroke joined one blob, not both"
    assert int(canvas.mask[25, 10]) != int(canvas.mask[25, 45])


# ---------------------------------------------------------------------------
# The screen: undo/redo byte-identity, save and the object table
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path):
    folder = tmp_path / "images"
    (folder / "masks").mkdir(parents=True)
    image = np.arange(IMG_N * IMG_N, dtype=np.uint16).reshape(IMG_N, IMG_N)
    for name in ("field_00", "field_01"):
        imageio.imwrite(folder / (name + ".tif"), image)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    qtbot.waitUntil(lambda: not widget._loading and widget._reader_worker is None
                    and widget._load_worker is None)
    widget._canvas.resize(CANVAS_W, CANVAS_H)
    widget._canvas.show()
    widget._canvas.refresh()
    widget._set_mode(MODE_BRUSH)
    widget._brush_slider.setValue(1)
    widget.folder = folder
    yield widget
    widget.close()


def _on_screen(screen):
    canvas = screen._canvas

    def to_widget(x, y):
        point = canvas._image_to_canvas(x + .5, y + .5)
        return point.x(), point.y()
    return to_widget


def _state(mask):
    mask = np.asarray(mask)
    return mask.dtype.str, mask.shape, mask.tobytes()


def test_undo_and_redo_restore_every_label_id_byte_for_byte(screen):
    canvas = screen._canvas
    to_widget = _on_screen(screen)
    gestures = [
        ([(5, 10), (20, 10)], Qt.NoModifier),
        ([(5, 30), (20, 30)], Qt.NoModifier),
        ([(40, 50), (55, 50)], Qt.ShiftModifier),
        ([(21, 31), (30, 40)], Qt.NoModifier),
        ([(5, 55), (20, 55)], Qt.NoModifier),
    ]
    states = [_state(canvas.mask)]
    manual = [canvas.manual_ids]
    for points, modifiers in gestures:
        stroke(canvas, points, modifiers, to_widget)
        states.append(_state(canvas.mask))
        manual.append(canvas.manual_ids)
    assert ids(canvas.mask) == {1, 2, 3}
    for index in range(len(gestures) - 1, -1, -1):
        screen._on_undo()
        assert _state(canvas.mask) == states[index]
        assert canvas.manual_ids == manual[index]
    for index in range(1, len(gestures) + 1):
        screen._on_redo()
        assert _state(canvas.mask) == states[index]
        assert canvas.manual_ids == manual[index]


def test_saved_mask_and_object_table_show_the_new_objects(screen):
    canvas = screen._canvas
    to_widget = _on_screen(screen)
    for row in (10, 30):
        stroke(canvas, [(5, row), (20, row)], Qt.NoModifier, to_widget)
    stroke(canvas, [(40, 50), (55, 50)], Qt.ShiftModifier, to_widget)
    stroke(canvas, [(5, 50), (15, 50)], Qt.NoModifier, to_widget)
    assert screen._objects_now() == 3
    screen._btn_save.click()
    saved = imageio.imread(engine.mask_save_path(str(screen.folder), "field_00.tif"))
    assert ids(saved) == {1, 2, 3}
    assert int(saved[50, 45]) == 2, "the Shift piece stays object 2 on disk"
    table = engine._vvvv_object_table(saved)
    assert sorted(table["label"]) == [1, 2, 3]
    assert len(table) == 3


def test_brush_label_256_and_shift_identity_survive_unicode_save_reopen(
        qtbot, qt_theme_applied, tmp_path):
    import unicodedata

    folder = tmp_path / unicodedata.normalize("NFD", "Ólafur's μασκ 测试")
    (folder / "masks").mkdir(parents=True)
    filename = unicodedata.normalize("NFD", "région_🧫.tif")
    image = np.arange(IMG_N * IMG_N, dtype=np.uint16).reshape(IMG_N, IMG_N)
    initial = np.zeros(image.shape, np.uint8)
    initial[4:9, 4:9] = 254
    initial[4:9, 50:55] = 255
    imageio.imwrite(folder / filename, image)
    imageio.imwrite(folder / "masks" / filename, initial)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    qtbot.waitUntil(lambda: not widget._loading and widget._reader_worker is None
                    and widget._load_worker is None)
    canvas = widget._canvas
    canvas.resize(CANVAS_W, CANVAS_H)
    canvas.show()
    canvas.refresh()
    widget._set_mode(MODE_BRUSH)
    widget._brush_slider.setValue(1)
    assert _state(canvas.mask) == _state(initial)
    states = [_state(canvas.mask)]
    stroke(canvas, [(5, 28), (20, 28)], Qt.NoModifier, _on_screen(widget))
    assert canvas.mask.dtype == np.uint16
    assert ids(canvas.mask) == {254, 255, 256}
    assert int(canvas.mask[28, 12]) == 256
    states.append(_state(canvas.mask))
    stroke(canvas, [(40, 50), (55, 50)], Qt.ShiftModifier, _on_screen(widget))
    assert ids(canvas.mask) == {254, 255, 256}
    assert int(canvas.mask[50, 45]) == 256 and canvas.manual_ids
    np.testing.assert_array_equal(canvas.mask[initial > 0], initial[initial > 0])
    states.append(_state(canvas.mask))
    for expected in reversed(states[:-1]):
        widget._on_undo()
        assert _state(canvas.mask) == expected
    for expected in states[1:]:
        widget._on_redo()
        assert _state(canvas.mask) == expected
    assert canvas.manual_ids
    expected = canvas.mask.copy()
    widget._btn_save.click()
    saved_path = engine.mask_save_path(str(folder), filename)
    assert mm._wait_for_mask_writes(saved_path, timeout=10)
    saved = imageio.imread(saved_path)
    assert _state(saved) == _state(expected)
    assert sorted(engine._vvvv_object_table(saved)["label"]) == [254, 255, 256]
    widget.close()
    reopened = MakeMasksScreen()
    qtbot.addWidget(reopened)
    assert reopened._open_folder(str(folder))
    qtbot.waitUntil(lambda: not reopened._loading and reopened._reader_worker is None
                    and reopened._load_worker is None)
    assert _state(reopened._canvas.mask) == _state(expected)
    assert reopened._canvas.manual_ids
    assert sorted(engine._vvvv_object_table(reopened._canvas.mask)["label"]) == [254, 255, 256]
    np.testing.assert_array_equal(imageio.imread(folder / filename), image)
    reopened.close()


# ---------------------------------------------------------------------------
# 3-D: per stroke in the shown section, 26-connected through its neighbours
# ---------------------------------------------------------------------------

@pytest.fixture
def stage(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1500, 950)
    widget.show()
    qtbot.waitExposed(widget)
    path = Path(tmp_path) / "volume.tif"
    tifffile.imwrite(path, np.random.default_rng(4).integers(0, 255, (10, 24, 32), dtype=np.uint8))
    stage = widget.open_volume(str(path))
    assert stage is widget._mask_stage
    for _ in range(4):
        QApplication.processEvents()
    widget._set_mode(MODE_BRUSH)
    widget._brush_slider.setValue(1)
    yield stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle(), timeout=60000)
    stage.dirty = False


def _stroke_3d(stage, points, modifiers=Qt.NoModifier):
    canvas = stage.canvas
    assert canvas._mask_radius_for_brush() == 1

    def to_widget(x, y):
        point = canvas._image_to_canvas(x + .5, y + .5)
        return point.x(), point.y()
    stroke(canvas, points, modifiers, to_widget)
    QApplication.processEvents()


def test_3d_strokes_follow_the_rule_per_section_and_through_neighbours(stage):
    session = stage.session
    stage.set_slice(5)
    _stroke_3d(stage, [(4, 4), (12, 4)])
    assert ids(session.mask) == {1} and ids(session.mask[5]) == {1}
    _stroke_3d(stage, [(4, 15), (12, 15)])
    assert ids(session.mask[5]) == {1, 2}, "a separate stroke in the section is new"
    stage.set_slice(6)
    _stroke_3d(stage, [(14, 6), (20, 6)])
    assert ids(session.mask[6]) == {1}, "a diagonal neighbour in the next section extends 1"
    stage.set_slice(8)
    _stroke_3d(stage, [(4, 4), (12, 4)])
    assert ids(session.mask[8]) == {3}, "two sections away is not a touch"
    assert stage._selected == 3
    stage.set_slice(2)
    _stroke_3d(stage, [(4, 20), (12, 20)], Qt.ShiftModifier)
    assert ids(session.mask[2]) == {3}, "Shift keeps the chosen object"
    stage.set_slice(5)
    _stroke_3d(stage, [(2, 10), (14, 10)])
    assert ids(session.mask[5]) == {1, 2, 4}
    assert ids(session.mask) == {1, 2, 3, 4}


def test_3d_undo_and_redo_restore_the_volume_exactly(stage):
    session = stage.session
    screen = stage._screen
    states = [_state(session.mask)]
    for z, row in ((5, 4), (5, 15), (6, 4)):
        stage.set_slice(z)
        _stroke_3d(stage, [(4, row), (12, row)])
        states.append(_state(session.mask))
    assert ids(session.mask) == {1, 2}
    for index in range(2, -1, -1):
        screen._on_undo()
        assert _state(session.mask) == states[index]
    for index in range(1, 4):
        screen._on_redo()
        assert _state(session.mask) == states[index]


@pytest.mark.parametrize('edge', (0, 9))
def test_volume_edge_strokes_never_wrap_to_the_opposite_section(stage, edge):
    session = stage.session
    stage.set_slice(9 - edge)
    _stroke_3d(stage, [(4, 4), (12, 4)])
    assert ids(session.mask) == {1}
    stage.set_slice(edge)
    _stroke_3d(stage, [(4, 4), (12, 4)])
    assert ids(session.mask[edge]) == {2}
    stage.set_slice(1 if edge == 0 else 8)
    _stroke_3d(stage, [(14, 6), (20, 6)])
    assert ids(session.mask[1 if edge == 0 else 8]) == {2}
    assert ids(session.mask) == {1, 2}
    stage.dirty = False
    stage.leave()
    assert stage._stroke_context(stage.canvas) == (1, ())


def test_touching_two_neighbouring_volume_objects_preserves_the_second(stage):
    session = stage.session
    stage.set_slice(4)
    _stroke_3d(stage, [(4, 4), (12, 4)])
    stage.set_slice(6)
    _stroke_3d(stage, [(4, 4), (12, 4)])
    assert ids(session.mask[4]) == {1}
    assert ids(session.mask[6]) == {2}
    second = session.mask[6].copy()
    stage.set_slice(5)
    _stroke_3d(stage, [(14, 6), (20, 6)])
    assert ids(session.mask[5]) == {1}
    assert ids(session.mask) == {1, 2}
    np.testing.assert_array_equal(session.mask[6], second)


def test_the_brush_and_its_shift_are_on_the_shortcut_panel_and_the_tooltip():
    hints = dict(mm.SHORTCUT_HINTS)
    assert "last object" in hints["Brush: Shift + drag"]
    assert hints["Alt + drag"] == "Pan; other tools also accept Shift + drag"
    assert "Shift or Alt + drag" not in hints
    assert all(len(does) <= 52 for _keys, does in mm.SHORTCUT_HINTS)


def test_the_brush_button_says_what_shift_does(screen):
    tip = screen._mode_buttons[MODE_BRUSH].toolTip()
    assert "Shift" in tip and "new object" in tip
    assert "tools other than the Brush" in screen._zoom_speed.property('apiTooltipDescriptionSource')


@pytest.mark.parametrize('language', ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'))
def test_brush_and_pan_hints_follow_each_language_without_changing_the_tool(screen, language):
    from spacr.qt.i18n import retranslate_widget_tree, tr

    retranslate_widget_tree(screen, 'en')
    screen._set_mode(MODE_BRUSH)
    brush_tip = screen._mode_buttons[MODE_BRUSH].property('apiTooltipDescriptionSource')
    zoom_tip = screen._zoom_speed.property('apiTooltipDescriptionSource')
    retranslate_widget_tree(screen, language)
    for keys in ('Alt + drag', 'Brush: Shift + drag'):
        key_label, detail_label = screen._shortcut_rows[keys]
        assert key_label.text() == tr(keys, language)
        assert detail_label.text() == tr(dict(mm.SHORTCUT_HINTS)[keys], language)
    assert tr(brush_tip, language) in screen._mode_buttons[MODE_BRUSH].toolTip()
    zoom_label = next(widget for widget in screen.findChildren(QWidget)
                      if widget.property('settingHelpLabel')
                      and widget.property('settingKey') == 'make_masks_zoom_speed')
    assert tr(zoom_tip, language) in zoom_label.toolTip()
    assert screen._canvas.mode == MODE_BRUSH
