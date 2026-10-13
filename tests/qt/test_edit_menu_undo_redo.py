"""The Edit menu follows the visible editor's existing history."""

import sqlite3

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtGui import QKeySequence, QUndoStack
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from spacr.qt.app import MainWindow
from spacr.qt.shortcuts import _record_edit


def _window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _screen(window, child=None):
    screen = QWidget()
    if child is not None:
        layout = QVBoxLayout(screen)
        layout.addWidget(child)
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    return screen


def _menu(window):
    window._edit_menu_about_to_show()
    return window._act_edit_undo, window._act_edit_redo


def test_edit_menu_is_disabled_on_home_and_adds_no_duplicate_undo_keys(qtbot):
    window = _window(qtbot)
    undo, redo = _menu(window)
    assert window._edit_menu.title() == "Edit"
    assert not undo.isEnabled() and not redo.isEnabled()
    assert undo.objectName() == "EditUndoAction"
    assert redo.objectName() == "EditRedoAction"
    assert undo.shortcut() == QKeySequence()
    assert redo.shortcut() == QKeySequence()


def test_edit_menu_routes_settings_stack_and_follows_navigation(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    window = _window(qtbot)
    screen = _screen(window)
    model = SettingsWidgets("measure", parent=screen)
    model.build_sections()
    model._enable_commit_observation(lambda _key: None)
    screen._settings_model = model
    value = {"setting": 2}
    assert _record_edit(model.undo_stack, "Change setting",
                        lambda number: value.__setitem__("setting", number), 1, 2)
    undo, redo = _menu(window)
    assert undo.isEnabled() and not redo.isEnabled()
    undo.trigger()
    assert value["setting"] == 1
    assert not undo.isEnabled() and redo.isEnabled()
    redo.trigger()
    assert value["setting"] == 2
    window._stack.setCurrentWidget(window._startup)
    assert not undo.isEnabled() and not redo.isEnabled()


def test_edit_menu_routes_real_gate_history(qtbot):
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate

    window = _window(qtbot)
    panel = GateEditorPanel()
    screen = _screen(window, panel)
    screen.gates = panel
    panel.set_gates(GateSet([ThresholdGate(name="g", column="a", low=1)]))
    undo, redo = _menu(window)
    assert undo.isEnabled() and not redo.isEnabled()
    undo.trigger()
    assert len(panel.gates) == 0
    assert redo.isEnabled()
    redo.trigger()
    assert len(panel.gates) == 1


def test_edit_menu_routes_mask_history_and_respects_disabled_buttons(qtbot):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    window = _window(qtbot)
    screen = MakeMasksScreen()
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    image = np.zeros((24, 24), np.uint16)
    before = np.zeros((24, 24), np.uint16)
    after = before.copy()
    after[10:14, 10:14] = 1
    screen._canvas.set_image_and_mask(image, after.copy())
    screen._history.push(before)
    screen._history.push(after)
    screen._refresh_history_buttons()
    undo, redo = _menu(window)
    assert undo.isEnabled() and not redo.isEnabled()
    undo.trigger()
    assert not screen._canvas.mask.any()
    assert not undo.isEnabled() and redo.isEnabled()
    redo.trigger()
    assert np.array_equal(screen._canvas.mask, after)


def test_edit_menu_routes_real_annotation_history(qtbot, tmp_path):
    from spacr.qt.screens.annotate import AnnotateScreen

    source = tmp_path / "experiment"
    (source / "measurements").mkdir(parents=True)
    (source / "data").mkdir()
    paths = []
    for index in range(6):
        path = source / "data" / f"crop_{index}.png"
        Image.fromarray(np.full((24, 24, 3), index * 20, np.uint8)).save(path)
        paths.append(str(path))
    with sqlite3.connect(source / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" VALUES (?)',
                         [(path,) for path in paths])
    window = _window(qtbot)
    screen = AnnotateScreen()
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    screen._settings.grid_rows, screen._settings.grid_cols = 2, 3
    screen._settings.image_size = (24, 24)
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(source))
    qtbot.waitUntil(lambda: len(screen._page_paths) == 6, timeout=5000)
    try:
        screen._set_focus_slot(0)
        screen.handle_key("1")
        assert screen._current_value(0) == 1
        undo, redo = _menu(window)
        assert undo.isEnabled() and not redo.isEnabled()
        undo.trigger()
        assert screen._current_value(0) is None
        assert redo.isEnabled()
        redo.trigger()
        assert screen._current_value(0) == 1
        assert not undo.shortcut().toString()
        assert not redo.shortcut().toString()
    finally:
        if screen._worker is not None:
            screen._worker.stop(wait=True)


def test_focused_text_owns_undo_before_a_parent_settings_stack(qtbot):
    window = _window(qtbot)
    line = QLineEdit()
    screen = _screen(window, line)
    model = type("Model", (), {})()
    model.undo_stack = QUndoStack(screen)
    screen._settings_model = model
    setting = {"value": 2}
    _record_edit(model.undo_stack, "Change setting",
                 lambda value: setting.__setitem__("value", value), 1, 2)
    line.setText("abc")
    line.insert("d")
    assert line.isUndoAvailable()
    window.show()
    window.activateWindow()
    line.setFocus()
    qtbot.waitUntil(lambda: QApplication.focusWidget() is line)
    window._edit_menu.popup(window.mapToGlobal(QPoint(20, 30)))
    qtbot.waitUntil(window._edit_menu.isVisible)
    assert window._edit_focus is line
    undo = window._act_edit_undo
    assert undo.isEnabled()
    undo.trigger()
    window._edit_menu.hide()
    assert line.text() == "abc"
    assert setting["value"] == 2
    assert model.undo_stack.canUndo()
