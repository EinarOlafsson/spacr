"""Persisted per-screen keys drive real input while preserving editor semantics."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QPixmap
from PySide6.QtWidgets import QApplication, QLineEdit, QMainWindow

from spacr.qt import shortcuts as S
from spacr.qt.screens import annotate as A
from spacr.qt.screens import make_masks as M
from spacr.qt.widgets.qc_field_browser import QCFieldBrowser, QCFieldTarget


def _editor(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    S.install(window)
    dialog = S._KeymapDialog(window)
    qtbot.addWidget(dialog)
    return window, dialog


def _event(widget, key, modifiers=Qt.NoModifier):
    event = QKeyEvent(QEvent.KeyPress, key, modifiers)
    QApplication.sendEvent(widget, event)
    return event


def test_scoped_rows_validate_conflicts_and_cancel_keeps_saved_keys(qtbot):
    _window, dialog = _editor(qtbot)
    dialog._set_screen_key("Make Masks", "B", "F8")
    dialog._set_screen_key("Annotate", "Y", "F8")
    assert not dialog.conflict_text(), "mutually exclusive screens may reuse a key"
    assert dialog.save()
    saved = S._load_screen_keymap()
    assert saved["Make Masks"] == {"B": "F8"}
    assert saved["Annotate"] == {"Y": "F8"}
    later = S._KeymapDialog(_window)
    qtbot.addWidget(later)
    later._set_screen_key("Make Masks", "E", "F8")
    assert "Brush" in later.conflict_text() and "Erase" in later.conflict_text()
    assert not later.save()
    later.restore_defaults()
    later._set_screen_key("Make Masks", "B", "Ctrl+K")
    assert "Open command palette" in later.conflict_text()
    later._set_screen_key("Make Masks", "B", "B")
    later._set_screen_key("Annotate", "Y", "1")
    assert "Assign class 1" in later.conflict_text()
    later.reject()
    assert S._load_screen_keymap() == saved


def test_clearing_restoring_sorting_and_reopening_preserve_binding_identity(qtbot):
    window, dialog = _editor(qtbot)
    dialog._table.sortItems(0, Qt.DescendingOrder)
    dialog._set_screen_key("Make Masks", "B", "")
    dialog._set_screen_key("Field browser", "Q", "F8")
    assert dialog.save()
    assert S._load_screen_keymap()["Make Masks"]["B"] == ""
    reopened = S._KeymapDialog(window)
    qtbot.addWidget(reopened)
    assert reopened._screen_keymap()["Field browser"]["Q"] == "F8"
    reopened.restore_defaults()
    assert reopened.save()
    assert not any(S._load_screen_keymap().values())
    from spacr.qt.preferences import _settings
    assert json.loads(_settings().value(S._SCREEN_KEYMAP_KEY)) == {}


@pytest.mark.parametrize("raw", ["{bad", "[]", '{"Make Masks": []}',
                                  '{"Make Masks": {"B": 4, "E": "F8, F9", "X": "bogus"}}'])
def test_corrupt_or_multistep_screen_values_do_not_disable_defaults(raw):
    from spacr.qt.preferences import _settings
    _settings().setValue(S._SCREEN_KEYMAP_KEY, raw)
    assert not any(S._load_screen_keymap().values())


def test_masks_rebinds_live_and_after_reconstruction_with_real_tool_input(qtbot):
    screen = M.MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.show()
    screen.activateWindow()
    screen._canvas.setFocus()
    screen._set_mode(M.MODE_ERASE)
    _window, dialog = _editor(qtbot)
    dialog._set_screen_key("Make Masks", "B", "F8")
    dialog._set_screen_key("Make Masks", "E", "")
    assert dialog.save()
    screen.activateWindow()
    screen._canvas.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is screen)
    qtbot.keyClick(screen._canvas, Qt.Key_B)
    assert screen._canvas.mode == M.MODE_ERASE
    qtbot.keyClick(screen._canvas, Qt.Key_F8)
    assert screen._canvas.mode == M.MODE_BRUSH
    qtbot.keyClick(screen._canvas, Qt.Key_E)
    assert screen._canvas.mode == M.MODE_BRUSH
    assert "F8" in screen._shortcut_rows["B E W D V Z R"][0].text()
    rebuilt = M.MakeMasksScreen()
    qtbot.addWidget(rebuilt)
    assert rebuilt._spacr_screen_holders["B"].key() == QKeySequence("F8")
    assert rebuilt._spacr_screen_holders["E"].key().isEmpty()
    rebuilt.show()
    rebuilt.activateWindow()
    rebuilt._canvas.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is rebuilt)
    rebuilt._set_mode(M.MODE_ERASE)
    qtbot.keyClick(rebuilt._canvas, Qt.Key_F8)
    assert rebuilt._canvas.mode == M.MODE_BRUSH
    assert screen._canvas.mode == M.MODE_BRUSH


def test_rebound_bare_key_does_not_steal_typing_from_a_text_editor(qtbot):
    S._save_keymap({}, {"Make Masks": {"B": "F8", "E": "P"}})
    screen = M.MakeMasksScreen()
    qtbot.addWidget(screen)
    edit = QLineEdit(screen)
    screen.show()
    screen.activateWindow()
    edit.show()
    edit.setFocus()
    screen._set_mode(M.MODE_BRUSH)
    qtbot.keyClick(edit, Qt.Key_P)
    assert edit.text() == "p"
    assert screen._canvas.mode == M.MODE_BRUSH


def test_annotate_event_routes_rebound_keys_and_preserves_real_label_undo(qtbot, tmp_path):
    src = tmp_path / "experiment"
    (src / "measurements").mkdir(parents=True)
    paths = []
    for index in range(6):
        path = src / f"crop{index}.png"
        Image.fromarray(np.full((24, 24, 3), index * 20, dtype=np.uint8)).save(path)
        paths.append(str(path))
    with sqlite3.connect(src / "measurements" / "measurements.db") as connection:
        connection.execute("CREATE TABLE png_list (png_path TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO png_list VALUES (?)", [(path,) for path in paths])
    screen = A.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 3
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == 6, timeout=5000)
    _window, dialog = _editor(qtbot)
    dialog._set_screen_key("Annotate", "Right", "F8")
    dialog._set_screen_key("Annotate", "Left", "Ctrl+F9")
    dialog._set_screen_key("Annotate", "Y", "F10")
    dialog._set_screen_key("Annotate", "Ctrl+Z", "Ctrl+F8")
    dialog._set_screen_key("Annotate", "Esc", "F12")
    assert dialog.save()
    assert screen.current_slot == 0
    _event(screen._grid_holder, Qt.Key_Right)
    assert screen.current_slot == 0
    _event(screen._grid_holder, Qt.Key_F8)
    assert screen.current_slot == 1
    _event(screen._grid_holder, Qt.Key_F9, Qt.ControlModifier)
    assert screen.current_slot == 0
    assert "F8" in screen._legend_label.text()
    screen._toggle_legend()
    assert "F8" in screen._legend_label.text()
    _event(screen._grid_holder, Qt.Key_1)
    assert screen._current_value(0) == 1
    screen.show()
    screen.activateWindow()
    screen._grid_holder.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is screen)
    qtbot.keyClick(screen._grid_holder, Qt.Key_F8, Qt.ControlModifier)
    assert screen._current_value(0) is None
    observed = []
    screen._kbd_judge = lambda **kw: observed.append(kw["confirm"]) or True
    _event(screen, Qt.Key_Y)
    _event(screen, Qt.Key_Y, Qt.ShiftModifier)
    _event(screen, Qt.Key_F10)
    _event(screen, Qt.Key_N)
    assert observed == [True, False]
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.red)
    screen._zoom_overlay.show_pixmap(pixmap, 0)
    qtbot.waitUntil(screen._zoom_is_open)
    _event(screen._grid_holder, Qt.Key_Escape)
    assert screen._zoom_is_open()
    _event(screen._grid_holder, Qt.Key_F12)
    assert not screen._zoom_is_open()


def test_browser_real_rebound_navigation_and_quarantine(qtbot, tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    targets = []
    for field in ("plate1_A01_1", "plate1_A02_1"):
        np.save(merged / f"{field}.npy", np.zeros((20, 20, 3), dtype=np.uint16))
        targets.append(QCFieldTarget(field=field, plate_root=str(tmp_path), merged_dir=str(merged)))
    browser = QCFieldBrowser(targets, threaded=False)
    qtbot.addWidget(browser)
    browser.show()
    _window, dialog = _editor(qtbot)
    dialog._set_screen_key("Field browser", "Right", "F8")
    dialog._set_screen_key("Field browser", "Q", "F9")
    assert dialog.save()
    browser.activateWindow()
    browser._view.setFocus()
    qtbot.keyClick(browser._view, Qt.Key_Right)
    assert browser.current_field == targets[0].field
    qtbot.keyClick(browser._view, Qt.Key_F8)
    assert browser.current_field == targets[1].field
    qtbot.keyClick(browser._view, Qt.Key_Q)
    assert (merged / f"{targets[1].field}.npy").exists()
    qtbot.keyClick(browser._view, Qt.Key_F9)
    assert not (merged / f"{targets[1].field}.npy").exists()
    assert (tmp_path / "merged_quarantined" / f"{targets[1].field}.npy").exists()
    qtbot.keyClick(browser._view, Qt.Key_F9)
    assert (merged / f"{targets[1].field}.npy").exists()
    assert "F9" in browser._quarantine.toolTip()
    _event(browser, Qt.Key_Left)
    assert browser.current_field == targets[0].field
    _event(browser, Qt.Key_F8)
    assert browser.current_field == targets[1].field
    _event(browser, Qt.Key_F12)
    assert browser.current_field == targets[1].field


@pytest.mark.parametrize("default", ["Up", "Down", "H", "J", "K", "L", "U",
                                    "Space", "Backspace", "Return", "Enter", "?", "Esc",
                                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
def test_every_existing_annotation_handler_key_can_be_rebound_or_cleared(qtbot, default):
    screen = A.AnnotateScreen()
    qtbot.addWidget(screen)
    seen = []
    screen.handle_key = lambda key, text="": seen.append(key) or True
    S._save_keymap({}, {"Annotate": {default: "F8"}})
    S._apply_screen_keymaps()
    original = QKeySequence(S._portable(default))[0].key()
    _event(screen, original)
    assert not seen, "the old declared key must stop executing its handler"
    _event(screen, Qt.Key_F8)
    assert seen == [original]
    S._save_keymap({}, {"Annotate": {default: ""}})
    S._apply_screen_keymaps()
    seen.clear()
    _event(screen, original)
    _event(screen, Qt.Key_F8)
    assert original not in seen


def test_escape_rebinding_applies_to_actual_annotation_zoom_overlay(qtbot):
    screen = A.AnnotateScreen()
    qtbot.addWidget(screen)
    called = []
    screen._zoom_overlay.dismissed.connect(lambda: called.append(True))
    S._save_keymap({}, {"Annotate": {"Esc": "F8"}})
    S._apply_screen_keymaps()
    _event(screen._zoom_overlay, Qt.Key_Escape)
    assert not called
    _event(screen._zoom_overlay, Qt.Key_F8)
    assert called == [True]


def test_default_annotation_modifiers_and_keyboard_text_keep_existing_actions(qtbot):
    screen = A.AnnotateScreen()
    qtbot.addWidget(screen)
    called = []
    screen._kbd_assign = lambda value: called.append(("label", value)) or True
    screen._kbd_judge = lambda **kwargs: called.append(("judge", kwargs["confirm"])) or True
    _event(screen, Qt.Key_1, Qt.KeypadModifier)
    _event(screen, Qt.Key_Y, Qt.ShiftModifier)
    QApplication.sendEvent(screen, QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "2"))
    assert called == [("label", 1), ("judge", True), ("label", 2)]


def test_multistep_or_invalid_saved_keys_are_refused_before_writing():
    for value in ("F8, F9", "not-a-key", 42):
        with pytest.raises(ValueError):
            S._save_keymap({}, {"Make Masks": {"B": value}})
        assert not any(S._load_screen_keymap().values())


def test_page_aliases_use_qt_portable_sequences_and_remain_live(qtbot):
    screen = A.AnnotateScreen()
    qtbot.addWidget(screen)
    assert screen._spacr_screen_holders["PageUp"].key() == QKeySequence(Qt.Key_PageUp)
    assert screen._spacr_screen_holders["PageDown"].key() == QKeySequence(Qt.Key_PageDown)
    S._save_keymap({}, {"Annotate": {"PageUp": "Ctrl+PgUp"}})
    S._apply_screen_keymaps()
    assert screen._spacr_screen_holders["PageUp"].key() == QKeySequence("Ctrl+PgUp")


def test_keyboard_layout_text_also_obeys_a_rebound_number(qtbot):
    screen = A.AnnotateScreen()
    qtbot.addWidget(screen)
    called = []
    screen._kbd_assign = lambda value: called.append(value) or True
    S._save_keymap({}, {"Annotate": {"2": "F8"}})
    S._apply_screen_keymaps()
    QApplication.sendEvent(screen, QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "2"))
    assert not called
    _event(screen, Qt.Key_F8)
    assert called == [2]


def test_a_deleted_screen_shortcut_does_not_prevent_other_live_updates(qtbot):
    from shiboken6 import delete
    screen = M.MakeMasksScreen()
    qtbot.addWidget(screen)
    delete(screen._spacr_screen_holders["B"])
    S._save_keymap({}, {"Make Masks": {"E": "F8"}})
    S._apply_screen_keymaps()
    assert screen._spacr_screen_holders["E"].key() == QKeySequence("F8")


def test_update_without_an_application_is_a_safe_noop(monkeypatch):
    class NoApplication:
        @staticmethod
        def instance():
            return None
    monkeypatch.setattr(S, "QApplication", NoApplication)
    visited = []
    monkeypatch.setattr(S, "_refresh_screen_hints", visited.append)
    assert S._apply_screen_keymaps() is None
    assert visited == []
