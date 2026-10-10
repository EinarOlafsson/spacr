"""A user keymap over the window shortcuts, and named settings profiles.

The keymap half drives the editor the cheat sheet opens: a key typed there
replaces the default on the live window, survives a fresh read of the store,
and a key two actions would share is named and cannot be saved. The profile
half drives the settings-template dialog's Rename button and its import of a
settings CSV, the file every run writes beside its results.
"""
from __future__ import annotations

import json

import pytest
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QInputDialog, QMainWindow, QFileDialog, QMessageBox

from spacr.qt import recipes as R
from spacr.qt import shortcuts as S


@pytest.fixture
def window(qtbot):
    win = QMainWindow()
    action = QAction("Preferences", win)
    action.setShortcut(QKeySequence("Ctrl+P"))
    win.addAction(action)
    win._prefs_action = action
    qtbot.addWidget(win)
    win.resize(900, 700)
    S.install(win)
    return win


def _live_keys(win):
    return {sc.key().toString(QKeySequence.PortableText)
            for sc in win.findChildren(QShortcut)}


def test_a_rebound_key_is_live_saved_and_shown_on_the_sheet(window, qtbot):
    dlg = S._KeymapDialog(window)
    qtbot.addWidget(dlg)
    dlg.set_key("Ctrl+K", "Ctrl+Shift+K")
    dlg.set_key("Ctrl+P", "Ctrl+Shift+P")
    assert dlg.conflict_text() == ""
    assert dlg.save() is True

    assert S._load_keymap() == {"Ctrl+K": "Ctrl+Shift+K",
                                "Ctrl+P": "Ctrl+Shift+P"}
    live = _live_keys(window)
    assert "Ctrl+Shift+K" in live and "Ctrl+K" not in live
    assert window._prefs_action.shortcut().toString(
        QKeySequence.PortableText) == "Ctrl+Shift+P"

    S.install(window)
    assert sum(1 for sc in window.findChildren(QShortcut)
               if sc.key() == QKeySequence("Ctrl+Shift+K")) == 1

    overlay = S.show_cheat_sheet(window)
    keys = {label.text() for label in overlay.findChildren(
        S.QLabel, "ShortcutOverlayKeys")}
    assert S.native("Ctrl+Shift+K") in keys
    assert overlay.findChild(S.QPushButton, "ShortcutOverlayEdit") is not None
    overlay.dismiss()


def test_a_key_two_actions_share_is_named_and_cannot_be_saved(window, qtbot):
    dlg = S._KeymapDialog(window)
    qtbot.addWidget(dlg)
    dlg.set_key("Ctrl+K", "Ctrl+F")
    assert not dlg._btn_save.isEnabled()
    assert "Open command palette" in dlg.conflict_text()
    assert "Search this module's settings" in dlg.conflict_text()
    assert dlg.save() is False
    assert S._load_keymap() == {}

    dlg.set_key("Ctrl+K", "B")
    assert "Brush" in dlg.conflict_text()

    dlg.restore_defaults()
    assert dlg.conflict_text() == "" and dlg._btn_save.isEnabled()
    with pytest.raises(ValueError):
        S._save_keymap({"Ctrl+K": "Ctrl+F"})


def test_a_cleared_key_unbinds_and_defaults_come_back(window, qtbot):
    dlg = S._KeymapDialog(window)
    qtbot.addWidget(dlg)
    dlg.set_key("Ctrl+K", "")
    assert dlg.save()
    assert S._load_keymap() == {"Ctrl+K": ""}
    assert "Ctrl+K" not in _live_keys(window)

    again = S._KeymapDialog(window)
    qtbot.addWidget(again)
    again.restore_defaults()
    assert again.save()
    assert S._load_keymap() == {}
    assert "Ctrl+K" in _live_keys(window)


def test_saved_overrides_bind_once_on_each_new_window(qtbot, monkeypatch):
    class Store:
        data = {}

        def value(self, key, default=""):
            return self.data.get(key, default)

        def setValue(self, key, value):
            self.data[key] = value

    store = Store()
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    monkeypatch.setattr(S, "_install_window_hooks", lambda window: None)
    S._save_keymap({"Ctrl+K": "Ctrl+Shift+K", "Ctrl+P": "Ctrl+Shift+P"})

    for _ in range(2):
        win = QMainWindow()
        qtbot.addWidget(win)
        preferences = QAction("Preferences", win)
        preferences.setShortcut(QKeySequence("Ctrl+P"))
        win.addAction(preferences)
        S.install(win)
        palette = [sc for sc in win.findChildren(QShortcut)
                   if sc.key() == QKeySequence("Ctrl+Shift+K")]
        assert len(palette) == 1
        assert "Ctrl+K" not in _live_keys(win)
        assert preferences.shortcut() == QKeySequence("Ctrl+Shift+P")

        S.install(win)
        assert [sc for sc in win.findChildren(QShortcut)
                if sc.key() == QKeySequence("Ctrl+Shift+K")] == palette
        assert preferences.shortcut() == QKeySequence("Ctrl+Shift+P")


def test_stale_saved_action_is_ignored_and_screen_key_cannot_be_taken(
        monkeypatch):
    class Store:
        data = {S._KEYMAP_KEY: json.dumps({
            "Ctrl+K": "Ctrl+Shift+K", "Ctrl+Q": "Ctrl+Alt+Q"})}

        def value(self, key, default=""):
            return self.data.get(key, default)

        def setValue(self, key, value):
            self.data[key] = value

    store = Store()
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    assert S._load_keymap() == {"Ctrl+K": "Ctrl+Shift+K"}

    with pytest.raises(ValueError, match="Undo"):
        S._save_keymap({"Ctrl+K": "Ctrl+Z"})
    assert S._load_keymap() == {"Ctrl+K": "Ctrl+Shift+K"}


def test_deleted_shortcut_or_menu_holder_is_replaced_and_rebound(
        qtbot):
    from shiboken6 import delete

    win = QMainWindow()
    qtbot.addWidget(win)
    old = S._bind(win, "Ctrl+K", lambda: None)
    assert S._holders(win)["Ctrl+K"] is old
    old_action = QAction("Preferences", win)
    old_action.setShortcut(QKeySequence("Ctrl+P"))
    win.addAction(old_action)
    assert S._holders(win)["Ctrl+P"] is old_action

    delete(old)
    delete(old_action)
    replacement = S._bind(win, "Ctrl+K", lambda: None)
    replacement_action = QAction("Preferences", win)
    replacement_action.setShortcut(QKeySequence("Ctrl+P"))
    win.addAction(replacement_action)
    assert replacement is not old
    assert S._holders(win)["Ctrl+K"] is replacement
    assert S._holders(win)["Ctrl+P"] is replacement_action
    assert S._apply_keymap(win, {"Ctrl+K": "Ctrl+Shift+K",
                                 "Ctrl+P": "Ctrl+Shift+P"}) == 2
    assert replacement.key() == QKeySequence("Ctrl+Shift+K")
    assert replacement_action.shortcut() == QKeySequence("Ctrl+Shift+P")


def test_a_broken_saved_keymap_does_not_prevent_default_registration(
        qtbot, monkeypatch):
    win = QMainWindow()
    qtbot.addWidget(win)
    monkeypatch.setattr(S, "_install_window_hooks", lambda window: None)

    def broken_keymap(window):
        raise RuntimeError("settings store unavailable")

    monkeypatch.setattr(S, "_apply_keymap", broken_keymap)
    S.install(win)
    assert {"Ctrl+K", "Ctrl+F", "F1"} <= _live_keys(win)


def test_the_cheat_sheet_button_opens_the_editor(window, qtbot):
    overlay = S.show_cheat_sheet(window)
    overlay.findChild(S.QPushButton, "ShortcutOverlayEdit").click()
    dialogs = window.findChildren(S._KeymapDialog)
    assert dialogs and dialogs[-1].isVisible()
    rows = dialogs[-1]._table.rowCount()
    assert rows == len(S._rebindable()) + sum(
        len(S._screen_specs(scope)) for scope in S._SCREEN_SCOPES)
    assert {item.text() for item in (dialogs[-1]._table.item(row, 3)
                                   for row in range(rows))} >= {
        S.EVERYWHERE, "the Annotate screen", "the Make Masks screen",
        "the QC field browser"}
    dialogs[-1].reject()


@pytest.fixture(autouse=True)
def _isolated_recipes(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_RECIPE_DIR", str(tmp_path / "recipes"))


@pytest.fixture
def profiles(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    dlg = R.RecipeDialog(screen)
    qtbot.addWidget(dlg)
    return screen, dlg


def test_a_profile_is_renamed_and_its_old_file_is_gone(profiles, monkeypatch):
    screen, dlg = profiles
    old = R.save_recipe(R.capture_recipe(screen, "Plate A"))
    dlg.reload()
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Toxo 40x", True)))
    dlg._on_rename()
    assert [r.name for r in R.list_recipes("mask")] == ["Toxo 40x"]
    assert not __import__("os").path.exists(old)


def test_a_rename_onto_an_existing_profile_is_refused(profiles, monkeypatch):
    screen, dlg = profiles
    R.save_recipe(R.capture_recipe(screen, "One"))
    R.save_recipe(R.capture_recipe(screen, "Two"))
    dlg.reload()
    seen = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda _p, title, text, *a, **k: seen.append(text)))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(
        lambda *a, **k: ("Two" if dlg.selected().name == "One" else "One",
                         True)))
    dlg._on_rename()
    assert sorted(r.name for r in R.list_recipes("mask")) == ["One", "Two"]
    assert seen and "already exists" in seen[0]


def test_a_settings_csv_imports_as_a_named_profile(profiles, monkeypatch,
                                                   tmp_path):
    screen, dlg = profiles
    csv = tmp_path / "plate_b_settings.csv"
    csv.write_text("Key,Value\nn_jobs,5\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(csv), "")))
    dlg._on_import()
    (recipe,) = R.list_recipes("mask")
    assert recipe.name == "plate_b_settings"
    assert str(recipe.settings["n_jobs"]) == "5"

    exported = tmp_path / "shared.json"
    dlg._list.setCurrentRow(0)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(exported), "")))
    dlg._on_export()
    data = json.loads(exported.read_text(encoding="utf-8"))
    assert data["name"] == "plate_b_settings" and data["app_key"] == "mask"
