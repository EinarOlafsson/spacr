"""Keymap, Jobs panel, recipe and external-mask helpers at their edges."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import shortcuts as sc  # noqa: E402


def test_values_that_cannot_compare_are_different():
    class _Odd:
        def __eq__(self, other):
            raise TypeError("incomparable")

    assert sc._same(_Odd(), 1) is False


def test_undo_keys_without_a_redo_slot_bind_only_undo(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    shortcuts = sc._bind_undo_keys(widget, lambda: None, None)
    assert shortcuts and len(shortcuts) == len(sc._UNDO_KEYS)


def test_unreadable_keymaps_read_empty(monkeypatch):
    from PySide6.QtGui import QKeySequence

    monkeypatch.setattr(QKeySequence, "toString",
                        lambda self, *a: (_ for _ in ()).throw(RuntimeError()))
    assert sc._portable("Ctrl+S") == ""
    monkeypatch.undo()
    stored = {}
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: SimpleNamespace(
        value=lambda key, default="": stored.get(key, default)))
    stored[sc._KEYMAP_KEY] = "{not json"
    assert sc._load_keymap() == {}
    stored[sc._KEYMAP_KEY] = json.dumps(["Ctrl+S"])
    assert sc._load_keymap() == {}


def test_holders_survive_windows_that_refuse_attributes_or_searches():
    class _Locked:
        def __setattr__(self, name, value):
            raise AttributeError("slots only")

    assert sc._holders(_Locked()) == {}

    class _Broken:
        def findChildren(self, *a, **k):
            raise RuntimeError("deleted")

    assert sc._holders(_Broken()) == {}


def test_a_deleted_holder_is_skipped_when_applying(monkeypatch):
    class _Gone:
        def setShortcut(self, sequence):
            raise RuntimeError("deleted")

    monkeypatch.setattr(sc, "_holders", lambda window: {"Ctrl+S": _Gone()})
    assert sc._apply_keymap(object(), {}) == 0


def test_reinstalling_a_deleted_shortcut_creates_one_live_binding(qtbot):
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import QMainWindow
    import shiboken6

    window = QMainWindow()
    qtbot.addWidget(window)
    activated = []
    old = sc._bind(window, "Ctrl+K", lambda: activated.append("old"))
    window._spacr_keymap_holders = {"Ctrl+K": old}
    shiboken6.delete(old)

    replacement = sc._bind(window, "Ctrl+K", lambda: activated.append("new"))
    assert replacement is not old
    assert replacement.key() == QKeySequence("Ctrl+K")
    assert window.findChildren(QShortcut) == [replacement]
    replacement.activated.emit()
    assert activated == ["new"]


def test_the_jobs_panel_without_a_registry_and_its_edge_paths(qtbot, monkeypatch):
    import spacr.qt.bridge as bridge
    from spacr.qt.widgets.activity_spinner import _JobsPanel

    def broken():
        raise RuntimeError("no registry yet")

    monkeypatch.setattr(bridge, "registry", broken)
    panel = _JobsPanel()
    qtbot.addWidget(panel)
    assert panel._visible_handles() == []
    panel._handles = [SimpleNamespace(elapsed=lambda: 5.0)]
    panel._table.setRowCount(1)
    panel._update_elapsed()
    panel._table.setItem(0, 2, panel._table.item(0, 0) or __import__(
        "spacr.qt.widgets.sortable_table", fromlist=["table_item"]).table_item(""))
    panel._update_elapsed()
    assert panel._table.item(0, 2).text() == "0:00:05"
    panel._rebuild_timer.start()
    panel._schedule_refresh()
    refused = SimpleNamespace(request_cancel=lambda why: (_ for _ in ()).throw(
        RuntimeError("gone")))
    from PySide6.QtWidgets import QPushButton

    button = QPushButton()
    qtbot.addWidget(button)
    panel._cancel(refused, button)
    assert id(refused) not in panel._cancelled


def test_recipe_names_and_settings_files_are_checked(tmp_path, monkeypatch):
    from spacr.qt import recipes as rc

    with pytest.raises(ValueError):
        rc._rename_recipe(SimpleNamespace(path="", name="a"), "  ")
    import spacr.utils as utils

    monkeypatch.setattr(utils, "load_settings", lambda path, **k: {})
    with pytest.raises(ValueError):
        rc._recipe_from_settings_csv(str(tmp_path / "s.csv"), "mask", screen=None)


def test_the_rename_button_needs_a_selection_and_a_new_name(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    from spacr.qt import recipes as rc

    renamed = []
    monkeypatch.setattr(rc, "_rename_recipe",
                        lambda recipe, name: renamed.append(name))
    dialog = SimpleNamespace(selected=lambda: None)
    rc.RecipeDialog._on_rename(dialog)
    recipe = SimpleNamespace(name="same", path="")
    dialog = SimpleNamespace(selected=lambda: recipe)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("same", True)))
    rc.RecipeDialog._on_rename(dialog)
    assert renamed == []


def test_external_mask_test_data_fills_the_destination(qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget

    images = tmp_path / "images"
    images.mkdir()
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "a.tif").write_bytes(b"")
    written = {}

    class _Model:
        _widgets = {"dst": object()}

        def _read_widget(self, widget):
            return ""

        def set_value_for_key(self, key, value):
            written[key] = value

    host = QWidget()
    qtbot.addWidget(host)
    host._settings_model = _Model()
    middle = QWidget(host)
    widget = ExternalMaskInputWidget(parent=middle)
    monkeypatch.setattr(widget, "add_paths", lambda paths: len(paths))
    assert widget._use_test_data({"images": images,
                                  "masks": {"cell": masks / "a.tif"}}) == 2
    assert written == {"dst": str(images) + "_spacr"}
