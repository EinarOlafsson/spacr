"""Redraws of the top-group lists and tables rewrite their items, never delete them."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtGui import QAccessible
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDialog, QFormLayout

from spacr.layers import LayerStack, Spacing


@pytest.fixture
def accessible():
    was = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        yield
    finally:
        QAccessible.setActive(was)


def _walk(view):
    iface = QAccessible.queryAccessibleInterface(view)
    names = []
    for i in range(iface.childCount()):
        names.append(iface.child(i).text(QAccessible.Name))
    return names


class _Seen:
    def __init__(self, view):
        self.view = view
        self.items = []

    def take(self):
        _walk(self.view)
        for i in range(self.view.count()):
            item = self.view.item(i)
            if not any(item is seen for seen in self.items):
                self.items.append(item)

    def all_valid(self):
        return all(shiboken6.isValid(item) for item in self.items)


def test_counting_tool_class_list(qtbot, qt_theme_applied, accessible):
    from spacr.qt import counting_tool as ct
    from spacr.qt import layer_viewer as lv

    stack = LayerStack()
    stack.add_image(np.zeros((32, 32), np.uint16), name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    panel = ct.CountingPanel(canvas)
    qtbot.addWidget(panel)
    seen = _Seen(panel.class_list)
    seen.take()
    for name in ("alpha", "beta", "gamma"):
        panel.add_class(name)
        seen.take()
        assert seen.all_valid()
    count = panel.class_list.count()
    panel.undo()
    panel.clear()
    seen.take()
    assert seen.all_valid()
    assert panel.class_list.count() == count
    texts = [panel.class_list.item(i).text() for i in range(count)]
    assert any("gamma" in t for t in texts)
    assert panel.class_list.currentRow() == panel.session.class_names.index(
        panel.session.active)


def test_layer_list(qtbot, qt_theme_applied, accessible):
    from spacr.qt import layer_viewer as lv

    stack = LayerStack()
    for name in ("a", "b", "c"):
        stack.add_image(np.zeros((8, 8), np.uint16), name=name,
                        spacing=Spacing.isotropic(2, 1.0, units="px"))
    listing = lv.LayerListWidget(stack)
    qtbot.addWidget(listing)
    seen = _Seen(listing)
    seen.take()
    assert [listing.item(i).text() for i in range(3)] == ["c", "b", "a"]
    stack.add_image(np.zeros((8, 8), np.uint16), name="d",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    seen.take()
    stack.select(stack["a"])
    seen.take()
    assert listing.currentItem().text() == "a"
    stack["b"].visible = False
    seen.take()
    assert listing.item(listing.count() - 2).checkState() == Qt.Unchecked
    stack.remove(stack["d"])
    seen.take()
    assert listing.count() == 3
    assert seen.all_valid()
    assert listing.item(0).flags() & Qt.ItemIsEditable


def test_recipes_list(qtbot, qt_theme_applied, accessible, tmp_path, monkeypatch):
    from spacr.qt import recipes as R
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setenv("SPACR_RECIPE_DIR", str(tmp_path / "recipes"))
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    dialog = R.RecipeDialog(screen)
    qtbot.addWidget(dialog)
    seen = _Seen(dialog._list)
    for i in range(3):
        recipe = R.capture_recipe(screen, f"template {i}")
        recipe.created = f"2026-01-0{i + 1}T00:00:00"
        R.save_recipe(recipe)
        dialog.reload()
        seen.take()
        assert dialog._list.count() == i + 1
        assert dialog._list.currentRow() == 0
        assert dialog.selected().path == dialog._list.item(0).data(Qt.UserRole)
    dialog.reload()
    seen.take()
    assert seen.all_valid()


def test_help_search_list(qtbot, qt_theme_applied, accessible):
    from spacr.qt.app import MainWindow
    from spacr.qt.help_index import HelpEntry
    from spacr.qt.help_search import MORE_ROLE, field_of

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    field = field_of(win)
    rows = [HelpEntry(kind="setting", title="zzsrc", subtitle=f"M{i}",
                      payload={"app": "mask", "key": "zzsrc", "category": ""})
            for i in range(20)]
    rows.append(HelpEntry(kind="module", title="Zzsrc module",
                          subtitle="Module", payload={"app": "mask"}))
    field.set_index(rows)
    seen = _Seen(field._list)
    for query in ("zzsrc", "Zzsrc module", "zzsrc", "nothing-like-this", "zzsrc"):
        field.type_and_search(query)
        seen.take()
        assert seen.all_valid()
    field.type_and_search("zzsrc")
    texts = [field._list.item(i).text() for i in range(field._list.count())]
    assert "and 12 more" in texts
    assert field._list.item(texts.index("and 12 more")).data(MORE_ROLE) == "setting"
    assert field._list.currentRow() == 0
    field.type_and_search("")
    assert field._list.count() == 0
    assert seen.all_valid()


def test_plugin_catalogue_table(qtbot, monkeypatch, tmp_path, accessible):
    from spacr.qt import preferences
    from tests.test_plugin_catalogue import _catalogue

    source = tmp_path / "catalogue"
    source.mkdir()
    _catalogue(source)
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "prefs.ini"), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_PLUGIN_HOME", str(tmp_path / "plugins"))
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    monkeypatch.delenv("SPACR_DISABLE_PLUGINS", raising=False)
    preferences._set_plugin_catalogue(str(source))
    preferences._set_show_alpha_features(True)
    dialog = QDialog()
    qtbot.addWidget(dialog)
    page = preferences._PluginCataloguePage(QFormLayout(dialog), dialog)
    qtbot.waitUntil(lambda: page._job is None)
    seen = []

    def snap():
        _walk(page.table)
        for r in range(page.table.rowCount()):
            for c in range(page.table.columnCount()):
                item = page.table.item(r, c)
                if item is not None and not any(item is s for s in seen):
                    seen.append(item)

    snap()
    rows = page.table.rowCount()
    assert rows > 0
    for _ in range(3):
        assert page.refresh()
        assert page.table.rowCount() == 0
        qtbot.waitUntil(lambda: page._job is None)
        snap()
        assert page.table.rowCount() == rows
        assert page.table.item(0, 0).data(Qt.UserRole)
    assert all(shiboken6.isValid(item) for item in seen)
    page._select_key(page._rows[0]["key"])
    assert page.selected() is not None
