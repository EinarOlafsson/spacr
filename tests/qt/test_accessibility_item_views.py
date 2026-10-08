"""Lists and tables expose translated names through Qt accessibility."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QHeaderView, QListView, QScrollBar,
    QTableView, QTabWidget, QTreeView, QVBoxLayout, QWidget,
)
from shiboken6 import getCppPointer

from spacr.qt import i18n, preferences
from spacr.qt.app import MainWindow
from spacr.qt.widgets.section import Section


def _name(widget):
    interface = QAccessible.queryAccessibleInterface(widget)
    return str(interface.text(QAccessible.Name) or "") if interface else ""


@pytest.mark.parametrize("constructor, object_name, source", [
    (QTableView, "Objects", "Objects"),
    (QListView, "Masks", "Masks"),
    (QTableView, "SettingsDiff", "Settings diff"),
    (QTableView, "StorageCacheTable", "Cache"),
    (QTableView, "DataManagerUsage", "Storage"),
    (QListView, "GraphColumnList", "Columns"),
])
def test_derived_view_names_follow_every_language(
        qtbot, monkeypatch, constructor, object_name, source):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    view = constructor(host)
    view.setObjectName(object_name)
    layout.addWidget(view)
    for language in (*i18n._TRANSLATED_CODES, "en"):
        monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
        i18n.retranslate_widget_tree(host, language)
        assert _name(view) == i18n.tr(source, language)
        if language != "en":
            assert _name(view) != source


def test_view_naming_preserves_an_explicit_application_name(qtbot):
    view = QTreeView()
    qtbot.addWidget(view)
    view.setObjectName("OtherName")
    view.setAccessibleName("Explicit application name")
    i18n.retranslate_widget_tree(view)
    assert _name(view) == "Explicit application name"


def test_item_view_families_include_lists_tables_trees_and_exclude_headers():
    interactive = i18n._a11y_interactive_types()
    assert all(any(issubclass(kind, base) for base in interactive)
               for kind in (QListView, QTableView, QTreeView))
    assert not any(issubclass(QHeaderView, base) for base in interactive)


def test_home_and_every_preferences_tab_name_visible_non_header_controls(
        qtbot, monkeypatch):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_ambient_animation("none")
    preferences.set_preload_policy("on_demand")
    preferences._set_restore_session(False)
    QSettings(*preferences._store_args("spacr", "qt")).setValue(
        "onboarding/first_run_tour_seen", True)
    main = MainWindow()
    qtbot.addWidget(main)
    main.resize(1400, 900)
    main.show()
    dialog = preferences.PreferencesDialog(main)
    qtbot.addWidget(dialog)
    dialog.resize(1100, 1000)
    dialog.show()
    for section in dialog.findChildren(Section):
        section.set_expanded(True)
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    assert tabs is not None and tabs.count() == 8
    checked = set()

    def check(host):
        i18n.retranslate_widget_tree(host)
        QApplication.processEvents()
        for widget in host.findChildren(QWidget):
            if not widget.isVisibleTo(host):
                continue
            if isinstance(widget, (QHeaderView, QScrollBar)):
                continue
            if not isinstance(widget, i18n._a11y_interactive_types()):
                continue
            checked.add(getCppPointer(widget)[0])
            assert _name(widget).strip(), (
                type(widget).__name__, widget.objectName())

    check(main)
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        check(dialog)
    storage = dialog.findChild(QAbstractItemView, "StorageCacheTable")
    assert storage is not None and _name(storage).strip()
    assert checked
    dialog.reject()
