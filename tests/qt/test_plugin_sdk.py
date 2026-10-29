"""Qt-side plugin settings, dispatch and drag-and-drop integration."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from spacr import plugins
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def qt_plugin(monkeypatch):
    module = types.ModuleType("spacr_qt_test_plugin")

    def defaults(settings=None):
        result = {"src": "", "threshold": 0.5, "save_table": True}
        result.update(settings or {})
        return result

    def run(settings):
        return settings["threshold"]

    from spacr.qt.dnd import DropHandler

    class PluginDropHandler(DropHandler):
        def can_accept(self, path: Path) -> bool:
            return path.is_dir()

        def apply(self, path: Path, screen) -> None:
            screen.accepted_path = str(path)

    module.defaults = defaults
    module.run = run
    module.PluginDropHandler = PluginDropHandler
    module.plugin = {
        "name": "Qt test plugin",
        "version": "1.0",
        "apps": [{
            "key": "qt_test_assay",
            "name": "Qt Test Assay",
            "description": "Exercise the generic plugin settings screen.",
            "entrypoint": "spacr_qt_test_plugin:run",
            "defaults": "spacr_qt_test_plugin:defaults",
            "section": "results",
            "categories": {
                "Input": ["src"],
                "Analysis": ["threshold"],
                "Output": ["save_table"],
            },
            "tooltips": {"threshold": "Probability required to retain an object."},
            "labels": {"threshold": "Retention threshold"},
            "docs_url": "https://example.invalid/qt-test-api",
            "drop_handler": "spacr_qt_test_plugin:PluginDropHandler",
        }],
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_qt_test_plugin:plugin")
    plugins.reload_plugins()
    yield module
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    sys.modules.pop(module.__name__, None)
    plugins.reload_plugins()


def test_plugin_settings_use_declared_tabs_labels_tooltips_and_docs(
    qapp, qtbot, qt_plugin,
):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens.settings_model import (
        SettingsWidgets,
        api_docs_url,
    )
    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets("qt_test_assay", parent=owner)
    sections = model.build_sections()
    assert [name for name, _rows in sections] == ["Input", "Analysis", "Output"]
    labels = [label for _name, rows in sections for label, _widget in rows]
    assert "Retention threshold" in labels
    assert "Probability required" in model.plain_tooltip_for("threshold")
    assert api_docs_url("qt_test_assay", "threshold") == (
        "https://example.invalid/qt-test-api"
    )


def test_plugin_pipeline_and_drop_handler_are_loaded_lazily(qapp, qt_plugin):
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.dnd_handlers import get_handler

    entry = resolve_pipeline_entry("qt_test_assay")
    assert callable(entry)
    assert entry({"threshold": 0.73}) == pytest.approx(0.73)

    handler = get_handler("qt_test_assay")
    assert type(handler).__name__ == "PluginDropHandler"


def test_plugin_with_empty_artwork_keeps_native_navigator_card_and_explicit_open(
        qtbot, qt_plugin, private_ai_profile, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    from spacr.qt import app

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    artwork = tmp_path / 'plugin.png'
    artwork.write_bytes(b'')
    contribution = qt_plugin.plugin['apps'][0]
    contribution['icon'] = str(artwork)
    contribution['screen_factory'] = 'spacr_qt_test_plugin:screen'
    opened, analyses = [], []

    def run(settings):
        analyses.append(settings)
        return 0

    def screen(*, app_key):
        opened.append(app_key)
        widget = QWidget()
        widget.setObjectName('PluginNativeView')
        return widget

    qt_plugin.run = run
    qt_plugin.screen = screen
    plugins.reload_plugins()
    record = plugins.get_app('qt_test_assay')
    assert record is not None and record.icon == str(artwork)
    registered = any(row[0] == record.key for row in app.APPS)
    if not registered:
        app.register_app(record.key, record.name, record.description,
                         app._PLUGIN_SECTION_MAP[record.section], stage=record.stage)
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1180, 850)
    window.show()
    try:
        page = window._startup
        page.path_edit.setText('Qt Test Assay')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: any(card._key == record.key for card in page.field.cards), timeout=5000)
        card = next(card for card in page.field.cards if card._key == record.key)
        assert not card._module_icon.isNull() and not card._module_icon.pixmap(64, 64).isNull()
        assert opened == [] and window._stack.currentWidget() is page
        qtbot.mouseClick(card, Qt.LeftButton)
        assert page._workflow is not None and opened == []
        qtbot.mouseClick(page._workflow.central, Qt.LeftButton)
        qtbot.waitUntil(lambda: record.key in window._screens, timeout=5000)
        assert opened == [record.key] and analyses == []
        assert window._stack.currentWidget().objectName() == 'PluginNativeView'
        assert artwork.read_bytes() == b''
    finally:
        window.close()
        if not registered:
            app.unregister_app(record.key)
