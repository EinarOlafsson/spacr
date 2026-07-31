"""Qt-side plugin settings, dispatch and drag-and-drop integration."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from spacr import plugins


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
    qapp, qt_plugin,
):
    from spacr.qt.screens.settings_model import (
        SettingsWidgets, api_docs_url,
    )
    model = SettingsWidgets("qt_test_assay")
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
