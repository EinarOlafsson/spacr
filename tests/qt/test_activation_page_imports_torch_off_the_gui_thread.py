"""The Activation page imports its attribution rules off the GUI thread.

Opening the Activation fold froze the window for over a second: greying the
``cam_type`` entries that do not apply asked :mod:`spacr.attribution`, which
imports torch, on the GUI thread. The import now runs on a worker and the
menu is greyed when it finishes, the same way it was before.
"""
from __future__ import annotations

import importlib
import sys
import threading

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("torch")

from PySide6.QtWidgets import QComboBox

from spacr.qt.screens import settings_model

pytestmark = pytest.mark.qt


def test_the_page_imports_the_attribution_rules_on_a_worker(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.screens.activation import build

    monkeypatch.delitem(sys.modules, "spacr.attribution", raising=False)
    monkeypatch.setattr(settings_model, "_ATTRIBUTION_IMPORT", [])
    on_the_gui_thread = []
    real_import = importlib.import_module

    def recording(name, *args, **kwargs):
        if name == "spacr.attribution":
            on_the_gui_thread.append(
                threading.current_thread() is threading.main_thread())
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", recording)
    screen = build(None)
    qtbot.addWidget(screen)
    qtbot.waitUntil(lambda: "spacr.attribution" in sys.modules,
                    timeout=60000)
    assert on_the_gui_thread == [False]

    model = screen._settings_model
    combo = model._built_control("cam_type")
    assert isinstance(combo, QComboBox)
    poll = getattr(model, "_attribution_import_poll", None)
    qtbot.waitUntil(lambda: poll is None or not poll.isActive(),
                    timeout=10000)
    from spacr.attribution import cam_type_applicability

    architecture = model._widgets.get("model_type")
    model_type = str(model._read_widget(architecture) or "").strip() or None
    items = combo.model()
    expected = [cam_type_applicability(
        str(combo.itemData(i) or combo.itemText(i)),
        model_type=model_type)[0] for i in range(combo.count())]
    shown = [items.item(i).isEnabled() for i in range(combo.count())]
    assert shown == expected
