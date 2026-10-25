"""The Activation screen gets its ambient backdrop before the torch import starts.

The attribution rules behind its ``cam_type`` menu import torch on a worker
that holds the heavy-import lock (49c1b07e0). The worker was started while
the page was still being built, so the screen asked for its backdrop with
the lock taken and deferred it until torch had loaded: Activation opened
bare, and ``test_the_real_widget_lands_on_every_module_screen[activation]``
failed whenever it ran first in its process. The worker now starts on the
poll's first tick, after the screen is built.
"""
from __future__ import annotations

import sys
import threading

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import settings_model  # noqa: E402

pytestmark = pytest.mark.qt


def test_the_backdrop_is_built_before_the_attribution_worker_starts(
        qtbot, qt_theme_applied, monkeypatch):
    ambient = pytest.importorskip("spacr.qt.widgets.ambient")
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.delitem(sys.modules, "spacr.attribution", raising=False)
    monkeypatch.setattr(settings_model, "_ATTRIBUTION_IMPORT", [])
    started = []

    def start():
        started.append(ambient._the_heavy_import_lock_is_free())
        thread = threading.Thread(target=lambda: None)
        settings_model._ATTRIBUTION_IMPORT.append(thread)
        thread.start()
        return thread

    monkeypatch.setattr(settings_model,
                        "_import_attribution_off_the_gui_thread", start)
    screen = AppScreen("activation")
    qtbot.addWidget(screen)

    assert started == []
    assert isinstance(screen._ambient, ambient.AmbientWidget)
    qtbot.waitUntil(lambda: bool(started), timeout=5000)
    assert started == [True]
    poll = screen._settings_model._attribution_import_poll
    qtbot.waitUntil(lambda: not poll.isActive(), timeout=5000)
