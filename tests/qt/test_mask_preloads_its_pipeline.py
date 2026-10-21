"""Opening Mask imports its pipeline in the background, so Run imports nothing.

Mask's Run imported ``spacr.core`` (torch, cellpose) at the click, and that
import starved the backdrop for a second (tutorial 07, 2026-10-17). The
maintainer accepted preloading it soon after Mask first opens.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")


def test_mask_open_preloads_so_run_needs_no_import(qtbot, qt_theme_applied,
                                                   monkeypatch):
    from spacr.qt import app as app_module
    from spacr.qt import bridge
    from spacr.qt.app import MainWindow

    imported = []

    def fake_import(name):
        imported.append((name, threading.current_thread() is threading.main_thread()))

    monkeypatch.setattr(bridge, "_ENTRY_IMPORTS", {})
    monkeypatch.setattr(bridge, "_module_is_imported", lambda _name: False)
    monkeypatch.setattr(bridge, "_import_the_entry_module", fake_import)
    monkeypatch.setattr(app_module, "_ENTRY_PRELOAD_AFTER_S", 0.05)

    assert bridge._entry_import_pending("mask"), "nothing to preload"
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.open_module("mask")

    qtbot.waitUntil(lambda: bool(imported), timeout=5000)
    qtbot.waitUntil(lambda: not bridge._entry_import_pending("mask"), timeout=5000)
    assert imported == [(bridge._ENTRY_MODULES["mask"], False)]

    # A second preload, or the Run that follows, starts no new import.
    assert app_module._preload_the_pipeline_entry("mask") is None
    assert bridge._import_the_entry_off_the_gui_thread("mask") is None
    assert len(imported) == 1
