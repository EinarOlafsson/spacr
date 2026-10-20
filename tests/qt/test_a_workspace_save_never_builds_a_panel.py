"""Saving the workspace after a run must not build the Regression panels.

A run of another module (Illumination) collects the sections of an unopened
Regression screen too. The run journal asks every workspace provider from the thread that ran the
pipeline. A provider that read a deferred panel with ``getattr`` built it there,
and the thread guard printed a traceback with source paths in the console.
"""
from __future__ import annotations

import threading

import pytest

from spacr import workspace
from spacr.qt.screens.app_screen import AppScreen, _REGRESSION_RESULTS

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    workspace.clear_providers()
    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    widget.register_workspace()
    try:
        yield widget
    finally:
        widget.unregister_workspace()
        workspace.clear_providers()


def test_the_worker_thread_collects_without_building_the_panel(screen, capsys,
                                                               qtbot):
    assert screen._part_is_owed(_REGRESSION_RESULTS)
    result = {}

    def worker():
        result["doc"] = workspace.collect(workspace.providers(),
                                          app_key="illumination",
                                          hash_files=False)

    thread = threading.Thread(target=worker)
    thread.start()
    # The sections are read on the GUI thread, so this one keeps its event
    # loop turning while the worker waits for them.
    qtbot.waitUntil(lambda: not thread.is_alive(), timeout=15000)
    assert screen._part_is_owed(_REGRESSION_RESULTS)
    sections = result["doc"]["sections"]
    assert "regression:settings" in sections
    assert "regression:regression" not in sections
    assert "regression:montage" not in sections
    assert "Traceback" not in capsys.readouterr().err


def test_a_built_panel_is_still_saved(screen):
    screen._build_owed_part(_REGRESSION_RESULTS)
    panel = screen._if_built("_results_panel")
    sections, _problems = workspace.section_states(workspace.providers())
    assert (panel is None) == ("regression:regression" not in sections)
