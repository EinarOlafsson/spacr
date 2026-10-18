"""A Run on Regression builds the deferred results in steps, then records it.

Pressing Run with the results part still owed read the Runs tab, which
built the whole part -- about 740 widgets and a language pass -- in one
freeze of about a second while lesson 13's batch started. The part is now
built one step per event-loop turn and the run is recorded on the Runs tab
when it is finished, with the same label, source and settings.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import app_screen
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt

PART = app_screen._REGRESSION_RESULTS


def _screen(qtbot):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    screen.resize(1500, 1000)
    screen.show()
    assert screen._part_is_owed(PART)
    return screen


def test_the_record_waits_for_a_stepped_build(qtbot, qt_theme_applied,
                                              monkeypatch):
    steps = []
    real = AppScreen._step_owed_part

    def counted(self, part):
        steps.append(part)
        return real(self, part)

    monkeypatch.setattr(AppScreen, "_step_owed_part", counted)
    screen = _screen(qtbot)
    settings = {"src": "/tmp/somewhere", "inference": "nonparametric"}

    handle = screen._record_run_in_runs_tab("Run  01:02:03", "run", settings)
    settings["hash_inputs"] = True

    assert handle is None
    assert screen._part_is_owed(PART), "the record built the part in one go"
    assert screen._results_panel_if_built() is None

    qtbot.waitUntil(lambda: getattr(screen, "_run_handle", None) is not None,
                    timeout=20000)
    assert not screen._part_is_owed(PART)
    assert steps.count(PART) >= 6
    assert screen._update_run_in_runs_tab(n_results=3)


def test_a_built_part_records_at_once(qtbot, qt_theme_applied):
    screen = _screen(qtbot)
    assert screen._sweep_runs is not None

    handle = screen._record_run_in_runs_tab("Run  01:02:03", "run",
                                            {"src": "/tmp/somewhere"})

    assert handle is not None
