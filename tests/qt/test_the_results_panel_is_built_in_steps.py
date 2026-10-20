"""Regression's results panel can be constructed a part at a time.

Its construction -- a dozen plots, two tables, the gene tile -- was one
step of the deferred results build, 0.2-0.3 s on the GUI thread on its own
in lesson 13. Built through the stepped results build it now gives the
event loop back between its larger parts, and ends as the same panel.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from spacr.qt.screens import app_screen
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.regression_results import RegressionResultsPanel

pytestmark = pytest.mark.qt


def _tabs(panel):
    return [panel.tabs.tabText(i) for i in range(panel.tabs.count())]


def test_the_stepped_panel_pauses_and_matches_one_built_at_once(qtbot):
    whole = RegressionResultsPanel(external_volcano=True)
    qtbot.addWidget(whole)

    steps = RegressionResultsPanel._constructed_in_steps(
        external_volcano=True)
    pauses = 0
    try:
        while True:
            next(steps)
            pauses += 1
    except StopIteration as done:
        panel = done.value
    qtbot.addWidget(panel)

    assert pauses >= 3
    assert isinstance(panel, RegressionResultsPanel)
    assert _tabs(panel) == _tabs(whole)
    assert panel.external_volcano is True
    assert panel.table.table.columnCount() == whole.table.table.columnCount()


def test_a_panel_built_directly_is_whole(qtbot):
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    assert panel.support is not None and panel.gene is not None
    assert panel._load_jobs is not None


def test_the_screen_steps_through_the_panel(qtbot, qt_theme_applied,
                                            monkeypatch):
    steps = []
    real = AppScreen._step_owed_part

    def counted(self, part):
        steps.append(part)
        return real(self, part)

    monkeypatch.setattr(AppScreen, "_step_owed_part", counted)
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    part = app_screen._REGRESSION_RESULTS

    assert screen._build_owed_part_in_steps(part)
    qtbot.waitUntil(lambda: not screen._part_is_owed(part), timeout=20000)

    assert steps.count(part) >= 9
    assert isinstance(screen._results_panel, RegressionResultsPanel)
    assert screen._results_panel.tabs.count() > 5
