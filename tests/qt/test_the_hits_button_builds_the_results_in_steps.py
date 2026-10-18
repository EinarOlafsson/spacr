"""The first press of Regression's Hits button builds the results in steps.

Regression defers its results panel -- about 740 widgets -- to first use.
The Hits button is such a use, and built in one go the panel, the figure
pages, the run and measurement tabs and the language pass over them were one
freeze of over a second. The button now builds them one step per event-loop
turn and raises the Hits tab once they are finished. A reader of any of the
deferred attributes in between still gets the finished part at once.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from spacr.qt.screens import app_screen, regression
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt

PART = app_screen._REGRESSION_RESULTS


def _screen(qtbot):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    screen.resize(1500, 1000)
    screen.show()
    strip = regression.install_folds(screen)
    assert strip is not None
    assert screen._part_is_owed(PART)
    return screen, strip


def test_the_first_press_builds_over_several_event_loop_turns(
        qtbot, qt_theme_applied, monkeypatch):
    steps = []
    real = AppScreen._step_owed_part

    def counted(self, part):
        steps.append(part)
        return real(self, part)

    monkeypatch.setattr(AppScreen, "_step_owed_part", counted)
    screen, strip = _screen(qtbot)

    qtbot.mouseClick(strip.button_for("hit_list"), Qt.LeftButton)
    assert screen._part_is_owed(PART)
    assert screen._results_panel_if_built() is None

    qtbot.waitUntil(lambda: not screen._part_is_owed(PART), timeout=20000)
    assert steps.count(PART) >= 6
    panel = screen._results_panel_if_built()
    assert panel is not None
    qtbot.waitUntil(lambda: panel.tabs.currentWidget() is panel.hits,
                    timeout=3000)
    assert screen._figures_card.isVisible()


def test_a_reader_between_the_steps_gets_the_finished_part(
        qtbot, qt_theme_applied):
    screen, strip = _screen(qtbot)

    qtbot.mouseClick(strip.button_for("hit_list"), Qt.LeftButton)
    QApplication.processEvents()
    assert screen._part_is_owed(PART)

    assert screen._figures_split is not None
    assert not screen._part_is_owed(PART)
    panel = screen._results_panel
    assert panel.hits is not None
    assert panel.tabs.currentWidget() is panel.hits
