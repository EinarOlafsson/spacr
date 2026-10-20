"""A Make Masks fold breathes while its screen is built, as a window open does.

The Cellpose workbench is two module pages, built when its button is
pressed. Outside a window open those pages were built without the breaths
the window gives a screen it opens, so the press was one freeze of most of
a second. The press now takes the same breaths, and leaves the window's own
breathing switch as it found it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from spacr.qt import screens as screens_package
from spacr.qt.screens.make_masks import MakeMasksScreen

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made._magnifier.close()
    made.close_folded()
    made.close()
    made.deleteLater()
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def _count_breaths(monkeypatch):
    breaths = []
    monkeypatch.setattr(screens_package, "BREATH_AFTER_S", 0.0)
    monkeypatch.setattr(screens_package, "_note_the_breath",
                        lambda worked, breathed: breaths.append(worked))
    return breaths


def test_building_the_cellpose_fold_breathes(screen, monkeypatch):
    breaths = _count_breaths(monkeypatch)
    assert not screens_package._a_window_is_opening_a_screen

    panel = screen.open_folded("train_cellpose")

    assert panel is not None
    assert len(breaths) >= 6
    assert not screens_package._a_window_is_opening_a_screen


def test_a_fold_opened_during_a_window_open_leaves_its_breathing_on(
        screen, monkeypatch):
    _count_breaths(monkeypatch)
    screens_package._start_breathing_while_a_window_opens(0.0)
    try:
        screen.open_folded("model_compare")
        assert screens_package._a_window_is_opening_a_screen
    finally:
        screens_package._stop_breathing_while_a_window_opens()
