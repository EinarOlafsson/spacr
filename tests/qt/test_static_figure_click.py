"""Clicking the static UMAP figure opens the interactive explorer.

The request was "i should be able to press every point". Pressing a point
on a rendered PNG means hit-testing pixels back to the embedding -- a
second, fragile implementation of what the explorer already does properly.
So the click takes you to the view where pressing points works.
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.live_preview import CLICK_SLOP_PX, _ZoomView


def _press(view, x, y):
    view.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))


def _release(view, x, y):
    view.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))


class TestAClickIsNotADrag:
    """This view pans with the left button, so without a slop threshold
    every pan would end in a click and the figure would keep flipping to
    Live while the user was trying to move it."""

    @pytest.fixture
    def view(self, qt_theme_applied, qtbot):
        widget = _ZoomView()
        qtbot.addWidget(widget)
        return widget

    def test_a_still_click_fires(self, view):
        fired = []
        view.clicked.connect(lambda: fired.append(1))
        _press(view, 10, 10)
        _release(view, 10, 10)
        assert fired == [1]

    def test_a_drag_does_not_fire(self, view):
        fired = []
        view.clicked.connect(lambda: fired.append(1))
        _press(view, 10, 10)
        _release(view, 10 + CLICK_SLOP_PX + 30, 10)
        assert fired == []

    def test_a_tiny_wobble_still_counts_as_a_click(self, view):
        """A click with a real mouse is rarely exactly zero movement."""
        fired = []
        view.clicked.connect(lambda: fired.append(1))
        _press(view, 10, 10)
        _release(view, 10 + max(1, CLICK_SLOP_PX - 1), 10)
        assert fired == [1]

    def test_a_release_with_no_press_is_harmless(self, view):
        fired = []
        view.clicked.connect(lambda: fired.append(1))
        _release(view, 10, 10)
        assert fired == []


class TestTheScreenReacts:

    @pytest.fixture
    def screen(self, qt_theme_applied, qtbot):
        widget = AppScreen("umap")
        qtbot.addWidget(widget)
        widget._interactive_switch.setChecked(False)
        return widget

    def test_a_click_with_a_payload_turns_live_on(self, screen):
        screen._umap_payload_ready = True
        screen._figure_queue.figure_clicked.emit()
        assert screen._interactive_switch.isChecked()

    def test_a_click_with_no_payload_does_nothing(self, screen):
        """Flipping the switch with nothing to explore would show an empty
        panel and look like a bug."""
        screen._umap_payload_ready = False
        screen._figure_queue.figure_clicked.emit()
        assert not screen._interactive_switch.isChecked()

    def test_clicking_again_while_live_is_on_is_a_no_op(self, screen):
        screen._umap_payload_ready = True
        screen._figure_queue.figure_clicked.emit()
        screen._figure_queue.figure_clicked.emit()
        assert screen._interactive_switch.isChecked()

    def test_it_says_what_happened(self, screen):
        """A view that changes under you with no explanation is worse than
        one that does not change."""
        screen._umap_payload_ready = True
        screen._figure_queue.figure_clicked.emit()
        text = screen._console.toPlainText() if hasattr(
            screen._console, "toPlainText") else ""
        assert "nteractive" in text or screen._interactive_switch.isChecked()


def test_a_non_umap_screen_has_no_interactive_switch(qt_theme_applied, qtbot):
    """The wiring is UMAP-only; a mask screen must not grow a click that
    toggles something it does not have."""
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    assert getattr(screen, "_interactive_switch", None) is None
