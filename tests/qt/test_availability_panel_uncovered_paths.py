"""Dismissing the panel, and the two places a press can come from.

The panel is a frameless top-level that watches the whole application for a
press outside itself. That watch has to work from a real ``QMouseEvent``,
which carries the position it happened at, and from an event that carries no
position at all -- a synthetic press, or one a platform plugin delivers as a
bare ``QEvent`` -- where the pointer's own position is the only source left.
Getting the second one wrong raises inside an application-wide event filter,
which is the worst place in Qt for an exception to appear.

The corridor clock and the screen lookup are the other two edges: the grace
period is measured from the FIRST hide request, not the latest, and a machine
with no screen attached still has to be able to put the panel somewhere.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt.widgets import availability_panel as ap
from spacr.qt.widgets.availability_panel import AvailabilityPanel

pytestmark = pytest.mark.qt


class _Offer:
    """The ``InstallOffer`` surface this module reads."""

    def __init__(self, action="installable"):
        self.action = action
        self.requirement = "cuml-cu12"
        self.command = ["pip", "install", "cuml-cu12"]
        self.message = "cuML would be installed from the RAPIDS index."
        self.title = "GPU acceleration"

    def as_text(self):
        return f"{self.title}: {self.action}"


def _entry(url="https://spacr.readthedocs.io/cuml"):
    return {"title": "cuML", "reason": "cuml is not installed",
            "url": url, "offer": _Offer()}


@pytest.fixture
def panel(qapp):
    """A panel of this test's own, never the process-wide singleton."""
    made = AvailabilityPanel()
    yield made
    made.dismiss()
    made.deleteLater()


@pytest.fixture
def anchor(qapp):
    widget = QWidget()
    widget.resize(180, 24)
    widget.show()
    yield widget
    widget.hide()
    widget.deleteLater()


def _press_at(point: QPoint) -> QMouseEvent:
    """A real press, carrying the global position it happened at."""
    local = QPointF(2.0, 2.0)
    return QMouseEvent(QEvent.MouseButtonPress, local, QPointF(point),
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


# -- a press that says where it happened -------------------------------------

def test_a_press_outside_the_panel_closes_it(panel, anchor, qtbot):
    """The press-away dismissal is what makes the panel escapable by mouse."""
    panel.show_for(anchor, [_entry()])
    assert panel.isVisible()
    closed = []
    panel.dismissed.connect(lambda: closed.append(True))

    outside = panel.geometry().topLeft() - QPoint(500, 500)
    handled = panel.eventFilter(anchor, _press_at(outside))

    assert handled is False, "the filter observes the press, it does not eat it"
    assert not panel.isVisible()
    assert closed == [True]


def test_a_press_on_the_panel_leaves_it_open(panel, anchor):
    """The Install word is inside the panel, so a press there must not close it."""
    panel.show_for(anchor, [_entry()])

    panel.eventFilter(anchor, _press_at(panel.geometry().center()))

    assert panel.isVisible()


def test_a_press_while_the_panel_is_hidden_is_ignored(panel, anchor,
                                                      monkeypatch):
    """Nothing on screen, nothing to dismiss.

    The filter is application-wide, so every press of an ordinary session
    reaches it while the panel is closed. Each one has to stop at the
    visibility check instead of running the dismissal -- which stops a
    pending hide, drops the pin and takes the focus back -- all over again.
    """
    panel.show_for(anchor, [_entry()])
    panel.dismiss()
    closed = []
    panel.dismissed.connect(lambda: closed.append(True))
    dismissals = []
    monkeypatch.setattr(panel, "dismiss", lambda: dismissals.append(True))

    handled = panel.eventFilter(anchor, _press_at(QPoint(-9000, -9000)))

    assert handled is False
    assert dismissals == []
    assert closed == []


def test_an_event_that_is_not_a_press_is_ignored(panel, anchor):
    """Every event in the application passes through here; only presses count."""
    panel.show_for(anchor, [_entry()])

    assert panel.eventFilter(anchor, QEvent(QEvent.KeyPress)) is False
    assert panel.isVisible()


# -- a press that does not say where it happened ------------------------------

def test_a_press_with_no_position_falls_back_to_the_pointer(panel, anchor,
                                                            monkeypatch):
    """A bare press event still has to be placed, or the filter raises.

    ``QEvent`` has no ``globalPosition``; only ``QMouseEvent`` does. An
    application-wide filter sees whatever any plugin or synthetic sender
    posts, and an ``AttributeError`` here escapes into the event loop where
    nothing catches it.
    """
    panel.show_for(anchor, [_entry()])
    monkeypatch.setattr(panel, "_cursor_pos",
                        lambda: panel.geometry().topLeft() - QPoint(400, 400))
    closed = []
    panel.dismissed.connect(lambda: closed.append(True))

    handled = panel.eventFilter(anchor, QEvent(QEvent.MouseButtonPress))

    assert handled is False
    assert not panel.isVisible()
    assert closed == [True]


def test_a_positionless_press_with_the_pointer_on_the_panel_keeps_it(
        panel, anchor, monkeypatch):
    """The fallback has to be able to say "inside" too, or it always closes."""
    panel.show_for(anchor, [_entry()])
    on_the_panel = panel.geometry().center()
    monkeypatch.setattr(panel, "_cursor_pos", lambda: on_the_panel)

    panel.eventFilter(anchor, QEvent(QEvent.MouseButtonPress))

    assert panel.isVisible()


# -- the corridor clock -------------------------------------------------------

def test_a_second_hide_request_does_not_restart_the_grace_period(panel,
                                                                 anchor):
    """The grace runs from when the pointer FIRST left, not from the latest event.

    A pointer wandering inside the corridor sends leave after leave; if each
    one reset the clock the panel could be kept open indefinitely by a
    stationary cursor.
    """
    panel.show_for(anchor, [_entry()])
    panel.start_hide()
    first = panel._hide_since

    time.sleep(0.01)
    panel.start_hide()

    assert panel._hide_since == first


def test_a_pinned_panel_never_arms_the_hide_timer(panel, anchor):
    """A reader who opened it by keyboard is not holding the mouse.

    The request is refused where it arrives, so no hide is left pending to
    fire once the reader looks away from the pointer.
    """
    panel.open_for(anchor, [_entry()])
    assert panel.is_pinned()

    panel.start_hide(1)

    assert not panel._hide_timer.isActive()
    assert panel.isVisible()


def test_a_pinned_panel_survives_the_hide_running_anyway(panel, anchor,
                                                         monkeypatch):
    """The pin is checked again where the hide is carried out.

    Everything else here says close: the grace period is long spent, and
    neither the anchor nor the panel is under the pointer. The pin is the
    only thing holding the panel open.
    """
    panel.open_for(anchor, [_entry()])
    assert panel.is_pinned()
    QApplication.sendEvent(anchor, QEvent(QEvent.Leave))
    assert not anchor.underMouse()
    panel._hide_since = time.monotonic() - 10.0
    monkeypatch.setattr(panel, "_cursor_pos",
                        lambda: panel.geometry().topLeft() - QPoint(900, 900))

    panel._maybe_hide()

    assert panel.isVisible()
    assert panel.is_pinned()


def test_a_panel_whose_anchor_is_gone_still_closes_on_the_timer(panel,
                                                               anchor,
                                                               monkeypatch):
    """With no anchor there is no corridor to travel, so the hide proceeds."""
    panel.show_for(anchor, [_entry()])
    panel._anchor = None
    panel._hide_since = time.monotonic() - 10.0
    monkeypatch.setattr(panel, "_cursor_pos",
                        lambda: panel.geometry().topLeft() - QPoint(900, 900))

    panel._maybe_hide()

    assert not panel.isVisible()


# -- placement with nothing to place it on ------------------------------------

def test_a_machine_with_no_screen_still_docks_the_panel_under_the_anchor(
        panel, anchor, monkeypatch):
    """No screen means no clamping, not no placement."""
    class _NoScreens:
        @staticmethod
        def screenAt(_point):        # noqa: N802 (Qt naming)
            return None

        @staticmethod
        def primaryScreen():         # noqa: N802 (Qt naming)
            return None

    monkeypatch.setattr(ap, "QGuiApplication", _NoScreens)

    panel.show_for(anchor, [_entry()])

    rect = panel._anchor_global_rect()
    assert panel.pos() == QPoint(rect.left(), rect.bottom() + 2)


# -- the API word -------------------------------------------------------------

def test_an_entry_with_no_documentation_opens_nothing(panel, anchor,
                                                      monkeypatch):
    """The API word is hidden without a URL, and pressing it does nothing."""
    opened = []

    class _Desktop:
        @staticmethod
        def openUrl(url):            # noqa: N802 (Qt naming)
            opened.append(url.toString())
            return True

    monkeypatch.setattr(ap, "QDesktopServices", _Desktop)
    asked = []
    panel.api_requested.connect(asked.append)

    panel.show_for(anchor, [_entry(url="")])
    assert not panel.api_link().isVisibleTo(panel)

    panel._on_link("api")

    assert opened == []
    assert asked == []


def test_an_entry_with_documentation_opens_it_once(panel, anchor,
                                                   monkeypatch):
    opened = []

    class _Desktop:
        @staticmethod
        def openUrl(url):            # noqa: N802 (Qt naming)
            opened.append(url.toString())
            return True

    monkeypatch.setattr(ap, "QDesktopServices", _Desktop)
    asked = []
    panel.api_requested.connect(asked.append)

    panel.show_for(anchor, [_entry()])
    panel._on_link("api")

    assert opened == ["https://spacr.readthedocs.io/cuml"]
    assert asked == ["https://spacr.readthedocs.io/cuml"]
