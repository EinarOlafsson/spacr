"""How the application-wide tooltip filter routes each kind of event.

One case table over :meth:`spacr.qt.tooltip_policy._TooltipFilter.eventFilter`:
which events cancel a pending or shown tip, which ones belong to someone
else and leave it alone, and that nothing but a ``ToolTip`` is consumed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from spacr.qt import tooltip_policy as tp  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_policy(monkeypatch):
    """Tooltips on and the delay unread, whatever another test left."""
    monkeypatch.setattr(tp, "_delay_ms", None)
    monkeypatch.setattr(tp, "_enabled", True)
    yield


def _move(widget, x):
    """A mouse move over ``widget`` at local ``(x, 1)``."""
    return QMouseEvent(QEvent.MouseMove, QPointF(x, 1),
                       widget.mapToGlobal(QPointF(x, 1)), Qt.NoButton,
                       Qt.NoButton, Qt.NoModifier)


class _Gone:
    """A tracked target whose C++ half was deleted."""

    def window(self):
        raise RuntimeError("deleted")


class _Typeless:
    """An event that cannot say what it is."""

    def type(self):
        raise RuntimeError("no type")


def test_the_filter_cancels_only_the_tip_an_event_concerns(qtbot, subtests):
    """Moving off a pending cell, leaving the target, pressing, scrolling,
    typing or hiding a dead target clear the tip; events about other widgets
    keep it; only ToolTip events are ever consumed."""
    label, other = QLabel("x"), QLabel("y")
    label.setToolTip("help")
    for widget in (label, other):
        qtbot.addWidget(widget)
        widget.show()

    def _pending(policy):
        """A tip waiting to show for ``label``, with no text yet."""
        policy._track_widget(label)
        policy._show_timer.start(10_000)
        policy._pos = label.mapToGlobal(QPointF(1, 1)).toPoint()

    def _armed(policy):
        """A tip for ``label`` with its text known, not yet showing."""
        policy._track_widget(label)
        policy._text = "help"

    def _showing(policy):
        _armed(policy)
        policy._showing = True

    def _dead(policy):
        policy._widget = _Gone()
        policy._text = "help"

    # (case, setup, target, event builder, kept text, timer active after,
    #  tracked widget after)
    cases = [
        ("an event without a type", _armed, QObject(), lambda: _Typeless(),
         "help", False, label),
        ("a move on the spot keeps the wait", _pending, label,
         lambda: _move(label, 1), "", True, label),
        ("a move off the spot cancels it", _pending, label,
         lambda: _move(label, 30), "", False, None),
        ("a move with text known is no cancel", _armed, label,
         lambda: _move(label, 30), "help", False, label),
        ("leaving a neighbour", _armed, other,
         lambda: QEvent(QEvent.Leave), "help", False, label),
        ("leaving the target before it shows", _armed, label,
         lambda: QEvent(QEvent.Leave), "", False, None),
        ("hiding a neighbour in another window", _armed, other,
         lambda: QEvent(QEvent.Hide), "help", False, label),
        ("hiding the target", _armed, label,
         lambda: QEvent(QEvent.Hide), "", False, None),
        ("closing the target's window", _armed, label.window(),
         lambda: QEvent(QEvent.Close), "", False, None),
        ("deactivating another window", _armed, other,
         lambda: QEvent(QEvent.WindowDeactivate), "help", False, label),
        ("hiding anything while the target is dead", _dead, other,
         lambda: QEvent(QEvent.Hide), "", False, None),
        ("a mouse press anywhere", _armed, other,
         lambda: QEvent(QEvent.MouseButtonPress), "", False, None),
        ("a wheel anywhere", _armed, other,
         lambda: QEvent(QEvent.Wheel), "", False, None),
        ("a key press anywhere", _armed, other,
         lambda: QEvent(QEvent.KeyPress), "", False, None),
    ]
    for case, setup, target, build, text, active, tracked in cases:
        with subtests.test(msg=case):
            policy = tp._TooltipFilter(show_delay_ms=10_000)
            try:
                setup(policy)
                assert policy.eventFilter(target, build()) is False
                assert policy._text == text
                assert policy._show_timer.isActive() is active
                assert policy._widget is tracked
            finally:
                policy._show_timer.stop()
                policy._hide_timer.stop()
                policy._track_widget(None)

    with subtests.test(msg="leaving the target while it shows lingers"):
        policy = tp._TooltipFilter(show_delay_ms=0, linger_ms=5_000)
        try:
            _showing(policy)
            assert policy.eventFilter(label, QEvent(QEvent.Leave)) is False
            assert policy._hide_timer.isActive() is True
            assert policy._hide_timer.interval() == 5_000
            assert (policy._text, policy._widget) == ("help", label)
        finally:
            policy.hide_now()
        assert (policy._text, policy._widget, policy._showing) == ("", None, False)

    with subtests.test(msg="entering a target with help starts the wait"):
        policy = tp._TooltipFilter(show_delay_ms=10_000)
        try:
            assert policy.eventFilter(label, QEvent(QEvent.Enter)) is False
            assert policy._widget is label
            assert policy._text == tp.tooltip_text_for(label)
            assert policy._show_timer.isActive() is True
            assert policy._show_timer.interval() == 10_000
        finally:
            policy.hide_now()
