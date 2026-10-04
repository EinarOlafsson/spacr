"""The tooltip policy's fallbacks and the edges of its hover bookkeeping."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPointF  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from spacr.qt import preferences, tooltip_policy as tp  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_policy(monkeypatch):
    monkeypatch.setattr(tp, "_delay_ms", None)
    monkeypatch.setattr(tp, "_enabled", True)
    yield


def test_an_unreadable_delay_preference_falls_back_to_two_seconds(monkeypatch):
    def broken():
        raise RuntimeError("no settings")

    monkeypatch.setattr(preferences, "_get_tooltip_delay", broken)
    assert tp._preferred_delay_ms() == tp.SHOW_DELAY_MS


def test_the_style_wake_up_is_never_negative_and_survives_no_style(monkeypatch):
    assert tp._style_wake_up_ms() >= 0
    monkeypatch.setattr(QApplication, "style", staticmethod(lambda: None))
    assert tp._style_wake_up_ms() == 0

    def broken():
        raise RuntimeError("gone")

    monkeypatch.setattr(QApplication, "style", staticmethod(broken))
    assert tp._style_wake_up_ms() == 0


def test_invalidating_forgets_a_hover_whose_object_is_gone(qapp, monkeypatch):
    import shiboken6

    delay = tp.HoverDelay()
    monkeypatch.setattr(shiboken6, "isValid", lambda obj: obj is not delay)
    tp.invalidate_tooltip_policy()
    assert delay not in tp._hover_delays


def test_cancel_after_the_timer_is_gone_does_nothing(qapp):
    delay = tp.HoverDelay()
    delay._timer = None
    delay.cancel()
    assert delay._timer is None


class _Fragile(QLabel):
    """A label whose filters and signal refuse to be released."""

    def removeEventFilter(self, obj):  # noqa: N802
        raise RuntimeError("already deleted")


def test_cancel_tolerates_targets_that_are_already_gone(qtbot):
    label = _Fragile("x")
    qtbot.addWidget(label)
    label.show()
    delay = tp.HoverDelay()
    delay.schedule(label, lambda: None)
    delay.cancel()
    assert delay._anchor is None


def test_a_leave_from_a_neighbour_keeps_the_pending_help(qtbot):
    a, b = QLabel("a"), QLabel("b")
    for widget in (a, b):
        qtbot.addWidget(widget)
        widget.show()
    delay = tp.HoverDelay()
    delay.schedule(a, lambda: None)
    delay.cancel_for(b)
    assert delay._anchor is a
    delay.cancel_for(a)
    assert delay._anchor is None


def test_help_is_not_delivered_when_tooltips_are_off_or_the_anchor_died(
        qtbot, monkeypatch):
    shown = []
    label = QLabel("x")
    qtbot.addWidget(label)
    label.show()
    delay = tp.HoverDelay()
    delay._anchor, delay._callback = label, lambda: shown.append(1)
    monkeypatch.setattr(tp, "_enabled", False)
    delay._deliver()
    monkeypatch.setattr(tp, "_enabled", True)

    class _Dead:
        def isVisible(self):
            raise RuntimeError("deleted")

    delay._anchor, delay._callback = _Dead(), lambda: shown.append(2)
    delay.cancel = lambda: None
    delay._deliver()
    assert shown == []


def test_a_fixed_delay_can_be_released_to_the_preference(monkeypatch):
    monkeypatch.setattr(tp, "_delay_ms", 123)
    policy = tp._TooltipFilter(show_delay_ms=50)
    assert policy.show_delay_ms == 50
    policy.show_delay_ms = None
    assert policy.show_delay_ms == 123


def test_an_event_without_a_type_is_passed_on(qapp):
    class _Odd:
        def type(self):
            raise RuntimeError("no type")

    assert tp._TooltipFilter(0).eventFilter(QObject(), _Odd()) is False


def _move(widget, x):
    return QMouseEvent(QEvent.MouseMove, QPointF(x, 1), widget.mapToGlobal(
        QPointF(x, 1)), Qt.NoButton, Qt.NoButton, Qt.NoModifier)


def test_moving_off_a_pending_cell_cancels_its_tip(qtbot):
    label = QLabel("x")
    label.setToolTip("help")
    qtbot.addWidget(label)
    label.show()
    policy = tp._TooltipFilter(show_delay_ms=10_000)
    policy._widget = label
    policy._show_timer.start(10_000)
    policy._pos = label.mapToGlobal(QPointF(1, 1)).toPoint()
    assert policy.eventFilter(label, _move(label, 1)) is False
    assert policy._show_timer.isActive()
    policy.eventFilter(label, _move(label, 30))
    assert not policy._show_timer.isActive()


def test_leaving_another_widget_or_hiding_a_dead_one(qtbot):
    label, other = QLabel("x"), QLabel("y")
    for widget in (label, other):
        qtbot.addWidget(widget)
    policy = tp._TooltipFilter(show_delay_ms=0)
    policy._widget = label
    policy._text = "help"
    policy.eventFilter(other, QEvent(QEvent.Leave))
    assert policy._text == "help"

    class _Gone:
        def window(self):
            raise RuntimeError("deleted")

    policy._widget = _Gone()
    policy.eventFilter(other, QEvent(QEvent.Hide))
    assert policy._text == ""


def test_switching_targets_survives_a_signal_that_will_not_disconnect(qtbot):
    policy = tp._TooltipFilter(show_delay_ms=0)

    class _Signal:
        def connect(self, slot):
            pass

        def disconnect(self, slot):
            raise TypeError("not connected")

    class _Target:
        destroyed = _Signal()

    first = _Target()
    policy._track_widget(first)
    policy._track_widget(None)
    assert policy._watched_widget is None and policy._widget is None


def test_entering_does_nothing_when_tooltips_are_off_or_opted_out(
        qtbot, monkeypatch):
    label = QLabel("x")
    label.setToolTip("help")
    qtbot.addWidget(label)
    policy = tp._TooltipFilter(show_delay_ms=10_000)
    monkeypatch.setattr(tp, "_enabled", False)
    policy._on_enter(label)
    assert not policy._show_timer.isActive()
    monkeypatch.setattr(tp, "_enabled", True)
    label.setProperty(tp.OPT_OUT_PROPERTY, True)
    policy._on_enter(label)
    assert not policy._show_timer.isActive()

    class _Strange(QObject):
        def toolTip(self):  # noqa: N802
            return "help"

        def property(self, name):
            raise RuntimeError("no properties")

    policy._on_enter(_Strange())
    assert not policy._show_timer.isActive()


def test_entering_the_same_pending_target_again_keeps_its_wait(qtbot):
    label = QLabel("x")
    label.setToolTip("help")
    qtbot.addWidget(label)
    policy = tp._TooltipFilter(show_delay_ms=10_000)
    try:
        policy._on_enter(label)
        remaining = policy._show_timer.remainingTime()
        policy._on_enter(label)
        assert policy._show_timer.isActive()
        assert policy._show_timer.remainingTime() <= remaining
    finally:
        policy._show_timer.stop()
        policy._track_widget(None)


def test_entering_a_new_target_while_a_tip_shows_hides_it_first(qtbot):
    a, b = QLabel("a"), QLabel("b")
    a.setToolTip("first")
    b.setToolTip("second")
    for widget in (a, b):
        qtbot.addWidget(widget)
    policy = tp._TooltipFilter(show_delay_ms=10_000)
    hidden = []
    policy._hide_text = lambda: hidden.append(True)
    policy._widget, policy._text, policy._showing = a, "first", True
    try:
        policy._on_enter(b)
        assert hidden == [True] and policy._text == "second"
    finally:
        policy._show_timer.stop()
        policy._track_widget(None)


def test_leaving_before_the_tip_shows_forgets_the_target(qtbot):
    label = QLabel("x")
    qtbot.addWidget(label)
    policy = tp._TooltipFilter(show_delay_ms=0)
    policy._widget, policy._text, policy._showing = label, "help", False
    policy._start_the_linger()
    assert policy._widget is None and policy._text == ""
