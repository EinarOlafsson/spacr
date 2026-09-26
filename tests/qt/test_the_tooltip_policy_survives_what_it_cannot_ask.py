"""Item 288: the tooltip filter's answers when Qt or a widget will not answer.

`spacr.qt.tooltip_policy` sits on ``QApplication`` and sees every tooltip
event in the program, so a widget whose ``toolTip()`` raises, a style that
cannot be asked for its delay, or a platform that cannot place the pointer
must never turn a hover into a traceback. Each case below pins the answer the
filter gives in that situation -- no text, Qt left to its own devices, or the
tooltip simply not shown -- rather than merely that nothing was raised.

The ordinary behaviour (the two-second wait, the one-second linger, the
switch in Preferences) is pinned in ``test_tooltips_wait_two_seconds.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint                     # noqa: E402
from PySide6.QtGui import QHelpEvent, QKeyEvent               # noqa: E402
from PySide6.QtWidgets import QPushButton, QToolTip, QWidget  # noqa: E402

from spacr.qt import tooltip_policy                           # noqa: E402
from spacr.qt.preferences import set_tooltips_enabled         # noqa: E402


@pytest.fixture
def policy(qapp):
    """A freshly installed filter, removed again afterwards."""
    tooltip_policy.uninstall_tooltip_policy(qapp)
    tooltip_policy.install_tooltip_policy(qapp)
    filter_ = tooltip_policy.tooltip_policy()
    yield filter_
    QToolTip.hideText()
    tooltip_policy.uninstall_tooltip_policy(qapp)
    set_tooltips_enabled(True)
    tooltip_policy.invalidate_tooltip_policy()


@pytest.fixture
def button(qapp):
    """A visible button carrying a tooltip, taken down after the test."""
    host = QWidget()
    host.resize(200, 80)
    press = QPushButton("Measure", host)
    press.setToolTip("Measure the objects you have segmented")
    host.show()
    yield press
    host.hide()
    host.deleteLater()


def _tooltip_event(widget):
    """The event Qt posts when the pointer has rested on ``widget``."""
    where = QPoint(320, 240)
    return QHelpEvent(QHelpEvent.Type.ToolTip, where,
                      widget.mapToGlobal(where))


class _Raises:
    """Stands in for anything whose every attribute call raises."""

    def __init__(self, *names):
        self._names = set(names)

    def __getattr__(self, name):
        if name in self._names:
            def boom(*_a, **_k):
                raise RuntimeError(f"{name} refused")
            return boom
        raise AttributeError(name)


def test_an_unreadable_preference_leaves_tooltips_on(monkeypatch):
    """A settings store that cannot be read must not silence every tooltip."""
    import spacr.qt.preferences as preferences

    def unreadable():
        raise OSError("settings file locked")

    monkeypatch.setattr(preferences, "get_tooltips_enabled", unreadable)
    monkeypatch.setattr(tooltip_policy, "_enabled", None)
    assert tooltip_policy.tooltips_enabled() is True
    monkeypatch.setattr(tooltip_policy, "_enabled", None)


def test_a_widget_whose_tooltip_raises_has_no_text():
    """The parent walk stops at a widget that cannot say its tooltip."""
    assert tooltip_policy.tooltip_text_for(_Raises("toolTip")) == ""


def test_a_widget_that_cannot_say_where_its_window_is_has_no_text():
    """No tooltip, and ``isWindow`` raising: the walk gives up with ``""``."""

    class Orphan(_Raises):
        def toolTip(self):
            return ""

    assert tooltip_policy.tooltip_text_for(Orphan("isWindow")) == ""


def test_an_unrelated_event_is_passed_on_and_changes_nothing(
        policy, button):
    """A move or a paint is neither a hover, a leave nor an interruption."""
    policy.eventFilter(button, _tooltip_event(button))
    assert policy.eventFilter(button, QEvent(QEvent.Type.Move)) is False
    assert policy._widget is button
    assert policy._show_timer.isActive()


def test_the_parent_walk_gives_up_after_sixty_four_levels():
    """A parent chain that never reaches a window cannot loop for ever."""
    visited = []

    class Endless:
        def toolTip(self):
            visited.append(self)
            return ""

        def isWindow(self):
            return False

        def parentWidget(self):
            return Endless()

    assert tooltip_policy.tooltip_text_for(Endless()) == ""
    assert len(visited) == 64


def test_an_event_without_a_type_is_left_to_qt(policy):
    """An event object that cannot report its type is not consumed."""
    assert policy.eventFilter(object(), _Raises("type")) is False


@pytest.mark.parametrize("make_event", [
    lambda: QEvent(QEvent.Type.MouseButtonPress),
    lambda: QEvent(QEvent.Type.Wheel),
    lambda: QKeyEvent(QEvent.Type.KeyPress, 65,
                      __import__("PySide6.QtCore").QtCore.Qt
                      .KeyboardModifier.NoModifier),
])
def test_a_click_a_wheel_or_a_key_takes_the_tooltip_away(
        policy, button, qapp, make_event):
    """Doing something else with the program is not reading a tooltip."""
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    assert policy._showing is True
    assert policy.eventFilter(button, make_event()) is False
    assert policy._showing is False
    assert policy._widget is None
    assert not policy._show_timer.isActive()


def test_leaving_a_different_widget_does_not_start_the_linger(
        policy, button, qapp):
    """Only the widget whose tooltip is up can start its linger."""
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    other = QWidget()
    try:
        policy.eventFilter(other, QEvent(QEvent.Type.Leave))
        assert not policy._hide_timer.isActive()
        assert policy._widget is button
    finally:
        other.deleteLater()


def test_a_tooltip_event_for_a_non_widget_is_left_to_qt(policy):
    """Something with no ``toolTip`` cannot be hovered for one."""
    handled = policy.eventFilter(QEvent.Type.None_, QHelpEvent(
        QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1)))
    assert handled is False
    assert policy._widget is None


def test_a_widget_whose_property_raises_is_still_given_the_policy(policy):
    """The opt-out check failing is not the same as opting out."""

    class Awkward:
        def toolTip(self):
            return "awkward but explained"

        def property(self, _name):
            raise RuntimeError("property refused")

        def isWindow(self):
            return True

    widget = Awkward()
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(3, 4), QPoint(30, 40))
    assert policy.eventFilter(widget, event) is True
    assert policy._widget is widget
    assert policy._text == "awkward but explained"
    assert policy._pos == QPoint(30, 40)
    assert policy._show_timer.isActive()


def test_an_event_without_a_position_falls_back_to_the_cursor(
        policy, button, monkeypatch):
    """A tooltip event that cannot say where it is uses the pointer's spot."""
    monkeypatch.setattr(tooltip_policy, "QCursor",
                        SimpleNamespace(pos=lambda: QPoint(11, 22)))
    event = _Raises("globalPos")
    event.type = lambda: QEvent.Type.ToolTip
    assert policy._on_tooltip(button, event) is True
    assert policy._pos == QPoint(11, 22)


def test_a_second_hover_before_the_tip_shows_keeps_the_same_wait(
        policy, button):
    """A pointer resting on one widget does not keep restarting its wait."""
    policy.eventFilter(button, _tooltip_event(button))
    first_id = policy._show_timer.timerId()
    assert policy._show_timer.isActive()
    policy.eventFilter(button, _tooltip_event(button))
    assert policy._widget is button
    assert policy._show_timer.isActive()
    assert policy._show_timer.timerId() == first_id


def test_a_style_that_cannot_be_asked_leaves_the_full_two_seconds(
        policy, monkeypatch):
    """With no style delay to subtract, the wait is the whole wait."""
    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(style=lambda: None))
    assert policy.remaining_delay_ms() == tooltip_policy.SHOW_DELAY_MS

    class Broken:
        def styleHint(self, _hint):
            raise RuntimeError("no style hint")

    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(style=Broken))
    assert policy.remaining_delay_ms() == tooltip_policy.SHOW_DELAY_MS


def test_nothing_hovered_means_nothing_shown(policy, qapp):
    """The wait ending with no widget remembered shows nothing."""
    QToolTip.hideText()
    policy._widget = None
    policy._text = "stale"
    policy._show_now()
    assert policy._showing is False


def test_a_hidden_widget_does_not_raise_its_tooltip(policy, qapp):
    """The pointer cannot be resting on a widget that is not on screen."""
    hidden = QPushButton("Hidden")
    hidden.setToolTip("never seen")
    try:
        policy._widget = hidden
        policy._text = "never seen"
        policy._show_now()
        assert policy._showing is False
    finally:
        hidden.deleteLater()


def test_a_widget_that_cannot_say_whether_it_is_visible_shows_nothing(
        policy):
    """A widget already torn down under the pointer raises no tooltip."""
    policy._widget = _Raises("isVisible")
    policy._text = "from a deleted widget"
    policy._show_now()
    assert policy._showing is False


def test_a_tooltip_qt_refuses_to_show_is_not_recorded_as_showing(
        policy, button, monkeypatch):
    """If the text never reached the screen, there is nothing to linger."""

    def refuse(*_a, **_k):
        raise RuntimeError("no screen")

    policy.eventFilter(button, _tooltip_event(button))
    monkeypatch.setattr(tooltip_policy, "QToolTip",
                        SimpleNamespace(showText=refuse))
    policy._show_now()
    assert policy._showing is False


def test_a_replay_that_cannot_be_built_is_dropped(policy):
    """No position to map means no event to send back, and no crash."""
    policy._pos = QPoint(5, 5)
    widget = _Raises("mapFromGlobal")
    policy._replay(widget)
    assert policy._replaying is False
    assert policy._widget is None
    assert policy._text == ""


def test_a_replay_qt_refuses_to_deliver_leaves_the_filter_listening(
        policy, button, monkeypatch):
    """The stand-aside flag is cleared even when delivery fails."""

    def refuse(*_a, **_k):
        raise RuntimeError("receiver gone")

    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(sendEvent=refuse))
    policy._pos = QPoint(5, 5)
    policy._replay(button)
    assert policy._replaying is False
    assert policy.eventFilter(button, _tooltip_event(button)) is True


def test_the_pointer_question_says_no_when_it_cannot_be_answered(
        policy, monkeypatch):
    """Nothing under the pointer, or a window that will not say its flags."""

    def refuse(*_a, **_k):
        raise RuntimeError("no screen")

    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(widgetAt=refuse))
    assert policy._pointer_is_on_the_tooltip() is False

    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(widgetAt=lambda _pos: None))
    assert policy._pointer_is_on_the_tooltip() is False

    class Under:
        def window(self):
            return _Raises("windowFlags")

    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(widgetAt=lambda _pos: Under()))
    assert policy._pointer_is_on_the_tooltip() is False


def test_a_window_flagged_as_a_tooltip_is_recognised(policy, monkeypatch):
    """The masked comparison answers yes for a real tooltip window."""
    from PySide6.QtCore import Qt

    class TipWindow:
        def windowFlags(self):
            return Qt.WindowType.ToolTip

    class Under:
        def window(self):
            return TipWindow()

    monkeypatch.setattr(tooltip_policy, "QApplication",
                        SimpleNamespace(widgetAt=lambda _pos: Under()))
    assert policy._pointer_is_on_the_tooltip() is True


def test_a_tooltip_qt_cannot_hide_is_still_forgotten(policy, monkeypatch):
    """The filter stops believing a tip is up even if hiding it failed."""

    def refuse(*_a, **_k):
        raise RuntimeError("no screen")

    monkeypatch.setattr(tooltip_policy, "QToolTip",
                        SimpleNamespace(hideText=refuse))
    policy._showing = True
    policy._hide_text()
    assert policy._showing is False


def test_install_and_uninstall_without_an_application(monkeypatch):
    """No QApplication yet: nothing is installed, and removal still works."""

    class NoApp:
        @staticmethod
        def instance():
            return None

    saved = tooltip_policy._filter
    monkeypatch.setattr(tooltip_policy, "_filter", None)
    monkeypatch.setattr(tooltip_policy, "QApplication", NoApp)
    assert tooltip_policy.install_tooltip_policy() is False
    assert tooltip_policy.tooltip_policy() is None

    monkeypatch.setattr(tooltip_policy, "_filter",
                        tooltip_policy._TooltipFilter())
    assert tooltip_policy.uninstall_tooltip_policy() is True
    assert tooltip_policy.tooltip_policy() is None
    monkeypatch.setattr(tooltip_policy, "_filter", saved)
