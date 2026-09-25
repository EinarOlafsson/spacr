"""Item 506: one rule for every tooltip, and a switch that silences them.

The maintainer's complaint was that tooltips "pop up too fast": Qt's style
raises one after about 700 ms, while the pointer is still on its way
somewhere else, and takes it away the instant the pointer moves off, which
is too soon to finish reading a long one.

`spacr.qt.tooltip_policy` answers that with ONE event filter on
``QApplication`` rather than a change per widget, so what is pinned here is
the filter's decisions, not any individual tooltip:

* the tooltip event is intercepted, and nothing is shown at the moment of
  the hover;
* the wait is two seconds, and the linger after leaving is one;
* the switch in Preferences silences every tooltip, and defaults to on.

The timers are fired by hand. Waiting two real seconds per case would add
ten seconds to the suite to measure a number this file can read directly.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                            # noqa: E402
from PySide6.QtGui import QHelpEvent                         # noqa: E402
from PySide6.QtWidgets import QPushButton, QToolTip, QWidget  # noqa: E402

from spacr.qt import tooltip_policy                           # noqa: E402
from spacr.qt.preferences import (get_tooltips_enabled,       # noqa: E402
                                  set_tooltips_enabled)


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


@pytest.fixture
def button(qapp):
    """A visible button carrying a tooltip, taken down after the test."""
    widget = QWidget()
    widget.resize(200, 80)
    press = QPushButton("Measure", widget)
    press.setToolTip("Measure the objects you have segmented")
    widget.show()
    yield press
    widget.hide()
    widget.deleteLater()


def gone(qapp, timeout_ms=2000):
    """Wait for the tooltip window to actually close.

    ``QToolTip.hideText`` does not close the window there and then: Qt gives
    the label a 300 ms grace so a pointer crossing a gap between two widgets
    does not flicker. So "gone" is a thing to wait for, not to read.
    """
    from PySide6.QtCore import QElapsedTimer

    clock = QElapsedTimer()
    clock.start()
    while clock.elapsed() < timeout_ms:
        if not QToolTip.isVisible():
            return True
        qapp.processEvents()
    return not QToolTip.isVisible()


def test_the_default_is_that_tooltips_are_shown():
    """"tooltips should be on by default" -- the request, verbatim."""
    assert get_tooltips_enabled() is True


def test_the_wait_is_two_seconds_and_the_linger_is_one():
    """The two numbers the request names, in one place, in milliseconds."""
    assert tooltip_policy.SHOW_DELAY_MS == 2000
    assert tooltip_policy.LINGER_MS == 1000


def test_nothing_is_shown_at_the_moment_of_the_hover(policy, button, qapp):
    """The whole point: the tooltip must not be there yet."""
    QToolTip.hideText()
    assert gone(qapp)
    handled = policy.eventFilter(button, _tooltip_event(button))
    assert handled is True, "the filter must take the event from Qt's style"
    assert not QToolTip.isVisible()
    assert policy._show_timer.isActive()
    assert policy._show_timer.interval() == policy.remaining_delay_ms()


def test_the_text_appears_once_the_wait_is_over(policy, button, qapp):
    """Firing the wait by hand is the same call the timer makes."""
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    assert QToolTip.text() == "Measure the objects you have segmented"


def test_hovering_the_same_widget_again_does_not_restart_the_wait(
        policy, button, qapp):
    """A pointer that trembles on one widget is still resting on it."""
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    shown_for = policy._widget
    policy.eventFilter(button, _tooltip_event(button))
    assert policy._widget is shown_for
    assert not policy._show_timer.isActive()


def test_leaving_the_widget_gives_the_reader_one_more_second(
        policy, button, qapp):
    """Qt hides on Leave at once; this is the change the request asked for."""
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    policy._start_the_linger()
    assert policy._hide_timer.isActive()
    assert policy._hide_timer.interval() == tooltip_policy.LINGER_MS
    assert QToolTip.isVisible() or QToolTip.text()


def test_the_linger_ends_with_the_tooltip_gone(policy, button, qapp):
    """And when the second is up, and the pointer is elsewhere, it goes."""
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    policy._start_the_linger()
    policy._hide_if_the_pointer_left()
    assert gone(qapp), "the tooltip is still on screen a second later"
    assert policy._widget is None


def test_a_tooltip_the_pointer_rests_on_is_not_taken_away(
        policy, button, qapp, monkeypatch):
    """"if the mouse stays on them they do not dissapear" -- the request.

    The pointer's whereabouts cannot be set under the offscreen platform,
    so the answer is given directly; what is pinned is that the filter
    ASKS, and waits another second rather than hiding.
    """
    policy.eventFilter(button, _tooltip_event(button))
    policy._show_now()
    policy._start_the_linger()
    monkeypatch.setattr(policy, "_pointer_is_on_the_tooltip", lambda: True)
    policy._hide_if_the_pointer_left()
    assert QToolTip.isVisible()
    assert policy._hide_timer.isActive()


def test_an_ordinary_window_under_the_pointer_is_not_a_tooltip(
        policy, button, qapp):
    """The guard for a mask that was wrong and hid nothing, ever.

    ``Qt::ToolTip`` is ``Popup | Sheet``, both of which carry the
    ``Window`` bit, so testing it without the window-type mask is true of
    every window on screen -- and a tooltip would then never be taken away
    while any spaCR window sat under the pointer.
    """
    assert policy._pointer_is_on_the_tooltip() is False


def test_moving_to_another_widget_hands_the_tooltip_over(policy, qapp):
    """The second widget's wait starts; the first one's text is taken away."""
    host = QWidget()
    first = QPushButton("One", host)
    first.setToolTip("the first")
    second = QPushButton("Two", host)
    second.move(0, 40)
    second.setToolTip("the second")
    host.resize(200, 80)
    host.show()
    try:
        policy.eventFilter(first, _tooltip_event(first))
        policy._show_now()
        assert QToolTip.text() == "the first"
        policy.eventFilter(second, _tooltip_event(second))
        assert gone(qapp), "the first widget's text outlived the pointer"
        assert policy._widget is second
        assert policy._show_timer.isActive()
    finally:
        host.hide()
        host.deleteLater()


def test_the_preference_silences_every_tooltip(policy, button, qapp):
    """Cleared, the event is swallowed and no wait is even started."""
    set_tooltips_enabled(False)
    try:
        handled = policy.eventFilter(button, _tooltip_event(button))
        assert handled is True
        assert not policy._show_timer.isActive()
        policy._show_now()
        assert gone(qapp)
    finally:
        set_tooltips_enabled(True)
    assert tooltip_policy.tooltips_enabled() is True


def test_a_widget_can_opt_out_and_keep_qt_s_own_tooltip(policy, button, qapp):
    """The escape hatch is a property, not an edit to the policy module."""
    button.setProperty(tooltip_policy.OPT_OUT_PROPERTY, True)
    handled = policy.eventFilter(button, _tooltip_event(button))
    assert handled is False, "Qt must be left to show this one itself"


def test_the_wait_counts_the_wait_qt_has_already_served(policy, qapp):
    """Two seconds means two, not two on top of the style's own 700 ms."""
    from PySide6.QtWidgets import QStyle

    style = qapp.style()
    already = int(style.styleHint(QStyle.StyleHint.SH_ToolTip_WakeUpDelay))
    assert policy.remaining_delay_ms() + already == \
        tooltip_policy.SHOW_DELAY_MS or already > tooltip_policy.SHOW_DELAY_MS


def test_a_child_without_a_tooltip_still_shows_its_parents(policy, qapp):
    """Qt propagates a tooltip to the parent, and so must this filter.

    A card that explains itself must not fall silent the moment the pointer
    lands on the label written on it.
    """
    card = QWidget()
    card.setToolTip("what this whole card is for")
    label = QPushButton("Run", card)
    card.resize(200, 80)
    card.show()
    try:
        assert tooltip_policy.tooltip_text_for(label) == \
            "what this whole card is for"
        policy.eventFilter(label, _tooltip_event(label))
        policy._show_now()
        assert QToolTip.text() == "what this whole card is for"
    finally:
        card.hide()
        card.deleteLater()


def test_a_widget_that_answers_for_itself_gets_the_event_back(policy, qapp):
    """A table's cell tooltips live in its own ``event``, not in toolTip().

    Swallowing those would take them away; showing them at the hover would
    leave them the only fast tooltips in the program. So the event is sent
    again once the wait is over, and the widget answers then.
    """
    seen = []

    class Answering(QWidget):
        def event(self, event):
            if event.type() == event.Type.ToolTip:
                seen.append(event.pos())
                return True
            return super().event(event)

    host = Answering()
    host.resize(120, 60)
    host.show()
    try:
        assert policy.eventFilter(host, _tooltip_event(host)) is True
        assert not seen, "nothing may reach the widget before the wait"
        policy._show_now()
        assert seen, "the widget never got its chance to answer"
        assert policy._replaying is False
    finally:
        host.hide()
        host.deleteLater()


def test_installing_twice_leaves_one_filter(qapp):
    """`apply_preferences_to_app` runs on every preference change."""
    tooltip_policy.uninstall_tooltip_policy(qapp)
    try:
        assert tooltip_policy.install_tooltip_policy(qapp) is True
        first = tooltip_policy.tooltip_policy()
        assert tooltip_policy.install_tooltip_policy(qapp) is False
        assert tooltip_policy.tooltip_policy() is first
    finally:
        tooltip_policy.uninstall_tooltip_policy(qapp)


def test_preferences_carries_the_switch_and_writes_it(qapp, qtbot):
    """"there should be a setting in preferences that toggles all tooltips
    off" -- and turning it off must reach the filter, not only the store."""
    from PySide6.QtWidgets import QDialogButtonBox

    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.widgets.toggle import Toggle

    set_tooltips_enabled(True)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    try:
        switch = dialog.findChild(Toggle, "TooltipsEnabled")
        assert switch is not None, "no master tooltip switch in Preferences"
        assert switch.isChecked() is True, "it must arrive on"
        switch.setChecked(False)
        dialog.findChild(QDialogButtonBox).accepted.emit()
        assert get_tooltips_enabled() is False
        assert tooltip_policy.tooltips_enabled() is False
    finally:
        set_tooltips_enabled(True)
        dialog.deleteLater()


def _tooltip_event(widget):
    """The event Qt posts when the pointer has rested on ``widget``.

    The point is deliberately far from the screen's origin. Offscreen, the
    mouse cursor reads as ``(0, 0)``; a tooltip raised at ``(2, 2)`` would
    be under it, and the filter would rightly refuse to hide a tooltip the
    reader appears to be resting on.
    """
    where = QPoint(320, 240)
    return QHelpEvent(QHelpEvent.Type.ToolTip, where,
                      widget.mapToGlobal(where))
