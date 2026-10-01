"""Actual timer/display regressions for native and rich tooltip surfaces."""
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QLabel, QWidget
from spacr.qt import preferences as prefs, tooltip_policy as policy
from spacr.qt.widgets.hover_tooltip import HoverTooltip


@pytest.fixture
def hovered(qtbot):
    host = QWidget()
    host.resize(300, 120)
    label = QLabel('Help', host)
    label.resize(100, 40)
    qtbot.addWidget(host)
    host.show()
    QCursor.setPos(host.mapToGlobal(host.rect().bottomRight()))
    qtbot.wait(20)
    prefs.set_tooltips_enabled(True)
    yield label
    prefs._set_tooltip_delay(2.0)


def test_rich_help_waits_four_seconds_and_explicit_click_is_immediate(hovered, qtbot):
    prefs._set_tooltip_delay(4.0)
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    popup.show_for(hovered, 'Delayed explanation', animation=None)
    qtbot.wait(3750)
    assert not popup.isVisible()
    qtbot.waitUntil(popup.isVisible, timeout=1200)
    assert popup.text_label().text() == 'Delayed explanation'
    popup.hide()
    popup.show_for(hovered, 'Clicked explanation', animation=None, immediate=True)
    assert popup.isVisible()


@pytest.mark.parametrize('cancel', ['leave', 'hide', 'destroy', 'preference'])
def test_pending_hover_never_survives_its_target_or_preference(hovered, qtbot, cancel):
    prefs._set_tooltip_delay(0.2)
    seen = []
    delay = policy.HoverDelay(hovered.parent())
    delay.schedule(hovered, lambda: seen.append('stale'))
    if cancel == 'leave':
        QApplication.sendEvent(hovered, QEvent(QEvent.Leave))
    elif cancel == 'hide':
        hovered.parent().hide()
    elif cancel == 'destroy':
        hovered.deleteLater()
    else:
        prefs._set_tooltip_delay(0.4)
    qtbot.wait(350)
    assert not seen
    assert not delay._timer.isActive()


def test_moving_to_another_control_and_reentering_restarts_full_wait(hovered, qtbot):
    prefs._set_tooltip_delay(0.2)
    other = QLabel('New dynamic target', hovered.parent())
    other.move(110, 0)
    other.show()
    delay = policy.HoverDelay(hovered.parent())
    seen = []
    delay.schedule(hovered, lambda: seen.append('old'))
    qtbot.wait(130)
    delay.schedule(other, lambda: seen.append('new'))
    qtbot.wait(120)
    assert not seen
    qtbot.waitUntil(lambda: seen == ['new'], timeout=200)
    delay.schedule(hovered, lambda: seen.append('reenter'))
    QApplication.sendEvent(hovered, QEvent(QEvent.Leave))
    delay.schedule(hovered, lambda: seen.append('reenter'))
    qtbot.wait(120)
    assert seen == ['new']
    qtbot.waitUntil(lambda: len(seen) == 2, timeout=200)


def test_zero_delay_and_disabled_help(hovered, qtbot):
    prefs._set_tooltip_delay(0)
    seen = []
    delay = policy.HoverDelay(hovered.parent())
    delay.schedule(hovered, lambda: seen.append(True))
    qtbot.waitUntil(lambda: bool(seen), timeout=200)
    prefs.set_tooltips_enabled(False)
    delay.schedule(hovered, lambda: seen.append(False))
    qtbot.wait(20)
    assert seen == [True]
    prefs.set_tooltips_enabled(True)


def test_native_four_second_delay_and_view_owned_help(hovered, qtbot):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip
    prefs._set_tooltip_delay(4)
    policy.uninstall_tooltip_policy()
    policy.install_tooltip_policy()
    hovered.setToolTip('Native four seconds')
    try:
        QApplication.sendEvent(hovered, QEvent(QEvent.Enter))
        qtbot.wait(3750)
        assert not QToolTip.isVisible()
        qtbot.waitUntil(lambda: QToolTip.isVisible() and
                       QToolTip.text() == 'Native four seconds', timeout=1200)
        policy.tooltip_policy().hide_now()
        prefs._set_tooltip_delay(.2)
        seen = []

        class ViewHelp(QWidget):
            def event(self, event):
                if event.type() == QEvent.ToolTip:
                    seen.append(True)
                    QToolTip.showText(event.globalPos(), 'Delegate help')
                    return True
                return super().event(event)

        view = ViewHelp(hovered.parent())
        view.show()
        event = QHelpEvent(QEvent.ToolTip, QPoint(1, 1),
                           view.mapToGlobal(QPoint(1, 1)))
        QApplication.sendEvent(view, event)
        qtbot.wait(120)
        assert not seen
        qtbot.waitUntil(lambda: bool(seen), timeout=300)
        assert QToolTip.text() == 'Delegate help'
    finally:
        QToolTip.hideText()
        policy.uninstall_tooltip_policy()


def test_availability_hover_waits_but_keyboard_help_does_not(hovered, qtbot):
    from spacr.qt.widgets.availability_panel import AvailabilityPanel
    prefs._set_tooltip_delay(.2)
    panel = AvailabilityPanel()
    qtbot.addWidget(panel)
    entries = [{'title': 'Optional backend', 'reason': 'Install the backend',
                'url': '', 'offer': None}]
    panel.show_for(hovered, entries)
    qtbot.wait(120)
    assert not panel.isVisible()
    # Repeated mouse moves over the same entry must not postpone it forever.
    panel.show_for(hovered, entries)
    qtbot.waitUntil(panel.isVisible, timeout=250)
    panel.hide()
    prefs._set_tooltip_delay(4)
    panel.open_for(hovered, entries)
    assert panel.isVisible() and panel.is_pinned()


def test_hint_bar_waits_before_changing_visible_text(hovered, qtbot):
    from spacr.qt.widgets.hint_bar import HintBar
    prefs._set_tooltip_delay(.2)
    bar = HintBar('Waiting', hovered.parent())
    bar.explain(hovered, 'A delayed explanation')
    bar.show()
    QApplication.sendEvent(hovered, QEvent(QEvent.Enter))
    qtbot.wait(120)
    assert bar.text() == 'Waiting'
    qtbot.waitUntil(lambda: bar.text() == 'A delayed explanation', timeout=200)


def test_test_data_descriptions_follow_global_delay(qtbot):
    from spacr.qt.widgets.test_data_chooser import TestDataChooser

    class Choices(TestDataChooser):
        ROUTES = (('example', 'Example', 'Example description'),)

    prefs._set_tooltip_delay(.2)
    chooser = Choices()
    qtbot.addWidget(chooser)
    chooser.show()
    resting = chooser.description_text()
    QApplication.sendEvent(chooser._buttons['example'], QEvent(QEvent.Enter))
    qtbot.wait(120)
    assert chooser.description_text() == resting
    qtbot.waitUntil(lambda: chooser.description_text() == 'Example description', timeout=200)
    prefs._set_tooltip_delay(2)


def test_a_late_leave_for_old_target_keeps_new_pending_help(hovered, qtbot):
    prefs._set_tooltip_delay(.2)
    second = QLabel('Second', hovered.parent())
    second.show()
    seen = []
    delay = policy.HoverDelay(hovered.parent())
    delay.schedule(hovered, lambda: seen.append('old'))
    delay.schedule(second, lambda: seen.append('new'))
    delay.cancel_for(hovered)
    qtbot.waitUntil(lambda: seen == ['new'], timeout=350)
