"""Preferences sets how long a hover waits before its tooltip appears.

Asked for on 2026-09-30: "in preferences withe the other tooltip settings,
there should be a time lag setting for the tooltip, how long it takes for
tooltips to appear upon hover. default to 2 seconds."

The wait is the application-wide tooltip policy's, so one preference moves
every tooltip in spaCR. These tests hover a real button and wait real time:
nothing is on screen before the delay, and the text is there after it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QElapsedTimer, QEvent, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QSlider,
                               QToolTip, QWidget)

from spacr.qt import preferences as prefs
from spacr.qt import tooltip_policy

TIP = "Measure the objects you have segmented"


@pytest.fixture
def policy(qapp):
    """A freshly installed filter that follows the preference."""
    tooltip_policy.uninstall_tooltip_policy(qapp)
    tooltip_policy.install_tooltip_policy(qapp)
    tooltip_policy.invalidate_tooltip_policy()
    yield tooltip_policy.tooltip_policy()
    QToolTip.hideText()
    tooltip_policy.uninstall_tooltip_policy(qapp)
    prefs._set_tooltip_delay(prefs._TOOLTIP_DELAY_DEFAULT)


@pytest.fixture
def button(qapp):
    """A visible button carrying a tooltip."""
    host = QWidget()
    host.resize(200, 80)
    press = QPushButton("Measure", host)
    press.setToolTip(TIP)
    host.show()
    yield press
    host.hide()
    host.deleteLater()


def _hover(widget):
    """Deliver the event Qt sends when the pointer rests on ``widget``."""
    centre = widget.rect().center()
    event = QHelpEvent(QEvent.Type.ToolTip, centre, widget.mapToGlobal(centre))
    return QApplication.sendEvent(widget, event)


def _wait(qapp, ms):
    """Let ``ms`` of real time go by with the event loop running."""
    clock = QElapsedTimer()
    clock.start()
    while clock.elapsed() < ms:
        qapp.processEvents()


def _shown():
    return QToolTip.isVisible() and QToolTip.text() == TIP


def test_the_default_is_two_seconds(policy):
    """"default to 2 seconds" -- the request."""
    assert prefs._get_tooltip_delay() == 2.0
    assert policy.show_delay_ms == 2000


def test_the_delay_is_kept_and_read_back(policy):
    """Saved like the other preferences, so it holds after a restart."""
    assert prefs._set_tooltip_delay(3.5) == 3.5
    assert prefs._settings().value("prefs/tooltip_delay") is not None
    assert prefs._get_tooltip_delay() == 3.5
    assert prefs._set_tooltip_delay(99) == 10.0
    assert prefs._set_tooltip_delay("junk") == 2.0


@pytest.mark.parametrize("seconds", [1.0, 2.0])
def test_no_tooltip_before_the_delay_and_one_after(policy, button, qapp,
                                                   seconds):
    """The hover raises nothing until the chosen wait is over."""
    prefs._set_tooltip_delay(seconds)
    QToolTip.hideText()
    _wait(qapp, 350)
    _hover(button)
    remaining = int(seconds * 1000)
    assert policy._show_timer.interval() == remaining
    _wait(qapp, max(0, remaining - 250))
    assert not _shown(), "the tooltip arrived before its delay"
    _wait(qapp, 500)
    assert _shown(), "the tooltip did not arrive after its delay"


def test_a_delay_shorter_than_qts_own_starts_on_entry(policy, button, qapp):
    """Qt asks only after its own wait, so a shorter one starts on Enter."""
    prefs._set_tooltip_delay(0.2)
    QToolTip.hideText()
    _wait(qapp, 350)
    QApplication.sendEvent(button, QEvent(QEvent.Type.Enter))
    assert policy._show_timer.isActive()
    assert policy._show_timer.interval() == 200
    assert not _shown()
    _wait(qapp, 450)
    assert _shown()


def test_saving_the_dialog_moves_every_tooltip(policy, qtbot,
                                               qt_theme_applied):
    """The slider sits with the other tooltip settings and applies on Save."""
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "TooltipDelay")
    assert slider is not None and slider.value() == 20
    switch = dialog.findChild(QWidget, "TooltipsEnabled")
    page = dialog.findChild(QWidget, "PreferencesTabAppearance")
    assert page.isAncestorOf(slider) and page.isAncestorOf(switch)
    from spacr.qt.widgets.hint_bar import HintBar
    caption = [label for label in dialog.findChildren(QLabel)
               if label.text() == "Tooltip delay"]
    assert len(caption) == 1
    explained = dialog.findChild(HintBar).explains(caption[0])
    assert explained.endswith("Default 2.0 s.")
    assert policy.show_delay_ms == 2000
    slider.setValue(5)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert prefs._get_tooltip_delay() == 0.5
    assert policy.show_delay_ms == 500


def test_reset_puts_the_delay_back_to_two_seconds(policy, qtbot,
                                                  qt_theme_applied):
    """Reset reads the default through the getter, like every other row."""
    from PySide6.QtWidgets import QPushButton as Button

    prefs._set_tooltip_delay(6.0)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "TooltipDelay")
    assert slider.value() == 60
    dialog.findChild(Button, "PreferencesReset").click()
    assert slider.value() == 20


def test_the_delay_follows_the_switch_that_silences_tooltips(qtbot, policy,
                                                             qt_theme_applied):
    """With tooltips off there is nothing to wait for."""
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "TooltipDelay")
    switch = dialog.findChild(QWidget, "TooltipsEnabled")
    switch.setChecked(False)
    assert not slider.isEnabled()
    switch.setChecked(True)
    assert slider.isEnabled()
