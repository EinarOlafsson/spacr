"""Real popup lifetime after hovering, leaving, and disabling tooltips."""

import pytest
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QCursor, QHelpEvent
from PySide6.QtWidgets import QApplication, QLabel, QToolTip, QWidget

from spacr.qt import preferences, tooltip_policy
from spacr.qt.widgets.hover_tooltip import HoverTooltip


@pytest.fixture
def lifetime_context(qtbot):
    tooltip_policy.uninstall_tooltip_policy()
    preferences.set_tooltips_enabled(True)
    tooltip_policy.install_tooltip_policy()
    host = QWidget()
    host.resize(300, 120)
    anchor = QLabel("Help", host)
    anchor.resize(100, 40)
    qtbot.addWidget(host)
    host.show()
    qtbot.wait(20)
    yield anchor, tooltip_policy.tooltip_policy()
    tooltip_policy.uninstall_tooltip_policy()
    QToolTip.hideText()
    preferences.set_tooltips_enabled(True)


def test_native_help_survives_qts_default_expiry_while_hovered(
        lifetime_context, qtbot):
    anchor, policy = lifetime_context
    anchor.setToolTip("Read this help at your own pace.")
    QCursor.setPos(anchor.mapToGlobal(anchor.rect().center()))
    event = QHelpEvent(QEvent.ToolTip, QPoint(5, 5),
                       anchor.mapToGlobal(QPoint(180, 80)))
    policy.eventFilter(anchor, event)
    policy._show_now()
    assert QToolTip.isVisible()
    qtbot.wait(11000)
    assert QToolTip.isVisible(), "Qt expired the help while its target was hovered"
    preferences.set_tooltips_enabled(False)
    qtbot.waitUntil(lambda: not QToolTip.isVisible(), timeout=1000)


@pytest.mark.parametrize("leave", ["anchor", "popup"])
def test_rich_help_lingers_one_second_after_leaving_either_surface(
        lifetime_context, qtbot, leave):
    anchor, _ = lifetime_context
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    popup.show_for(anchor, "An explanation to finish reading.",
                   animation=None, immediate=True)
    QCursor.setPos(QPoint(2000, 2000))
    qtbot.wait(20)
    popup.cancel_hide()
    if leave == "anchor":
        popup.start_hide()
    else:
        QApplication.sendEvent(popup, QEvent(QEvent.Leave))
    qtbot.wait(850)
    assert popup.isVisible(), "The popup disappeared before the one-second grace"
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=700)
