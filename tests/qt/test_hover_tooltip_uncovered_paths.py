"""The popup outlives the things it points at, and has to survive that.

It is a process-wide singleton holding a plain reference to a widget it does
not own, and its hide is deferred by a timer. Between the hover and the hide
the anchor's C++ object can be destroyed, the module can be switched, and the
popup itself can be torn down. Every one of those turns an ordinary geometry
question into a ``RuntimeError`` raised inside the Qt event loop, where there
is nothing to catch it.

The reveal is the other half: pressing **Animation** names ONE setting, and it
has to work whether or not the popup is on screen when it is pressed.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel

from spacr.qt import preferences as prefs
from spacr.qt.widgets.hover_tooltip import HoverTooltip

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never touch the developer's real preferences."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in os.path.normpath(store.fileName()), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def tooltip(qtbot):
    """A fresh popup, never the singleton the rest of the session shares."""
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    return popup


@pytest.fixture
def anchor(qtbot):
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    label.show()
    return label


# -- the popup's own C++ half can be gone -------------------------------------

def test_a_destroyed_popup_reports_the_pointer_is_not_on_it(qapp):
    """Asking a torn-down popup where the pointer is must answer, not raise.

    The hide runs from a timer. A popup destroyed while that timer is still
    pending would raise ``RuntimeError`` from inside the Qt event loop, and
    the honest answer is that nothing can be hovering a window that is gone.
    """
    popup = HoverTooltip()
    shiboken6.delete(popup)

    assert HoverTooltip._pointer_is_on_me(popup) is False


def test_a_live_popup_answers_from_its_own_geometry(tooltip, anchor):
    """The geometry decides, so a popup nowhere near the pointer says so."""
    tooltip.show_for(anchor, "<p>Expected cell diameter in pixels.</p>", immediate=True)
    tooltip.move(4000, 4000)

    assert tooltip._pointer_is_on_me() is False


# -- the anchor can be gone ---------------------------------------------------

def test_a_popup_that_has_lost_its_anchor_simply_hides(tooltip, anchor):
    """Nothing can be hovering a label that is no longer there."""
    tooltip.show_for(anchor, "<p>Expected cell diameter in pixels.</p>", immediate=True)
    tooltip.move(4000, 4000)
    tooltip._anchor = None

    tooltip._maybe_hide()

    assert not tooltip.isVisible()


def test_a_popup_whose_anchor_was_destroyed_forgets_it_and_hides(tooltip,
                                                                 qtbot):
    """The dead reference is dropped, so the next hide does not ask again."""
    label = QLabel("Cell diameter")
    label.show()
    tooltip.show_for(label, "<p>Expected cell diameter in pixels.</p>", immediate=True)
    tooltip.move(4000, 4000)
    shiboken6.delete(label)

    tooltip._maybe_hide()

    assert tooltip._anchor is None
    assert not tooltip.isVisible()


# -- the reveal, pressed while the popup is not on screen ---------------------

def test_the_reveal_can_be_pressed_before_the_popup_is_shown(tooltip):
    """A press names a setting; it does not need a window to do that.

    ``toggle_animation`` re-docks the popup only when there is one on screen
    and an anchor to dock it under. Without either, the state still flips.
    """
    assert not tooltip.isVisible()
    tooltip._setting_key = "cell_diameter"
    before = tooltip.pos()

    tooltip.toggle_animation()

    assert tooltip.toggled_setting() == "cell_diameter"
    assert tooltip.animations_shown() is True
    assert tooltip.pos() == before
    assert not tooltip.isVisible()


def test_pressing_the_reveal_again_folds_it_away(tooltip):
    """The word is a toggle, and it stays scoped to the setting it named."""
    tooltip._setting_key = "cell_diameter"

    tooltip.toggle_animation()
    tooltip.toggle_animation()

    assert tooltip.toggled_setting() == "cell_diameter"
    assert tooltip.animations_shown() is False


def test_a_reveal_on_one_setting_does_not_follow_to_the_next(tooltip):
    """One press must not put every later hover back on the decode path."""
    tooltip._setting_key = "cell_diameter"
    tooltip.toggle_animation()
    assert tooltip.animations_shown() is True

    tooltip._setting_key = "nucleus_channel"

    assert tooltip.animations_shown() is False, (
        "the next setting falls back to the preference, which is off")


def test_sticky_tooltip_swallows_native_help_delivery_without_losing_accessible_text(
        qapp, qtbot, tooltip):
    """Claiming an anchor stops its native help event, preserving ordinary delivery."""
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QAccessible, QHelpEvent
    from PySide6.QtWidgets import QApplication, QToolTip

    class HelpReceiver(QLabel):
        """Record events that reach the widget instead of opening native help."""
        def __init__(self):
            super().__init__('Cell diameter')
            self.delivered = []

        def event(self, event):
            if event.type() in (QEvent.ToolTip, QEvent.User):
                self.delivered.append(event.type())
                event.accept()
                return True
            return super().event(event)

    label = HelpReceiver()
    qtbot.addWidget(label)
    body = 'Expected cell diameter in pixels.'
    label.setToolTip(body)
    label.setToolTipDuration(-1)
    label.resize(240, 40)
    label.show()
    qapp.processEvents()
    accessible = QAccessible.queryAccessibleInterface(label)
    assert accessible is not None
    assert accessible.text(QAccessible.Text.Description) == body
    point = QPoint(8, 8)
    QApplication.sendEvent(label, QHelpEvent(QEvent.ToolTip, point, label.mapToGlobal(point)))
    QApplication.sendEvent(label, QEvent(QEvent.User))
    assert label.delivered == [QEvent.ToolTip, QEvent.User]

    tooltip.show_for(label, body, animation=None, immediate=True)
    tooltip.show_for(label, body, animation=None, immediate=True)
    assert tooltip.isVisible() and tooltip._label.text() == body
    QApplication.sendEvent(label, QHelpEvent(QEvent.ToolTip, point, label.mapToGlobal(point)))
    QApplication.sendEvent(label, QEvent(QEvent.User))
    assert label.delivered == [QEvent.ToolTip, QEvent.User, QEvent.User]
    assert label.toolTip() == body and label.toolTipDuration() == -1
    assert accessible.text(QAccessible.Text.Description) == body
    assert tooltip.isVisible() and not QToolTip.isVisible()
