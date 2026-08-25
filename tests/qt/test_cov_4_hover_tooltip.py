"""The setting tooltip outlives the label it is describing.

It is a process-wide singleton holding a plain reference to a widget it does
not own, and the hide is on a timer: hovering a settings label and switching
module inside that delay destroys the label's C++ object while the timer is
still pending. Every call into that dead object raises inside the Qt event
loop, where nothing catches it. The same rule covers the animation: a GIF
that will not decode falls back to text rather than interrupting a hover.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QEvent, QPoint, QRect, QSettings, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QLabel

from spacr.qt import preferences as prefs
from spacr.qt.widgets import hover_tooltip as ht
from spacr.qt.widgets.hover_tooltip import HoverTooltip


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never touch the developer's real preferences."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def tooltip(qtbot):
    """A fresh popup, never the singleton the rest of the session shares."""
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    return popup


# -- which setting an anchor speaks for --------------------------------------

def test_no_anchor_speaks_for_no_setting():
    """A press with nothing under it must not be scoped to a setting."""
    assert ht._anchor_setting_key(None) == ""


def test_a_destroyed_anchor_speaks_for_no_setting(qtbot):
    """Reading a property off a dead widget raises inside the event loop."""
    label = QLabel("Cell diameter")
    label.setProperty("settingKey", "cell_diameter")
    assert ht._anchor_setting_key(label) == "cell_diameter"
    shiboken6.delete(label)
    assert ht._anchor_setting_key(label) == ""


# -- the two clickable words -------------------------------------------------

def test_only_a_left_release_over_the_word_counts_as_a_click(qtbot):
    """A right-click on the footer must not open the documentation."""
    word = ht._LinkWord("Animation", "SettingTooltipAnimationWord")
    qtbot.addWidget(word)
    word.resize(80, 20)
    clicks = []
    word.clicked.connect(lambda: clicks.append(True))

    inside = QPointF(10.0, 10.0)
    word.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, inside, inside,
        Qt.RightButton, Qt.NoButton, Qt.NoModifier))
    assert clicks == []

    word.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, inside, inside,
        Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    assert clicks == [True]


# -- the animation view ------------------------------------------------------

def test_no_animation_clears_the_view_and_says_so(qtbot):
    """A setting with no animation must not keep the previous one playing."""
    view = ht._AnimationView(120)
    qtbot.addWidget(view)
    assert view.load(None) is False
    assert view.frame_count() == 0
    assert view.slug() == ""


def test_an_animation_that_will_not_decode_falls_back_to_text(qtbot,
                                                              monkeypatch):
    """A broken GIF is not worth interrupting a hover for."""
    from spacr.qt.widgets import animation_zoom as az

    monkeypatch.setattr(az, "zoomed_animation", lambda *_a, **_k: None)
    view = ht._AnimationView(120)
    qtbot.addWidget(view)

    class _Animation:
        slug = "cell_diameter"
        path = "/nowhere/cell_diameter.gif"
        title = "Cell diameter"

    assert view.load(_Animation()) is False
    assert view.frame_count() == 0


def test_a_still_image_has_nothing_to_schedule(qtbot):
    """One frame on a repeating timer would spin against the paint loop."""
    view = ht._AnimationView(120)
    qtbot.addWidget(view)
    view._frames = ["one"]
    view._schedule()
    assert view.is_playing() is False


def test_advancing_with_no_frames_does_nothing(qtbot):
    """A timer that fired after the frames were dropped must not index them."""
    view = ht._AnimationView(120)
    qtbot.addWidget(view)
    view._advance()
    assert view.frame_count() == 0


# -- the documentation word --------------------------------------------------

def test_the_documentation_word_does_nothing_without_a_link(tooltip,
                                                            monkeypatch):
    """A setting whose help carries no link must not open a blank page."""
    opened = []
    monkeypatch.setattr(ht.QDesktopServices, "openUrl", opened.append,
                        raising=False)
    tooltip._api_url = ""
    tooltip.open_api_documentation()
    assert opened == []


# -- deriving the animation --------------------------------------------------

def test_an_invalid_animation_registry_leaves_text_help_alone(tooltip,
                                                              qtbot,
                                                              monkeypatch):
    """A broken registry must not cost the user the written explanation."""
    from spacr import setting_animations as sa

    def _boom(_key):
        raise sa.SettingAnimationError("two entries claim one slug")

    monkeypatch.setattr(sa, "animation_for_setting", _boom)
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    label.setProperty("settingKey", "cell_diameter")
    assert tooltip._resolve_animation(label, ht._DERIVE) is None


# -- placement ---------------------------------------------------------------

def test_a_popup_with_no_anchor_is_placed_at_the_origin(tooltip):
    """A tooltip with nothing to dock under must still land somewhere."""
    tooltip._position_under(None)
    assert tooltip.pos().x() >= 0


def test_a_popup_with_no_screen_is_placed_where_the_anchor_said(tooltip,
                                                                monkeypatch):
    """With no screen to clamp against, the anchor's own point is the answer."""
    class _NoScreens:
        @staticmethod
        def screenAt(_point):
            return None

        @staticmethod
        def primaryScreen():
            return None

    monkeypatch.setattr(ht, "QGuiApplication", _NoScreens)
    tooltip._position_under(None)
    assert tooltip.pos() == QPoint(0, 0)


def test_a_popup_that_will_not_fit_below_flips_above_its_anchor(tooltip):
    """Docked below the screen edge the tooltip would be unreadable."""
    geometry = QGuiApplication.primaryScreen().availableGeometry()

    class _DiesAfterTheFirstAsk:
        def __init__(self):
            self.asks = 0

        def rect(self):
            return QRect(0, 0, 40, 20)

        def mapToGlobal(self, _point):
            self.asks += 1
            if self.asks > 1:
                raise RuntimeError("Internal C++ object already deleted")
            return QPoint(geometry.left(), geometry.bottom())

    anchor = _DiesAfterTheFirstAsk()
    tooltip.resize(200, 120)
    tooltip._position_under(anchor)
    assert anchor.asks == 2, "the flip-above path was never taken"
    assert tooltip.pos().y() < geometry.bottom()


# -- claiming the anchor -----------------------------------------------------

def _watch_native_tooltip(monkeypatch):
    """Record every attempt to hide Qt's own tooltip popup."""
    hidden = []

    class _ToolTip:
        @staticmethod
        def hideText():
            hidden.append(True)

    monkeypatch.setattr(ht, "QToolTip", _ToolTip)
    return hidden


def test_claiming_a_live_anchor_hides_the_native_tooltip(tooltip, qtbot,
                                                         monkeypatch):
    """A native tooltip already on screen would sit over this one."""
    hidden = _watch_native_tooltip(monkeypatch)
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    tooltip._claim_anchor(label)
    assert hidden == [True]


def test_claiming_no_anchor_takes_no_tooltip_duty(tooltip, monkeypatch):
    """A hover that arrived with no widget has no duty to take."""
    hidden = _watch_native_tooltip(monkeypatch)
    tooltip._claim_anchor(None)
    assert hidden == []


def test_claiming_a_destroyed_anchor_takes_no_tooltip_duty(tooltip, qtbot,
                                                           monkeypatch):
    """There is no native tooltip left to suppress on a dead widget."""
    hidden = _watch_native_tooltip(monkeypatch)
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    shiboken6.delete(label)
    tooltip._claim_anchor(label)
    assert hidden == []


# -- hiding ------------------------------------------------------------------

def test_the_popup_stays_while_the_cursor_is_still_on_the_anchor(tooltip,
                                                                 qtbot,
                                                                 monkeypatch):
    """Leaving the label towards the popup must not dismiss it."""
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    label.resize(80, 20)
    tooltip._anchor = label
    monkeypatch.setattr(tooltip, "_pointer_is_on_me", lambda: False)
    hidden = []
    monkeypatch.setattr(tooltip, "hide", lambda: hidden.append(True))
    tooltip._maybe_hide()
    assert hidden == []
    assert tooltip._anchor is label


def test_leaving_the_popup_itself_restarts_the_hide_timer(tooltip):
    """Leaving the popup is deliberate, so it gets a shorter grace period."""
    tooltip.show()
    tooltip.cancel_hide()
    tooltip.leaveEvent(QEvent(QEvent.Leave))
    assert tooltip._hide_timer.isActive()
    tooltip.cancel_hide()
    tooltip.hide()
