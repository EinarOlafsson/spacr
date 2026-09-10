"""What the card does when the things it decorates itself with are missing.

Every one of these is a decoration: the hover typing, the palette, the global
cursor and the ``spaceout`` dressing. None of them may take the card down with
them, so each is removed in turn and the card is asked to go on working.
"""
from __future__ import annotations

import builtins

import pytest

from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QHoverEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.setup_card import SPACEOUT_RIM_SPREAD, SetupCard

pytestmark = pytest.mark.qt


@pytest.fixture
def card(qapp):
    widget = SetupCard(radius=18, arc=280)
    widget.resize(400, 300)
    yield widget
    widget._timer.stop()
    widget.deleteLater()


def block_imports(monkeypatch, *names):
    """Make ``from x import <name>`` raise, for each of ``names``."""
    real_import = builtins.__import__
    blocked = set(names)

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and blocked.intersection(fromlist):
            raise ImportError(f"blocked: {sorted(blocked.intersection(fromlist))}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)


@pytest.fixture
def forget_the_dressing():
    """Hand back a way to drop the cached ``(enabled, drift)`` pair.

    The cache is a CLASS attribute, so it is shared with every other card
    in the process -- including ones an earlier test showed and never hid,
    whose 16ms timers are still running. ``_tick`` asks ``animates()``,
    which asks ``spaceout()``, which fills the cache in again; and pytest-qt
    spins the event loop once BETWEEN fixture setup and the test body, so a
    fixture that cleared the cache on the way in would hand over a cache
    that a stray card had already refilled. Clearing it is therefore the
    caller's first line rather than this fixture's; what the fixture owns is
    putting the process back the way it found it.
    """
    saved = SetupCard._DRESSING

    def forget():
        SetupCard._DRESSING = None

    try:
        yield forget
    finally:
        SetupCard._DRESSING = saved


# --------------------------------------------------------------------------
# the theme dressing
# --------------------------------------------------------------------------

def test_a_card_with_no_theme_dressing_is_a_plain_glow(card, monkeypatch,
                                                       forget_the_dressing):
    """No ``spaceout`` functions to ask means the rim is not spectral.

    The lookup is cached on the class, so a card built before the theme was
    importable must still answer -- with the ordinary glow rather than by
    raising into a paint.
    """
    forget_the_dressing()
    block_imports(monkeypatch, "spaceout_enabled", "spaceout_drift")

    assert SetupCard._dressing() is None
    assert card.spaceout() is False
    # No drift to add, so the only hue the run carries is the spread along
    # its own length: the tail sits at 0 and nothing moves it over time.
    assert card.spaceout_hue(0.0) == 0.0
    assert card.spaceout_hue(1.0) == pytest.approx(SPACEOUT_RIM_SPREAD)


def test_the_dressing_is_looked_up_again_after_it_becomes_importable(
        card, forget_the_dressing):
    """The lookup is cached only once it succeeds, so a failed one is not
    remembered as an answer."""
    forget_the_dressing()

    assert SetupCard._DRESSING is None
    first = SetupCard._dressing()
    assert first is not None
    assert SetupCard._dressing() is first


# --------------------------------------------------------------------------
# the hover
# --------------------------------------------------------------------------

def test_a_hover_that_cannot_be_typed_leaves_the_rim_where_it_was(card, qapp):
    """Without ``QEvent`` the hover cannot be recognised, so it steers
    nothing -- and the widget still handles the event as a widget should."""
    card.show()
    qapp.processEvents()
    corner = QPointF(card.width() - 2.0, card.height() - 2.0)
    card.flow_towards(QPointF(2.0, 2.0))
    settled = card._towards

    with pytest.MonkeyPatch.context() as patch:
        block_imports(patch, "QEvent")
        hover = QHoverEvent(QEvent.HoverMove, corner, QPointF(0, 0), corner)
        card.event(hover)

    assert card._towards == settled
    assert card._towards != card.perimeter_position(corner)

    # The same hover with the typing available does steer it, which is what
    # makes the reading above a loss rather than a coincidence.
    QApplication.sendEvent(card, QHoverEvent(
        QEvent.HoverMove, corner, QPointF(0, 0), corner))
    assert card._towards == card.perimeter_position(corner)


# --------------------------------------------------------------------------
# the paint
# --------------------------------------------------------------------------

def test_a_card_whose_palette_will_not_resolve_still_paints(card, monkeypatch):
    """The decoration is skipped; the widget is still rendered and usable."""
    from spacr.qt import theme

    def _no_palette():
        raise RuntimeError("no palette")

    monkeypatch.setattr(theme, "active_palette", _no_palette)

    pixmap = card.grab()

    assert not pixmap.isNull()
    assert pixmap.size() == card.size()
    # And the controls the card carries go on working: the rim still takes
    # an aim, it simply had nothing to draw itself with this frame.
    card.flow_towards(QPointF(card.width() - 2.0, 2.0))
    assert card._towards == card.perimeter_position(
        QPointF(card.width() - 2.0, 2.0))


# --------------------------------------------------------------------------
# the global cursor
# --------------------------------------------------------------------------

def test_a_frame_with_no_readable_cursor_keeps_the_aim_it_had(card):
    """``_aim_at_the_cursor`` reads the GLOBAL cursor; with no ``QCursor`` to
    read there is nothing to aim at, and the accent goes on easing towards
    wherever it was last pointed."""
    card.flow_towards(QPointF(card.width() - 2.0, 2.0))
    aimed = card._towards
    card._at = 0.0

    with pytest.MonkeyPatch.context() as patch:
        block_imports(patch, "QCursor")
        assert card._aim_at_the_cursor() is False
        assert card._towards == aimed
        before = card._at
        card._tick()

    assert card._towards == aimed
    assert card._at != before          # it eased, it just did not re-aim
