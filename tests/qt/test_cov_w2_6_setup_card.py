"""The card's rim: where it points, how much of it lights, and what it costs.

Decoration is never load-bearing here -- a card whose accent cannot be
painted is still a card with working controls -- so the preference lookups
are tested by taking the preference away, and the geometry by asking the
real QPainterPath where the run actually is.
"""
from __future__ import annotations

import builtins

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QHoverEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.setup_card import CORNERS, REFERENCE_CARD, SetupCard


@pytest.fixture
def card(qapp):
    widget = SetupCard(radius=18, arc=280)
    widget.resize(400, 300)
    yield widget
    widget._timer.stop()
    widget.deleteLater()


@pytest.fixture
def no_preferences(monkeypatch):
    """Take the settings store away, as a bare widget test has it."""
    real_import = builtins.__import__
    blocked = {"get_rim_length", "get_rim_lag", "get_rim_mode",
               "get_rim_period", "get_rim_alignment"}

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and blocked.intersection(fromlist):
            raise ImportError("no preference store")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)


# --------------------------------------------------------------------------
# the pointer steers the rim
# --------------------------------------------------------------------------

def test_a_mouse_move_over_the_card_steers_the_rim(card, qapp):
    """It used to set only `_corner`, which the next frame recomputes -- so
    while the pointer was over the CARD nothing steered the rim at all."""
    card.show()
    qapp.processEvents()
    before = card._towards
    point = QPointF(card.width() - 2, 2)          # the top-right corner
    QApplication.sendEvent(card, QMouseEvent(
        QMouseEvent.MouseMove, point, card.mapToGlobal(point.toPoint()),
        Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    assert card._towards != before
    assert card._towards == card.perimeter_position(point)


def test_a_hover_move_steers_it_too(card, qapp):
    """A hover move arrives even when the widget has no mouse grab, which is
    the ordinary case on a card the user is only reading."""
    card.show()
    qapp.processEvents()
    point = QPointF(2, card.height() - 2)         # the bottom-left corner
    QApplication.sendEvent(card, QHoverEvent(
        QEvent.HoverMove, point, QPointF(0, 0), point))
    assert card._towards == card.perimeter_position(point)


def test_a_pointer_at_the_exact_centre_names_no_direction(card):
    """It is ignored rather than read as the top-left corner."""
    centre = QPointF(card.width() / 2.0, card.height() / 2.0)
    assert card.perimeter_position(centre) is None
    card._towards = 0.4
    card.flow_towards(centre)
    assert card._towards == 0.4


def test_a_cursor_at_the_exact_centre_does_not_move_the_accent(card,
                                                               monkeypatch):
    card.move(0, 0)
    monkeypatch.setattr(card, "mapFromGlobal",
                        lambda _pos: QPoint(card.width() // 2,
                                            card.height() // 2))
    assert card._aim_at_the_cursor() is False


def test_a_cursor_off_to_one_side_moves_the_accent(card, monkeypatch):
    monkeypatch.setattr(card, "mapFromGlobal", lambda _pos: QPoint(2, 2))
    card._towards = 0.5
    assert card._aim_at_the_cursor() is True
    assert card._towards != 0.5


def test_the_ray_from_the_centre_walks_the_rim_continuously(card):
    """Projecting onto the nearest EDGE jumps along the diagonals, which is
    what was reported as the rim being unsynced with the pointer."""
    width, height = card.width(), card.height()
    corners = [QPointF(width, 0), QPointF(width, height),
               QPointF(0, height), QPointF(0, 0)]
    fractions = [card.perimeter_position(p) for p in corners]
    assert fractions[0] == pytest.approx(width / (2.0 * (width + height)))
    assert fractions[:3] == sorted(fractions[:3])
    assert fractions[3] == pytest.approx(0.0)


def test_a_pointer_outside_the_card_still_names_a_place_on_the_rim(card):
    """It is a RAY, so it needs no clamping and answers for a pointer that
    has left the window."""
    outside = card.perimeter_position(QPointF(9999, card.height() / 2.0))
    assert outside is not None
    assert 0.0 <= outside <= 1.0


def test_the_pointer_does_not_steer_while_a_circuit_is_running(card):
    """A lap dragged off course by a mouse movement is not a lap, and the
    user cannot tell whether it went round."""
    card.circuit()
    card._towards = 0.0
    card.flow_towards(QPointF(card.width() - 1, 1))
    assert card._towards == 0.0
    assert card.spinning is True


# --------------------------------------------------------------------------
# what the preference store supplies, and what happens without it
# --------------------------------------------------------------------------

def test_without_a_settings_store_every_rim_setting_has_a_shipped_default(
        qapp, no_preferences):
    bare = SetupCard()
    try:
        assert bare._preferred_arc() == 280
        assert bare.ease() == pytest.approx(SetupCard.EASE)
        assert bare.mode() == "glow"
        assert bare.period() == pytest.approx(2.4)
        assert bare.alignment() == "centre"
        assert bare.animates() is False
    finally:
        bare._timer.stop()
        bare.deleteLater()


def test_a_caller_that_names_a_look_keeps_it_whatever_is_stored(qapp):
    named = SetupCard(arc=120, lag=0.5, align="head", mode="beat")
    try:
        assert named._arc == 120
        assert named.ease() == pytest.approx(0.5)
        assert named.alignment() == "head"
        assert named.mode() == "beat"
        assert named.animates() is True
    finally:
        named._timer.stop()
        named.deleteLater()


def test_rereading_the_preferences_takes_the_length_again(card, monkeypatch):
    """The card the user is looking at while they drag a slider is the one
    that answers."""
    monkeypatch.setattr(SetupCard, "_preferred_arc", staticmethod(lambda: 99))
    card.reread_the_preferences()
    assert card._arc == 99


def test_a_still_glow_does_not_animate_but_a_pulse_does(qapp):
    """An arrived plain rim must not cost sixty composites a second over a
    live backdrop for no visible change."""
    for mode, animates in (("glow", False), ("beat", True),
                           ("rainbow", True)):
        widget = SetupCard(mode=mode)
        try:
            assert widget.animates() is animates
        finally:
            widget._timer.stop()
            widget.deleteLater()


# --------------------------------------------------------------------------
# how much of the rim lights
# --------------------------------------------------------------------------

def test_every_card_lights_the_same_fraction_of_its_rim(qapp):
    """The arc preference is a length on the REFERENCE surface; measured
    against each card's own perimeter, 280 px is a sixth of the setup window
    and two fifths of a small popup."""
    big = SetupCard(arc=280, radius=18)
    small = SetupCard(arc=280, radius=18)
    try:
        big.resize(*(int(v) for v in REFERENCE_CARD))
        small.resize(240, 160)
        assert big.accent_span(QRectF(big.rect())) == pytest.approx(
            small.accent_span(QRectF(small.rect())))
    finally:
        for widget in (big, small):
            widget._timer.stop()
            widget.deleteLater()


def test_the_lit_run_is_clamped_at_both_ends(qapp):
    tiny = SetupCard(arc=1)
    huge = SetupCard(arc=100000)
    try:
        tiny.resize(400, 300)
        huge.resize(400, 300)
        assert tiny.accent_span(QRectF(tiny.rect())) == pytest.approx(0.04)
        assert huge.accent_span(QRectF(huge.rect())) == pytest.approx(0.62)
    finally:
        for widget in (tiny, huge):
            widget._timer.stop()
            widget.deleteLater()


def test_a_centred_run_lands_on_the_pointer_and_a_head_run_trails_it(qapp):
    centred = SetupCard(arc=280, align="centre")
    trailing = SetupCard(arc=280, align="head")
    try:
        for widget in (centred, trailing):
            widget.resize(400, 300)
            widget._at = 0.5
        span = centred.accent_span(QRectF(centred.rect()))
        assert centred.accent_start(span) == pytest.approx(0.5 - span / 2.0)
        assert trailing.accent_start(span) == pytest.approx(0.5 - span)
    finally:
        for widget in (centred, trailing):
            widget._timer.stop()
            widget.deleteLater()


# --------------------------------------------------------------------------
# the shape of the lit run
# --------------------------------------------------------------------------

def test_the_lit_run_is_one_unbroken_path_along_the_rim(card):
    """THE WHOLE RIM IS BUILT ONCE and sampled along its length, so the run
    crosses a corner without the seam four separate corner paths have."""
    rect = QRectF(card.rect()).adjusted(1, 1, -1, -1)
    card._at = 0.0
    path = card._accent_path(rect)
    assert path.elementCount() == 49          # one moveTo plus 48 lineTo
    first = path.elementAt(0)
    assert first.isMoveTo()
    assert all(path.elementAt(i).isLineTo()
               for i in range(1, path.elementCount()))
    grown = rect.adjusted(-2, -2, 2, 2)
    for index in range(path.elementCount()):
        element = path.elementAt(index)
        assert grown.contains(QPointF(element.x, element.y))


def test_the_lit_run_follows_the_accent_round_the_card(card):
    rect = QRectF(card.rect()).adjusted(1, 1, -1, -1)
    card._at = 0.0
    top = card._accent_path(rect).elementAt(24)
    card._at = 0.5
    bottom = card._accent_path(rect).elementAt(24)
    assert QPointF(top.x, top.y) != QPointF(bottom.x, bottom.y)
    assert top.y < rect.center().y() < bottom.y


def test_a_run_that_wraps_past_the_start_stays_on_the_rim(card):
    """The run is taken modulo the rim length, so one that begins near the
    end of the perimeter continues past zero rather than stopping."""
    rect = QRectF(card.rect()).adjusted(1, 1, -1, -1)
    card._at = 0.99
    path = card._accent_path(rect)
    grown = rect.adjusted(-2, -2, 2, 2)
    for index in range(path.elementCount()):
        element = path.elementAt(index)
        assert grown.contains(QPointF(element.x, element.y))


# --------------------------------------------------------------------------
# the corner path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index,name", list(enumerate(CORNERS)))
def test_each_corner_path_hugs_the_corner_it_is_named_for(card, index, name):
    rect = QRectF(0.0, 0.0, 400.0, 300.0)
    card._corner = index
    path = card._corner_path(rect)
    points = [QPointF(path.elementAt(i).x, path.elementAt(i).y)
              for i in range(path.elementCount())]
    xs = [p.x() for p in points]
    ys = [p.y() for p in points]
    near_left = min(xs) <= rect.left() + 1
    near_right = max(xs) >= rect.right() - 1
    near_top = min(ys) <= rect.top() + 1
    near_bottom = max(ys) >= rect.bottom() - 1
    expected = {
        "topLeft": (near_left, near_top),
        "topRight": (near_right, near_top),
        "bottomRight": (near_right, near_bottom),
        "bottomLeft": (near_left, near_bottom),
    }[name]
    assert all(expected), (name, points[0], points[-1])
    # It is two edge runs meeting at one arc, so it never reaches the
    # opposite corner.
    assert not (near_left and near_right)
    assert not (near_top and near_bottom)


def test_the_corner_the_accent_names_is_the_one_nearest_it(card):
    for at, name in ((0.0, "topLeft"), (0.25, "topRight"),
                     (0.5, "bottomRight"), (0.75, "bottomLeft")):
        card._at = at
        card._corner = int((card.position + 0.125) % 1.0 * 4) % 4
        assert card.corner() == name


def test_the_nearest_corner_to_a_point_is_the_one_it_is_closest_to(card):
    width, height = card.width(), card.height()
    assert CORNERS[card.nearest_corner(QPointF(1, 1))] == "topLeft"
    assert CORNERS[card.nearest_corner(QPointF(width - 1, 1))] == "topRight"
    assert CORNERS[card.nearest_corner(
        QPointF(width - 1, height - 1))] == "bottomRight"
    assert CORNERS[card.nearest_corner(
        QPointF(1, height - 1))] == "bottomLeft"


# --------------------------------------------------------------------------
# the colour along the run
# --------------------------------------------------------------------------

def test_a_glow_is_one_colour_the_whole_way(qapp):
    widget = SetupCard(mode="glow")
    try:
        accent = QColor("#4A9EFF")
        assert widget.ink_at(0.0, accent) == accent
        assert widget.ink_at(1.0, accent) == accent
    finally:
        widget._timer.stop()
        widget.deleteLater()


def test_a_rainbow_walks_the_hue_along_the_run_and_turns_it_over_time(qapp):
    widget = SetupCard(mode="rainbow")
    try:
        accent = QColor("#4A9EFF")
        tail = widget.ink_at(0.0, accent)
        head = widget.ink_at(0.5, accent)
        assert tail.hueF() != head.hueF()
        widget._phase = 1.0
        assert widget.ink_at(0.0, accent).hueF() != tail.hueF()
    finally:
        widget._timer.stop()
        widget.deleteLater()


# --------------------------------------------------------------------------
# one frame at a time
# --------------------------------------------------------------------------

def test_a_circuit_ends_exactly_where_it_started(card):
    """Floating error across thirty-odd frames would otherwise leave the
    accent a little further round after every lap, and after ten slides it
    would be somewhere the pointer never put it."""
    card._at = card._towards = 0.3
    card.circuit(clockwise=True)
    for _ in range(200):
        card._tick()
        if not card.spinning:
            break
    assert card.spinning is False
    assert card.position == pytest.approx(0.3, abs=1e-9)


def test_an_anticlockwise_circuit_goes_the_other_way(card):
    card._at = card._towards = 0.3
    card.circuit(clockwise=False)
    card._tick()
    assert card._at < 0.3
    for _ in range(200):
        card._tick()
        if not card.spinning:
            break
    assert card.position == pytest.approx(0.3, abs=1e-9)


def test_the_accent_eases_towards_the_pointer_rather_than_jumping(card,
                                                                  monkeypatch):
    monkeypatch.setattr(card, "_aim_at_the_cursor", lambda: False)
    card._at = 0.0
    card._towards = 0.25
    card._laps = 0.0
    card._tick()
    assert 0.0 < card._at < 0.25
    assert card._at == pytest.approx(0.25 * card.ease())


def test_the_card_stops_ticking_when_nobody_is_looking_at_it(card, qapp):
    """A card nobody is looking at does not need sixty frames a second."""
    card.show()
    qapp.processEvents()
    assert card._timer.isActive()
    card.hide()
    qapp.processEvents()
    assert not card._timer.isActive()


def test_an_arrived_glow_stops_repainting_but_a_pulse_does_not(qapp):
    """Sixty needless composites a second over a live backdrop."""
    painted = []
    for mode in ("glow", "beat"):
        widget = SetupCard(mode=mode)
        widget.resize(200, 150)
        widget._aim_at_the_cursor = lambda: False
        widget.update = lambda w=widget: painted.append(w.mode())
        try:
            widget._at = widget._towards = 0.25
            widget._laps = 0.0
            widget._tick()
        finally:
            widget._timer.stop()
            widget.deleteLater()
    assert painted == ["beat"]


def test_the_animation_clock_advances_whatever_else_happens(card):
    """A pulse keeps its rhythm through a circuit and across a slide."""
    before = card._phase
    card._tick()
    assert card._phase > before
