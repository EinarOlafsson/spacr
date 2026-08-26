"""What the setup card's lit rim costs, and that it still looks the same.

The rim is drawn as hundreds of short strokes, because a QPen carries one
colour and the run has to fade along its length. Everything each stroke
asks for -- the mode, the period, the alignment, the pulse, the drift --
is the same for every stroke in a frame, so asking per stroke turned a
frame into several hundred openings of the settings store.

THE ASSERTIONS HERE ARE COUNTS, NOT CLOCKS. A wall-clock threshold says
different things on different machines and on a busy one says nothing at
all; what actually went wrong is that the work per frame scaled with the
number of segments, and a count catches that on any machine.
"""
from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QRectF, QSize          # noqa: E402
from PySide6.QtGui import QImage                           # noqa: E402
from PySide6.QtWidgets import QApplication                 # noqa: E402

from spacr.qt.widgets.setup_card import SetupCard          # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config():
    """Every rim key this file writes, saved and put back."""
    from spacr.qt import preferences

    keys = ("rim_length", "rim_lag", "rim_alignment", "rim_mode", "rim_period")
    before = {k: getattr(preferences, f"get_{k}")() for k in keys}
    yield preferences
    for key, value in before.items():
        getattr(preferences, f"set_{key}")(value)


def _card(app, **kwargs):
    """A card the size the report measures, with its timer held still."""
    card = SetupCard(**kwargs)
    card.resize(900, 600)
    card._timer.stop()
    return card


def _paint_once(card):
    """Draw one frame into an image, off screen, and give back the image."""
    image = QImage(QSize(card.width(), card.height()), QImage.Format_ARGB32)
    image.fill(0)
    card.render(image)
    return image


def _count(monkeypatch, module, name):
    """Replace ``module.name`` with a counting wrapper. Returns the list."""
    seen = []
    original = getattr(module, name)

    def counted(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return seen


# --------------------------------------------------------------------------
# the cost of a frame no longer scales with the length of the run
# --------------------------------------------------------------------------

def test_the_settings_store_is_opened_a_handful_of_times_a_frame(
        app, monkeypatch):
    """Every stroke used to ask, and a long rim is hundreds of strokes."""
    from spacr.qt import preferences

    card = _card(app)
    try:
        _paint_once(card)                       # warm anything cacheable
        opened = _count(monkeypatch, preferences, "_settings")
        _paint_once(card)
        assert len(opened) <= 8, (
            f"one frame opened the settings store {len(opened)} times")
    finally:
        card.deleteLater()


def test_the_cost_of_a_frame_does_not_grow_with_the_length_of_the_rim(
        app, monkeypatch, own_config):
    """THIS IS THE FAULT ITSELF. A rim twice as long is twice as many
    strokes, and while each stroke asked the settings store for the mode a
    longer rim cost proportionally more to draw."""
    from spacr.qt import preferences

    counts = {}
    for length in (40, 900):
        own_config.set_rim_length(length)
        card = _card(app)
        try:
            _paint_once(card)
            opened = _count(monkeypatch, preferences, "_settings")
            _paint_once(card)
            counts[length] = len(opened)
            monkeypatch.undo()
        finally:
            card.deleteLater()
    assert counts[900] == counts[40], (
        f"a long rim cost more lookups than a short one: {counts}")


def test_each_taste_preference_is_read_at_most_once_a_frame(app, monkeypatch):
    """Mode, period and alignment cannot change while one frame is drawn."""
    from spacr.qt import preferences

    card = _card(app, mode="rainbow")           # the busiest colouring
    try:
        _paint_once(card)
        reads = {name: _count(monkeypatch, preferences, f"get_{name}")
                 for name in ("rim_period", "rim_alignment")}
        _paint_once(card)
        for name, seen in reads.items():
            assert len(seen) <= 1, f"{name} was read {len(seen)} times"
    finally:
        card.deleteLater()


def test_the_rim_path_is_built_once_per_size_not_once_per_frame(app):
    """The rounded path is a function of the rectangle and the radius, and
    measuring its length is not free."""
    card = _card(app)
    try:
        rect = QRectF(card.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        first, first_length = card._rim(rect)
        again, again_length = card._rim(rect)
        assert again is first, "the rim was rebuilt for the same rectangle"
        assert again_length == first_length

        card.resize(500, 400)
        wider = QRectF(card.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        rebuilt, rebuilt_length = card._rim(wider)
        assert rebuilt is not first, "a resize must build the rim again"
        assert rebuilt_length != first_length
    finally:
        card.deleteLater()


def test_the_lit_fraction_is_measured_again_when_the_length_changes(app):
    """The card the user is looking at while they drag the slider is the
    one that answers, so the cached span cannot outlive the preference."""
    card = _card(app, arc=100)
    try:
        rect = QRectF(card.rect())
        short = card.accent_span(rect)
        assert card.accent_span(rect) == short
        card._arc = 600
        assert card.accent_span(rect) > short
    finally:
        card.deleteLater()


# --------------------------------------------------------------------------
# holding the answers for a frame changes nothing the user can see
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ("glow", "rainbow", "beat"))
@pytest.mark.parametrize("align", ("centre", "head"))
def test_the_same_state_paints_the_same_pixels(app, mode, align):
    """Held or read fresh, a frame drawn from one state is one picture."""
    digests = []
    for _ in range(2):
        card = _card(app, mode=mode, align=align, arc=280)
        try:
            card._at = card._towards = 0.137
            card._phase = 1.7
            digests.append(hashlib.sha256(
                _paint_once(card).bits().tobytes()).hexdigest())
        finally:
            card.deleteLater()
    assert digests[0] == digests[1]


def test_a_cold_card_and_a_warm_one_paint_the_same_frame(app):
    """Whatever a card has cached from earlier frames must not show."""
    warm = _card(app, mode="rainbow", arc=280)
    cold = _card(app, mode="rainbow", arc=280)
    try:
        for card in (warm, cold):
            card._at = card._towards = 0.62
            card._phase = 3.9
        for _ in range(5):                      # give `warm` a history
            warm._phase = 3.9
            _paint_once(warm)
        assert (hashlib.sha256(_paint_once(warm).bits().tobytes()).hexdigest()
                == hashlib.sha256(
                    _paint_once(cold).bits().tobytes()).hexdigest())
    finally:
        warm.deleteLater()
        cold.deleteLater()


def test_a_mode_changed_between_frames_reaches_the_very_next_one(
        app, own_config):
    """Nothing is held BETWEEN frames -- a preference dialog open while the
    setting changes is the normal case, not the exotic one."""
    own_config.set_rim_mode("glow")
    card = _card(app)
    try:
        _paint_once(card)
        assert card.mode() == "glow"
        own_config.set_rim_mode("beat")
        assert card.mode() == "beat", "the frame's answer outlived its frame"
        _paint_once(card)
        assert card.mode() == "beat"
    finally:
        card.deleteLater()


def test_nothing_is_held_once_the_frame_is_over(app):
    card = _card(app)
    try:
        assert card._frame is None
        _paint_once(card)
        assert card._frame is None
    finally:
        card.deleteLater()


def test_a_frame_that_fails_still_lets_go_of_its_answers(app, monkeypatch):
    """Decoration is never load-bearing: a paint that throws must not leave
    the card answering every later question from a dead frame."""
    card = _card(app)
    try:
        def explode(*args, **kwargs):
            raise RuntimeError("no accent today")

        monkeypatch.setattr(SetupCard, "_paint_accent", explode)
        _paint_once(card)                       # paintEvent swallows it
        assert card._frame is None
        monkeypatch.undo()
        assert card.mode() in ("glow", "rainbow", "beat")
    finally:
        card.deleteLater()


# --------------------------------------------------------------------------
# the animations still animate, and the rim still follows the mouse
# --------------------------------------------------------------------------

def test_the_pulse_still_breathes_from_one_frame_to_the_next(app):
    """`beat` is now read once a frame rather than once a stroke, which
    must not flatten the pulse across frames."""
    card = _card(app, mode="beat")
    try:
        # WITHIN A QUARTER OF THE CYCLE, where the sine is monotonic:
        # sampled across a whole one, symmetric phases share a value and
        # the count says nothing.
        period = card.period()
        seen = set()
        for share in (0.0, 0.06, 0.12, 0.18, 0.25):
            card._phase = period * share
            seen.add(round(card.beat(), 6))
        assert len(seen) == 5, f"the pulse stopped breathing: {seen}"
        assert min(seen) > 0.0, "a rim that vanishes reads as a fault"
    finally:
        card.deleteLater()


def test_the_spectrum_still_spreads_along_the_run_under_spaceout(app):
    """One drift for the frame, but the hue must still walk the run."""
    card = _card(app, mode="rainbow")
    saved = SetupCard._DRESSING
    try:
        SetupCard._DRESSING = (lambda: True, lambda: 90.0)
        card._frame = {}                        # as a frame being drawn
        hues = [card.spaceout_hue(along) for along in (0.0, 0.5, 1.0)]
        card._frame = None
        assert len(set(round(h, 6) for h in hues)) == 3, (
            f"the run carried one flat colour: {hues}")
    finally:
        SetupCard._DRESSING = saved
        card.deleteLater()


def test_the_rim_still_aims_at_a_cursor_outside_the_card(app):
    """READ, NOT RECEIVED. A widget is sent no mouse events once the
    pointer leaves it, and this rim is meant to follow one that has."""
    from PySide6.QtGui import QCursor

    card = _card(app)
    card.move(0, 0)
    card.show()
    app.processEvents()
    try:
        aimed = []
        for global_point in ((-400, -300), (2400, 1500)):
            here = card.mapFromGlobal(QCursor.pos().__class__(*global_point))
            assert not card.rect().contains(here), "point was not outside"
            card._towards = -1.0
            card._frame = None
            target = card.perimeter_position(QPointF(here))
            assert target is not None, "a point outside named no direction"
            aimed.append(target)
        assert aimed[0] != aimed[1], (
            "two opposite corners aimed the rim at the same place")
    finally:
        card.close()
        card.deleteLater()


def test_aiming_at_the_cursor_reports_whether_it_moved(app, monkeypatch):
    """The tick uses this to decide whether anything changed."""
    from PySide6.QtGui import QCursor

    card = _card(app)
    card.show()
    app.processEvents()
    try:
        monkeypatch.setattr(QCursor, "pos", staticmethod(
            lambda: card.mapToGlobal(card.rect().topLeft())))
        card._aim_at_the_cursor()
        assert card._aim_at_the_cursor() is False, (
            "a still cursor reported movement")
        monkeypatch.setattr(QCursor, "pos", staticmethod(
            lambda: card.mapToGlobal(card.rect().bottomRight())))
        assert card._aim_at_the_cursor() is True, (
            "the rim stopped noticing the cursor move")
    finally:
        card.close()
        card.deleteLater()
