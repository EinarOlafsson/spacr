"""The setup screen's second pass, from what was reported on seeing it.

    "the startup has some black boxes that should be removed. there should
     be a lag after the first next click to make time for Hello in the
     chosen language. the claude gpt and gemeni buttons should be the logos
     for each not just buttons. the blue rim is much much better but it
     could become more transparent towards the ends an longer and more
     responsive ... most notably make the transitions smoother!"

FIVE SEPARATE CLAIMS, each measured rather than asserted from the source.
The black boxes in particular were invisible to a code reading: the palette
that produces them has ``bg`` at literal ``#000000``, so a container nobody
tagged paints pure black over the drifting backdrop -- and the widget tree
looks perfectly ordinary either way. It took a grab and a pixel count.

INVARIANTS 10: every one of these is decoration. If none of it can be drawn
the slides still ask the same questions and write the same answers.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QRectF                    # noqa: E402
from PySide6.QtWidgets import (QApplication, QPushButton,     # noqa: E402
                               QWidget)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A config dir of this test's own, or it answers the setup screen on
    the maintainer's machine and they never see it again."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences

    importlib.reload(preferences)
    yield
    importlib.reload(preferences)


@pytest.fixture()
def themed(app):
    """The application styled, which is how the dialog is actually met.

    WITHOUT THIS EVERY COLOUR ASSERTION IS A LIE: with no stylesheet the
    labels come out in Qt's default black-on-grey, and a test that measured
    that would be measuring a screen no user ever sees.
    """
    from spacr.qt.preferences import apply_preferences_to_app

    apply_preferences_to_app(app)
    return app


@pytest.fixture()
def slides(themed):
    from spacr.qt.widgets.setup_slides import SetupSlides

    dialog = SetupSlides()
    dialog.resize(720, 560)
    dialog.show()
    for _ in range(8):
        themed.processEvents()
    yield dialog
    dialog.deleteLater()


# ---------------------------------------------------------------------------
# 1. The black boxes
# ---------------------------------------------------------------------------

def _pure_black_fraction(dialog, widget) -> float:
    """How much of ``widget`` is literal #000000 in a grab of ``dialog``."""
    image = dialog.grab().toImage()
    origin = widget.mapTo(dialog, widget.rect().topLeft())
    black = total = 0
    for y in range(origin.y() + 2, origin.y() + widget.height() - 2, 2):
        for x in range(origin.x() + 2, origin.x() + widget.width() - 2, 2):
            if not (0 <= x < image.width() and 0 <= y < image.height()):
                continue
            pixel = image.pixelColor(x, y)
            total += 1
            if (pixel.red(), pixel.green(), pixel.blue()) == (0, 0, 0):
                black += 1
    return black / max(1, total)


def test_the_page_stack_is_not_a_black_box(slides):
    """MEASURED, NOT READ. `bg` is literally #000000 in this palette, so an
    untagged container over the backdrop is a pure black rectangle -- and
    the widget tree gives no sign of it."""
    assert _pure_black_fraction(slides, slides._pages) < 0.02


def test_every_page_is_see_through(slides):
    for index in range(slides._pages.count()):
        page = slides._pages.widget(index)
        if page.width() < 20 or page.height() < 20:
            continue
        assert _pure_black_fraction(slides, page) < 0.02, index


def test_the_provider_strip_is_see_through(slides):
    strip = [w for w in slides.findChildren(QWidget)
             if w.property("spacrProviderStrip")]
    assert strip, "the provider strip is not tagged, so nothing clears it"
    assert _pure_black_fraction(slides, strip[0]) < 0.05


def test_the_controls_are_still_opaque(slides):
    """THE CONTAINERS ONLY. A control you can see through is a control you
    cannot read, so the combos keep their own surface."""
    from PySide6.QtWidgets import QComboBox

    for box in slides.findChildren(QComboBox):
        assert not box.property("transparentBg"), box.objectName()


# ---------------------------------------------------------------------------
# 2. The pause for the greeting
# ---------------------------------------------------------------------------

def test_the_first_next_waits_for_the_greeting(slides):
    """The greeting lives on the page being left, so without a pause it is
    on screen for one frame of a fade."""
    assert slides.next() == 0, "the first Next did not wait"
    assert slides._pending is not None
    assert not slides._next.isEnabled(), (
        "a second click during the pause would skip a slide")


def test_the_pause_ends_on_the_next_slide(slides):
    slides.next()
    slides._finish_the_greeting()
    assert slides.slide() == 1
    assert slides._next.isEnabled()
    assert slides._pending is None


def test_only_the_first_next_waits(slides):
    """A pause on every return to the first slide would be a delay the user
    has already sat through."""
    slides.next()
    slides._finish_the_greeting()
    assert slides.next() == 2
    slides.previous()
    assert slides.slide() == 1
    slides.previous()
    assert slides.slide() == 0
    assert slides.next() == 1, "it waited a second time"


def test_the_wait_is_long_enough_to_read(slides):
    from spacr.qt.widgets.setup_slides import GREETING_MS

    assert GREETING_MS >= 500, "shorter than this and it is a stutter"
    assert GREETING_MS <= 2000, "longer and it reads as the app hanging"


# ---------------------------------------------------------------------------
# 3. The providers are marks
# ---------------------------------------------------------------------------

def test_no_provider_is_a_plain_button(slides):
    holder = slides._editors["ai_provider"]
    for mark in holder._buttons.values():
        assert not isinstance(mark, QPushButton), (
            "three words in three boxes is a list wearing buttons")


def test_each_provider_draws_its_own_mark():
    """Three marks that came out the same shape would be three of nothing."""
    from spacr.qt.widgets.provider_marks import MARKS, mark_for

    box = QRectF(0, 0, 40, 40)
    shapes = {}
    for code in MARKS:
        path = mark_for(code, box)
        assert path is not None and not path.isEmpty(), code
        shapes[code] = round(path.length(), 3)
    assert len(set(shapes.values())) == len(shapes), shapes


def test_the_three_marks_carry_a_comparable_weight(qapp):
    """The lightest of three logos in a row reads as the disabled one.

    Measured as INK, not eyeballed: the burst was first drawn as tapered
    points -- narrow at the tip, all the area at the hub -- and came out at
    a third of the knot's coverage beside it, which on screen looked like a
    provider that was not available.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    from spacr.qt.widgets.provider_marks import MARKS, mark_for

    inks = {}
    for code in MARKS:
        image = QImage(64, 64, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawPath(mark_for(code, QRectF(0, 0, 64, 64)))
        painter.end()
        inks[code] = sum(1 for y in range(64) for x in range(64)
                         if image.pixelColor(x, y).alpha() > 120) / 4096.0
    assert min(inks.values()) > 0.20, inks
    assert max(inks.values()) / min(inks.values()) < 2.0, inks


def test_an_unknown_provider_draws_nothing_rather_than_guessing():
    from spacr.qt.widgets.provider_marks import mark_for

    assert mark_for("copilot", QRectF(0, 0, 40, 40)) is None


def test_the_mark_is_inside_the_box_it_was_given():
    """A path that overflows its box paints over the label beneath it."""
    from spacr.qt.widgets.provider_marks import MARKS, mark_for

    box = QRectF(10, 10, 40, 40)
    for code in MARKS:
        bounds = mark_for(code, box).boundingRect()
        assert box.adjusted(-1, -1, 1, 1).contains(bounds), (code, bounds)


def test_an_unavailable_provider_is_shown_and_not_chooseable(app):
    from spacr.qt.widgets.provider_marks import ProviderMark

    mark = ProviderMark("claude", "Claude", available=False)
    picked = []
    mark.chosen.connect(picked.append)
    mark.mousePressEvent(_left_click())
    assert picked == [], "an unusable provider accepted a choice"


def test_an_available_provider_emits_its_code(app):
    from spacr.qt.widgets.provider_marks import ProviderMark

    mark = ProviderMark("gemini", "Gemini", available=True)
    picked = []
    mark.chosen.connect(picked.append)
    mark.mousePressEvent(_left_click())
    assert picked == ["gemini"]


def _left_click():
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(4, 4),
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def test_a_brand_colour_is_only_used_when_the_cli_is_here(app):
    """Colour is the fastest answer to "which of these can I use", so an
    absent provider is drawn in the theme's muted ink instead of a greyed
    version of its own colour."""
    from spacr.qt.widgets.provider_marks import BRAND, ProviderMark

    here = ProviderMark("claude", "Claude", available=True)
    gone = ProviderMark("claude", "Claude", available=False)
    assert here._colours()[0].name().lower() == BRAND["claude"].lower()
    assert gone._colours()[0].name().lower() != BRAND["claude"].lower()


# ---------------------------------------------------------------------------
# 4. The rim
# ---------------------------------------------------------------------------

def test_the_run_of_rim_is_longer_than_it_was(app):
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    card.resize(600, 440)
    span = card.accent_span(QRectF(card.rect()))
    # A QUARTER OF THE RIM, measured on a card the size the dialog gives it.
    # At 0.09 -- the original 90 px arc -- it read as a dash sitting on one
    # edge rather than as something flowing along it, and 0.15 still did.
    assert span > 0.22, (
        f"the lit run is {span:.3f} of the rim, which is the short one that "
        f"was reported")
    assert span < 0.62, "past this it is a border, not a highlight"


def test_the_rim_fades_to_nothing_at_both_ends(app):
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    assert card.accent_alpha(0.0) == pytest.approx(0.0, abs=1e-6)
    assert card.accent_alpha(1.0) == pytest.approx(0.0, abs=1e-6)
    assert card.accent_alpha(0.72) > 0.9, "it never reaches full brightness"


def test_the_rim_is_brightest_behind_the_head_not_in_the_middle(app):
    """A highlight that peaks in the centre reads as a bar; one that peaks
    at the front reads as something moving."""
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    assert card.accent_alpha(0.72) > card.accent_alpha(0.5)


def test_the_alpha_ramp_is_monotonic_on_each_side(app):
    """A ramp that wobbles would show as banding along the rim."""
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    rising = [card.accent_alpha(i / 40.0 * 0.72) for i in range(41)]
    falling = [card.accent_alpha(0.72 + i / 40.0 * 0.28) for i in range(41)]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)


def test_the_accent_chases_harder_than_it_did(app):
    from spacr.qt.widgets.setup_card import SetupCard

    assert SetupCard.EASE > 0.18, "this is the value that was reported slow"
    assert SetupCard.EASE < 1.0, (
        "at 1.0 the head is under the pointer with no travel, and the "
        "travel is the whole effect")


def test_it_still_arrives_rather_than_overshooting(app):
    """More responsive must not mean it rings around the target."""
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    card.resize(600, 440)
    card._towards = 0.4
    seen = []
    for _ in range(60):
        card._tick()
        seen.append(card.position)
    assert seen[-1] == pytest.approx(0.4, abs=0.01), seen[-1]
    assert max(seen) <= 0.4 + 1e-6, "it overshot the pointer"


def test_a_circuit_still_ends_where_it_started(app):
    """The lap is the signal that a slide changed; one that drifts leaves
    the accent somewhere the pointer never put it."""
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    card.resize(600, 440)
    start = card.position
    card.circuit(clockwise=True)
    for _ in range(400):
        card._tick()
        if not card.spinning:
            break
    assert not card.spinning
    assert card.position == pytest.approx(start, abs=0.005)


# ---------------------------------------------------------------------------
# 5. The transitions
# ---------------------------------------------------------------------------

def test_a_slide_change_cross_fades(slides):
    """`setCurrentIndex` swaps the page between two frames, so the contents
    changed instantly under a rim that took half a second to travel -- two
    speeds in one gesture."""
    slides.next()
    slides._finish_the_greeting()
    assert slides._fade is not None, "the page was cut, not faded"
    assert slides._pages.graphicsEffect() is not None


def test_the_fade_is_held_or_it_never_runs(slides):
    """A QPropertyAnimation with no owner is collected the moment the method
    returns, and the fade never happens -- which looks exactly like not
    having written one."""
    slides.next()
    slides._finish_the_greeting()
    animation = slides._fade
    assert animation is not None
    assert animation.parent() is slides


def test_the_effect_comes_off_when_the_fade_ends(slides):
    """A QGraphicsOpacityEffect renders its widget into an offscreen pixmap
    on every repaint; six of them left alive over a drifting backdrop is a
    cost paid per frame for an animation that has ended."""
    slides.next()
    slides._finish_the_greeting()
    slides._drop_the_fade()
    assert slides._fade is None
    assert slides._pages.graphicsEffect() is None


def test_the_fade_is_seen_but_not_waited_for(slides):
    from spacr.qt.widgets.setup_slides import FADE_MS

    assert 120 <= FADE_MS <= 500, FADE_MS


def test_the_slides_work_with_no_animation_at_all(slides, monkeypatch):
    """INVARIANTS 10. Every answer is the same on a machine where none of
    the decoration can be drawn."""
    monkeypatch.setattr(slides, "_fade_in",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    try:
        slides._show_slide(2, fade=False)
    except RuntimeError:
        pytest.fail("a slide change depended on the animation")
    assert slides.slide() == 2
    assert set(slides.answers()) >= {"language", "theme"}
