"""The ambient backdrop — themes, palettes, painting and the CPU guarantee.

Everything here is deterministic: engines are seeded and the clock is driven
by ``advance_frame``/``set_time``, so no test ever waits on a real timer.

The tests that matter most are the ones at the bottom of the file. This thing
paints behind every module screen while a pipeline runs, so "the timer is
inactive while the screen is not visible" is a correctness property rather
than a nicety, and it is asserted four ways: the widget hidden, its host
screen hidden, the window minimised, and a paused one that must stay paused
when the screen comes back.
"""
from __future__ import annotations

import gc
import weakref

import pytest

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt.widgets import ambient as amb
from spacr.qt.widgets.ambient import (AMBIENT_THEMES, AmbientWidget,
                                      BUFFER_MAX_EDGE, DEFAULT_PALETTE,
                                      DEFAULT_THEME, PALETTE_SETS,
                                      coerce_palette, default_palette_for,
                                      install_ambient, is_dark_background,
                                      is_valid_palette, is_valid_theme,
                                      make_engine, palette_colors,
                                      palette_label, palette_note,
                                      palettes_for, theme_label, theme_note)

DT = 1.0 / 24.0
DARK = "#101418"
LIGHT = "#f6f7f9"


def render(engine, width=320, height=200, background=DARK) -> QImage:
    """Paint one frame of ``engine`` exactly as the widget would."""
    image = QImage(width, height, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), QColor(background))
    engine.paint(painter, width, height)
    painter.end()
    return image


def pixels(image: QImage, step=5):
    return [image.pixelColor(x, y).rgb()
            for x in range(0, image.width(), step)
            for y in range(0, image.height(), step)]


def make_widget(qtbot, **kwargs) -> AmbientWidget:
    kwargs.setdefault("seed", 1234)
    kwargs.setdefault("background", DARK)
    size = kwargs.pop("_size", (480, 320))
    widget = AmbientWidget(**kwargs)
    qtbot.addWidget(widget)
    widget.resize(*size)
    return widget


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

def test_defaults_are_the_feature_that_was_asked_for():
    assert DEFAULT_THEME == "blobs"
    assert DEFAULT_PALETTE == "spacr"
    assert DEFAULT_THEME in AMBIENT_THEMES
    assert DEFAULT_PALETTE in palettes_for(DEFAULT_THEME)


def test_spacr_palette_is_the_users_own_three_colours():
    """Chosen by the user themselves; see spacr/qt/theme.py STAGE_HOVER."""
    from spacr.qt.theme import STAGE_HOVER

    colours = {c.upper() for c in palette_colors("blobs", "spacr")}
    assert colours == {c.upper() for c in STAGE_HOVER.values()}
    assert colours == {"#3B82F6", "#FF00FF", "#00CEC8"}


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_every_theme_offers_palettes_that_resolve_to_real_colours(theme):
    offered = palettes_for(theme)
    assert isinstance(offered, tuple) and offered, theme
    assert len(set(offered)) == len(offered), "duplicate palette listed"
    for name in offered:
        colours = palette_colors(theme, name)
        assert len(colours) >= 3, (theme, name)
        for hexcode in colours:
            colour = QColor(hexcode)
            assert colour.isValid(), (theme, name, hexcode)
            # A real colour, not a placeholder: valid hex, and it survives
            # the round trip that the engines put it through.
            assert colour.name().lower() == hexcode.lower()


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_every_theme_has_a_label_and_a_note(theme):
    assert theme_label(theme) and theme_label(theme) != theme
    assert theme_note(theme).endswith(".")


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_every_offered_palette_has_a_label_and_a_note(theme):
    for name in palettes_for(theme):
        assert palette_label(theme, name)
        assert palette_note(theme, name).endswith(".")


def test_a_colour_blind_safe_palette_is_offered_by_every_theme():
    """spaCR already has a colour-vision setting; a backdrop that only works
    for trichromats would be out of keeping."""
    for theme in AMBIENT_THEMES:
        assert "okabe" in palettes_for(theme), theme
    note = palette_note("blobs", "okabe").lower()
    # It has to say WHICH deficiency it is safe for, or it is a claim
    # nobody can act on.
    assert "protanopia" in note and "deuteranopia" in note
    assert palette_colors("blobs", "okabe")[0] == "#0072B2"


def test_a_greyscale_palette_is_offered_for_people_who_want_no_colour():
    for theme in AMBIENT_THEMES:
        assert "mono" in palettes_for(theme), theme
    for hexcode in palette_colors("blobs", "mono"):
        colour = QColor(hexcode)
        assert colour.red() == colour.green() == colour.blue(), \
            f"{hexcode} has a colour cast, and the palette is called mono"


def _colours(name):
    return [QColor(c) for c in PALETTE_SETS[name].colors]


def _mean(values):
    return sum(values) / len(values)


def test_the_palettes_have_genuinely_different_character():
    """Warm, cool, pastel and mono must actually be warm, cool, pale, grey —
    otherwise the menu offers five names for one look.

    Hues are checked per colour rather than averaged: warm hues straddle the
    wrap point at 0 (orange 0.05, crimson 0.98), and their mean lands on
    cyan, which is exactly wrong.
    """
    for colour in _colours("ember"):
        hue = colour.hueF()
        assert hue <= 0.14 or hue >= 0.88, f"{colour.name()} is not warm"
    for colour in _colours("ocean"):
        assert 0.45 <= colour.hueF() <= 0.68, f"{colour.name()} is not cool"

    assert _mean([c.saturationF() for c in _colours("mono")]) < 0.02
    assert _mean([c.saturationF() for c in _colours("ember")]) > 0.6
    # Pastel is pale and unsaturated relative to the vivid sets.
    assert _mean([c.lightnessF() for c in _colours("pastel")]) > 0.75
    assert _mean([c.saturationF() for c in _colours("pastel")]) \
        < _mean([c.saturationF() for c in _colours("spacr")])


def test_pastel_is_withheld_from_the_themes_it_would_be_invisible_in():
    """A 1-3 px dot or a thin ring in a pale low-contrast hue is not a
    setting, it is a no-op. Offering it would be dishonest."""
    assert "pastel" in palettes_for("blobs")
    assert "pastel" in palettes_for("aurora")
    assert "pastel" not in palettes_for("drift")
    assert "pastel" not in palettes_for("ripple")


# ---------------------------------------------------------------------------
# Validation — unknown names raise, they do not quietly become the default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["lattice", "BLOBS", "", None, 3])
def test_unknown_theme_raises_everywhere_it_can_be_named(bad):
    with pytest.raises(ValueError):
        theme_label(bad)
    with pytest.raises(ValueError):
        theme_note(bad)
    with pytest.raises(ValueError):
        palettes_for(bad)
    with pytest.raises(ValueError):
        make_engine(bad, "spacr", DARK)
    assert not is_valid_theme(bad)


@pytest.mark.parametrize("bad", ["neon", "SPACR", "", None])
def test_unknown_palette_raises_everywhere_it_can_be_named(bad):
    with pytest.raises(ValueError):
        palette_label("blobs", bad)
    with pytest.raises(ValueError):
        palette_note("blobs", bad)
    with pytest.raises(ValueError):
        palette_colors("blobs", bad)
    with pytest.raises(ValueError):
        make_engine("blobs", bad, DARK)
    assert not is_valid_palette("blobs", bad)


def test_unknown_theme_or_palette_raises_from_the_widget(qtbot):
    with pytest.raises(ValueError):
        AmbientWidget(theme="lattice")
    with pytest.raises(ValueError):
        AmbientWidget(palette="neon")
    widget = make_widget(qtbot)
    with pytest.raises(ValueError):
        widget.set_theme("lattice")
    with pytest.raises(ValueError):
        widget.set_palette("neon")
    # And the widget is unharmed by the refusal.
    assert widget.theme() == DEFAULT_THEME
    assert widget.palette_name() == DEFAULT_PALETTE


def test_a_real_palette_the_theme_does_not_offer_is_still_refused(qtbot):
    """'pastel' exists, but not for drift. An explicit request for it is a
    mistake worth hearing about, not something to silently substitute."""
    widget = make_widget(qtbot, theme="drift", palette="ocean")
    with pytest.raises(ValueError) as excinfo:
        widget.set_palette("pastel")
    assert "unknown" not in str(excinfo.value), \
        "pastel is a real palette; the message must not call it unknown"
    assert widget.palette_name() == "ocean"


def test_error_messages_name_the_valid_options():
    with pytest.raises(ValueError) as excinfo:
        palettes_for("lattice")
    for name in AMBIENT_THEMES:
        assert name in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        palette_label("drift", "pastel")
    for name in palettes_for("drift"):
        assert name in str(excinfo.value)


def test_stored_palette_from_another_theme_is_downgraded_not_fatal(qtbot):
    """A preferences file naming a palette the user picked under a different
    theme must not stop a screen from being built."""
    assert coerce_palette("drift", "pastel") == default_palette_for("drift")
    assert coerce_palette("blobs", "pastel") == "pastel"
    with pytest.raises(ValueError):
        coerce_palette("blobs", "neon")

    widget = make_widget(qtbot, theme="drift", palette="pastel")
    assert widget.palette_name() == default_palette_for("drift")


def test_switching_to_a_theme_without_the_current_palette_downgrades(qtbot):
    widget = make_widget(qtbot, theme="blobs", palette="pastel")
    assert widget.palette_name() == "pastel"
    widget.set_theme("drift")
    assert widget.theme() == "drift"
    assert widget.palette_name() == default_palette_for("drift")
    assert is_valid_palette("drift", widget.palette_name())


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_default_palette_is_always_one_the_theme_offers(theme):
    assert default_palette_for(theme) in palettes_for(theme)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_same_seed_gives_the_same_first_frame(theme):
    a = make_engine(theme, "spacr", DARK, seed=99)
    b = make_engine(theme, "spacr", DARK, seed=99)
    assert render(a) == render(b), theme
    assert a.geometry(320, 200) == b.geometry(320, 200)


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_different_seeds_give_different_frames(theme):
    a = make_engine(theme, "spacr", DARK, seed=1)
    b = make_engine(theme, "spacr", DARK, seed=2)
    assert a.geometry(320, 200) != b.geometry(320, 200), theme


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_the_clock_alone_decides_the_frame(theme):
    """Stepping in twelve small steps and jumping straight to the same time
    must land on the same frame — that is what makes ``set_time`` legitimate
    and what stops the animation drifting after a stall."""
    stepped = make_engine(theme, "ocean", DARK, seed=5)
    for _ in range(12):
        stepped.advance(0.25)
    jumped = make_engine(theme, "ocean", DARK, seed=5)
    jumped.set_time(3.0)
    assert stepped.time == pytest.approx(3.0)
    flat = [v for item in stepped.geometry(320, 200) for v in item]
    assert flat == pytest.approx(
        [v for item in jumped.geometry(320, 200) for v in item])
    assert render(stepped) == render(jumped)


def test_negative_and_zero_steps_do_not_move_the_clock_backwards():
    engine = make_engine("blobs", "spacr", DARK, seed=5)
    engine.advance(1.0)
    engine.advance(-5.0)
    engine.advance(0.0)
    assert engine.time == pytest.approx(1.0)
    assert engine.frames == 3


# ---------------------------------------------------------------------------
# The frame actually paints, and it actually changes
# ---------------------------------------------------------------------------

#: Minimum share of a 320x200 frame each theme must actually paint, and
#: minimum share that must differ between two animation times.
#:
#: One threshold cannot serve both kinds of theme. The soft fields cover
#: 87-95 % of the page; a starfield lights a few hundred pixels out of 64 000
#: by design, and demanding 15 % of it would be demanding that it stop being
#: a starfield. The numbers below are half of what each theme measures, so
#: they catch "it stopped painting" without pinning the artwork down.
MIN_PAINTED = {"blobs": 0.40, "aurora": 0.40, "ripple": 0.40, "drift": 0.003}
MIN_CHANGED = {"blobs": 0.40, "aurora": 0.40, "ripple": 0.40, "drift": 0.006}


def all_pixels(image: QImage):
    return [image.pixelColor(x, y).rgb()
            for x in range(image.width())
            for y in range(image.height())]


@pytest.mark.parametrize("background", [DARK, LIGHT])
@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_a_frame_is_not_just_the_background(theme, background):
    """Both pages, because a blob set tuned for dark reads as nothing at all
    on light — which is a bug you cannot see in a construction test."""
    engine = make_engine(theme, "spacr", background, seed=7)
    engine.set_time(11.0)
    every = all_pixels(render(engine, background=background))
    flat = QColor(background).rgb()
    painted = sum(1 for p in every if p != flat)
    assert painted > len(every) * MIN_PAINTED[theme], \
        f"{theme} on {background} painted only {painted} of {len(every)} px"


@pytest.mark.parametrize("background", [DARK, LIGHT])
@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_the_frame_changes_between_two_animation_times(theme, background):
    engine = make_engine(theme, "spacr", background, seed=7)
    engine.set_time(2.0)
    first = all_pixels(render(engine, background=background))
    engine.set_time(9.0)
    second = all_pixels(render(engine, background=background))
    differing = sum(1 for a, b in zip(first, second) if a != b)
    assert differing > len(first) * MIN_CHANGED[theme], \
        f"{theme} moved only {differing} of {len(first)} px in seven seconds"


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_the_palette_reaches_the_painted_pixels(theme):
    """Two palettes, same seed, same clock: only the colours differ, so the
    frames must differ. This is what catches a palette that is accepted,
    stored and then never used."""
    a = make_engine(theme, "spacr", DARK, seed=3)
    b = make_engine(theme, "ember", DARK, seed=3)
    a.set_time(6.0)
    b.set_time(6.0)
    assert a.geometry(320, 200) == b.geometry(320, 200), \
        "the palette must not move anything"
    assert render(a) != render(b), theme


def test_setting_the_palette_live_recolours_without_moving_anything(qtbot):
    engine = make_engine("blobs", "spacr", DARK, seed=3)
    engine.set_time(6.0)
    before = engine.geometry(320, 200)
    first = render(engine)
    engine.set_colors(palette_colors("blobs", "ember"))
    assert engine.geometry(320, 200) == before
    assert render(engine) != first


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_dark_and_light_pages_get_different_treatments(theme):
    """Additive on dark, multiply on light. If both used the same mode one of
    the two would be invisible, and the shipped app has both."""
    dark = make_engine(theme, "spacr", DARK, seed=4)
    light = make_engine(theme, "spacr", LIGHT, seed=4)
    assert dark.dark and not light.dark
    assert dark.mode == QPainter.CompositionMode_Plus
    assert light.mode == QPainter.CompositionMode_Multiply

    # Additive never darkens the page; multiplicative never brightens it.
    dark.set_time(11.0)
    light.set_time(11.0)
    base_dark, base_light = QColor(DARK), QColor(LIGHT)
    for pixel in pixels(render(dark, background=DARK)):
        assert QColor(pixel).lightnessF() >= base_dark.lightnessF() - 0.02
    for pixel in pixels(render(light, background=LIGHT)):
        assert QColor(pixel).lightnessF() <= base_light.lightnessF() + 0.02


def test_the_composition_mode_follows_a_background_change():
    engine = make_engine("blobs", "spacr", DARK, seed=4)
    assert engine.dark
    engine.set_background(LIGHT)
    assert not engine.dark
    assert engine.mode == QPainter.CompositionMode_Multiply
    assert engine.identity == QColor(255, 255, 255)


@pytest.mark.parametrize("colour,expected", [
    ("#000000", True), ("#101418", True), ("#1a2b3c", True),
    ("#ffffff", False), ("#f6f7f9", False), ("#cccccc", False),
])
def test_dark_background_detection(colour, expected):
    assert is_dark_background(colour) is expected


def test_blobs_vary_in_size_and_pulse_over_time():
    """'big and small and changing size' — the user's words, asserted."""
    engine = make_engine("blobs", "spacr", DARK, seed=21)
    radii = [r for _, _, r in engine.geometry(1000, 800)]
    assert max(radii) > 3 * min(radii), "no size variety"

    # Every blob's radius must actually change over its own pulse cycle.
    later = [r for _, _, r in engine.geometry(1000, 800)]
    assert later == radii
    engine.set_time(5.0)
    grown = [r for _, _, r in engine.geometry(1000, 800)]
    assert all(abs(a - b) > 1e-6 for a, b in zip(radii, grown))
    # ... and the field must move, not just breathe.
    start = [(x, y) for x, y, _ in engine.geometry(1000, 800)]
    engine.set_time(40.0)
    moved = [(x, y) for x, y, _ in engine.geometry(1000, 800)]
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1e-6
               for a, b in zip(start, moved))


def test_blobs_spread_over_the_whole_canvas():
    """A field that clumps in one corner is not a backdrop."""
    engine = make_engine("blobs", "spacr", DARK, seed=21)
    centres = engine.geometry(1000, 800)
    assert min(x for x, _, _ in centres) < 300
    assert max(x for x, _, _ in centres) > 700
    assert min(y for _, y, _ in centres) < 250
    assert max(y for _, y, _ in centres) > 550


def test_every_palette_colour_gets_used_by_the_blob_field():
    engine = make_engine("blobs", "okabe", DARK, seed=21)
    used = {blob.color % len(engine.paint_colors) for blob in engine.blobs}
    assert used == set(range(len(engine.paint_colors)))


def test_drift_thins_out_on_a_small_canvas_and_fills_a_big_one():
    engine = make_engine("drift", "ocean", DARK, seed=8)
    small = engine.geometry(400, 300)
    big = engine.geometry(1920, 1080)
    assert len(small) < len(big), "a laptop screen is not a wall display"
    assert len(small) == amb.DRIFT_MIN_PARTICLES, "a small screen went bare"
    # The pool is a ceiling, and an enormous canvas reaches it.
    assert len(engine.geometry(6000, 4000)) == amb.DRIFT_POOL
    # The pool itself never changes, so a resize re-frames the field rather
    # than re-rolling it: going small and back gives the same stars back.
    assert engine.geometry(1920, 1080) == big


def test_ripple_rings_fade_in_and_out_rather_than_popping():
    engine = make_engine("ripple", "ocean", DARK, seed=8)
    for t in (0.0, 3.0, 7.0, 13.0):
        engine.set_time(t)
        for _, _, radius, fade in engine.geometry(800, 600):
            assert 0.0 <= fade <= 1.0
            # The youngest, smallest rings are the faintest.
            assert radius > 0


def test_aurora_bands_cross_fade_between_two_palette_colours():
    engine = make_engine("aurora", "spacr", DARK, seed=8)
    band = engine.bands[0]
    seen = set()
    for t in range(0, 60, 3):
        engine.set_time(float(t))
        seen.add(engine.band_color(band).name())
    assert len(seen) > 10, "the hue is not shifting"


# ---------------------------------------------------------------------------
# The buffer
# ---------------------------------------------------------------------------

def test_the_buffer_is_small_and_reused_across_frames():
    """Never allocate a QImage per frame. The whole cost argument rests on
    this, so assert on the object identity rather than on a timing."""
    engine = make_engine("blobs", "spacr", DARK, seed=2)
    render(engine, 1920, 1080)
    buffer = engine._buffer
    assert buffer is not None
    assert max(buffer.width(), buffer.height()) <= BUFFER_MAX_EDGE
    for _ in range(20):
        engine.advance(DT)
        render(engine, 1920, 1080)
    assert engine._buffer is buffer, "reallocated the buffer mid-animation"


def test_the_buffer_is_reallocated_only_when_the_canvas_changes():
    engine = make_engine("aurora", "spacr", DARK, seed=2)
    render(engine, 800, 600)
    first = engine._buffer
    render(engine, 1600, 900)
    assert engine._buffer is not first
    second = engine._buffer
    render(engine, 1600, 900)
    assert engine._buffer is second


@pytest.mark.parametrize("size", [(1, 1), (3, 400), (1920, 1080), (0, 0)])
@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_absurd_canvases_still_paint_without_raising(theme, size):
    engine = make_engine(theme, "spacr", DARK, seed=2)
    engine.set_time(9.0)
    width, height = size
    image = QImage(max(1, width), max(1, height), QImage.Format_RGB32)
    painter = QPainter(image)
    engine.paint(painter, width, height)
    painter.end()


def test_a_half_written_engine_fails_loudly_rather_than_painting_nothing():
    """The contract for whoever adds the fifth theme: an engine that does not
    implement its two methods must say so, not silently paint an empty
    frame that looks like the animation is switched off."""
    bare = amb.AmbientEngine(["#00CEC8"], DARK, seed=1)
    with pytest.raises(NotImplementedError):
        bare.paint(QPainter(), 100, 100)
    with pytest.raises(NotImplementedError):
        bare.geometry(100, 100)

    class Incomplete(amb._BufferedEngine):
        pass

    engine = Incomplete(["#00CEC8"], DARK, seed=1)
    image = QImage(64, 64, QImage.Format_RGB32)
    painter = QPainter(image)
    with pytest.raises(NotImplementedError):
        engine.paint(painter, 64, 64)
    painter.end()


def test_an_empty_or_invalid_palette_falls_back_to_a_paintable_grey():
    """Colours reach the engine from a palette table, but the engine is the
    thing that paints — it must not divide by zero on an empty list."""
    engine = amb.BlobsEngine([], DARK, seed=1)
    assert engine.colors and engine.colors[0].isValid()
    engine = amb.BlobsEngine(["not-a-colour", "#00CEC8"], DARK, seed=1)
    assert [c.name() for c in engine.colors] == ["#00cec8"]
    assert render(engine) is not None


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

def test_widget_takes_no_focus_and_no_mouse(qtbot):
    widget = make_widget(qtbot)
    assert widget.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert widget.focusPolicy() == Qt.NoFocus
    widget.show()
    qtbot.waitExposed(widget)
    widget.setFocus(Qt.OtherFocusReason)
    assert not widget.hasFocus()


def test_install_lowers_it_behind_every_sibling(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    content = QWidget()
    layout.addWidget(content)
    host.resize(600, 400)

    widget = install_ambient(host, layout, theme="blobs", palette="spacr",
                             seed=1)
    assert widget.parent() is host
    assert widget.geometry() == host.rect()
    children = [c for c in host.children() if isinstance(c, QWidget)]
    assert children[0] is widget, "must be at the bottom of the stack"
    # The installer takes a layout for signature parity with the DNA rain,
    # but the backdrop is an overlay and must never be laid out.
    assert layout.indexOf(widget) == -1
    assert layout.count() == 1


def test_install_follows_the_hosts_size(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(600, 400)
    widget = install_ambient(host, theme="aurora", palette="ocean", seed=1)
    assert widget.geometry() == host.rect()
    host.show()
    qtbot.waitExposed(host)
    host.resize(900, 500)
    qtbot.waitUntil(lambda: widget.geometry() == host.rect(), timeout=2000)
    # ... and the engine follows it into the new canvas.
    assert len(widget.engine.geometry(*host.rect().size().toTuple())) > 0


def test_widget_paints_the_background_and_the_animation(qtbot):
    widget = make_widget(qtbot, _size=(320, 240), background="#000000")
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_time(11.0)
    image = widget.grab().toImage()
    assert image.width() == widget.width()
    assert any(QColor(p).lightness() > 8 for p in pixels(image)), \
        "nothing was painted over the background"


def test_widget_frames_differ_over_time(qtbot):
    widget = make_widget(qtbot, _size=(320, 240))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_time(1.0)
    first = widget.grab().toImage()
    for _ in range(60):
        widget.advance_frame(DT)
    assert widget.time() == pytest.approx(1.0 + 60 * DT)
    assert widget.grab().toImage() != first


def test_background_setter_reaches_the_engine_and_is_forced_opaque(qtbot):
    widget = make_widget(qtbot)
    widget.set_background_color(QColor(250, 250, 250, 5))
    assert widget.background_color().alpha() == 255
    assert not widget.engine.dark
    widget.set_background_color("not-a-colour")
    assert widget.background_color() == QColor(250, 250, 250)


def test_a_backdrop_image_shows_through_the_animation(qtbot):
    """The Space and Cell themes have a wallpaper. An opaque backdrop widget
    would hide it on every screen at once, which is the bug this composites
    its way around."""
    wallpaper = QPixmap(400, 300)
    wallpaper.fill(QColor("#204060"))
    widget = make_widget(qtbot, _size=(400, 300), background="#000000")
    widget.set_backdrop(wallpaper)
    assert widget.backdrop() is not None
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_time(11.0)
    image = widget.grab().toImage()
    # Additive over the picture: every pixel is at least the wallpaper.
    sampled = [QColor(p) for p in pixels(image)]
    assert all(c.blue() >= 0x5e for c in sampled), "the wallpaper was buried"
    assert any(c.blue() > 0x62 or c.green() > 0x42 for c in sampled), \
        "the animation did not reach the picture"

    widget.set_backdrop(None)
    assert widget.backdrop() is None


def test_a_missing_backdrop_is_cosmetic_not_fatal(qtbot):
    widget = make_widget(qtbot)
    widget.set_backdrop("/no/such/wallpaper.png")
    assert widget.backdrop() is None
    widget.set_backdrop(QImage(8, 8, QImage.Format_RGB32))
    assert widget.backdrop() is not None
    widget.set_backdrop(QImage())          # null image, not a picture
    assert widget.backdrop() is None


def test_a_backdrop_that_blows_up_on_inspection_is_still_not_fatal(qtbot):
    """The wallpaper comes from the theme layer, which is free to hand over
    something lazier than a path. Losing the picture is a cosmetic miss; a
    traceback on every module screen is not."""
    class Exploding:
        def __str__(self):
            raise RuntimeError("the wallpaper went away")

    widget = make_widget(qtbot)
    widget.set_backdrop(Exploding())
    assert widget.backdrop() is None
    widget.show()
    qtbot.waitExposed(widget)
    widget.advance_frame(DT)


def test_a_widget_the_wallpaper_does_not_reach_gets_the_flat_colour(qtbot):
    """The QSS wallpaper is centred and does not repeat, so a widget out in
    the corner of a big window can fall entirely outside it. It still has to
    paint every one of its pixels — ``WA_OpaquePaintEvent`` promises Qt there
    is nothing underneath worth keeping."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(800, 600)
    widget = AmbientWidget(host, background=DARK, seed=1)
    widget.setGeometry(620, 430, 160, 140)
    host.show()
    qtbot.waitExposed(host)
    small = QPixmap(40, 30)
    small.fill(QColor("#204060"))
    widget.set_backdrop(small)

    image = QImage(160, 140, QImage.Format_RGB32)
    image.fill(QColor("#ff00ff"))
    painter = QPainter(image)
    widget._paint_base(painter, widget.rect())
    painter.end()
    assert set(all_pixels(image)) == {QColor(DARK).rgb()}


def test_a_parentless_backdrop_can_still_be_installed(qtbot):
    widget = make_widget(qtbot)
    assert widget.parent() is None
    widget.follow_parent()          # nothing to follow, must not raise
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_running()


def test_backdrop_is_centred_on_the_window_like_the_stylesheet(qtbot):
    """The QSS paints the wallpaper centred on the window and does not
    repeat; land it anywhere else and the picture jumps at this widget's
    edge."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(800, 600)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(40, 30, 40, 30)
    widget = AmbientWidget(host, background=DARK, seed=1)
    layout.addWidget(widget)
    host.show()
    qtbot.waitExposed(host)
    wallpaper = QPixmap(400, 300)
    wallpaper.fill(QColor("#204060"))
    widget.set_backdrop(wallpaper)
    origin = widget._backdrop_origin()
    offset = widget.mapTo(host, QPoint(0, 0))
    assert origin.x() + offset.x() == (800 - 400) // 2
    assert origin.y() + offset.y() == (600 - 300) // 2


# ---------------------------------------------------------------------------
# Live switching — no leaks, no orphaned timers
# ---------------------------------------------------------------------------

def test_switching_theme_swaps_the_engine_and_keeps_the_clock(qtbot):
    widget = make_widget(qtbot)
    widget.advance_frame(2.0)
    first = widget.engine
    assert first.name == "blobs"
    widget.set_theme("aurora")
    assert widget.engine is not first
    assert widget.engine.name == "aurora"
    assert widget.engine.time == pytest.approx(first.time)
    assert widget.engine.background == widget.background_color()


def test_switching_theme_does_not_leak_the_old_engine(qtbot):
    widget = make_widget(qtbot)
    dead = weakref.ref(widget.engine)
    widget.set_theme("ripple")
    gc.collect()
    assert dead() is None, "the old engine outlived the switch"


def test_switching_theme_never_creates_a_second_timer(qtbot):
    """A second timer is the exact shape of the performance complaint this
    feature could become: two engines painting one widget forever."""
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    assert len(widget.findChildren(QTimer)) == 1
    timer = widget.findChildren(QTimer)[0]

    for theme in AMBIENT_THEMES * 3:
        widget.set_theme(theme)
        for name in palettes_for(theme):
            widget.set_palette(name)
        # findChildren is recursive, so this counts every timer anywhere
        # under the widget, not just its direct children.
        assert len(widget.findChildren(QTimer)) == 1
        assert widget.findChildren(QTimer)[0] is timer
        assert widget.is_running()
    assert widget._timer is timer


def test_setting_the_same_theme_or_palette_twice_is_a_no_op(qtbot):
    widget = make_widget(qtbot)
    engine = widget.engine
    widget.set_theme(widget.theme())
    assert widget.engine is engine
    widget.set_palette(widget.palette_name())
    assert widget.engine is engine


def test_palette_switch_keeps_the_engine_and_the_motion(qtbot):
    widget = make_widget(qtbot, theme="blobs", palette="spacr")
    widget.advance_frame(3.0)
    engine = widget.engine
    before = engine.geometry(400, 300)
    widget.set_palette("ember")
    assert widget.engine is engine, "recolouring must not restart the motion"
    assert engine.geometry(400, 300) == before
    assert [c.name() for c in engine.colors] == \
        [c.lower() for c in palette_colors("blobs", "ember")]


def test_theme_switch_reaches_the_painted_output(qtbot):
    widget = make_widget(qtbot, _size=(320, 240))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_time(9.0)
    blobs = widget.grab().toImage()
    widget.set_theme("drift")
    assert widget.grab().toImage() != blobs


# ---------------------------------------------------------------------------
# Live theme switch
# ---------------------------------------------------------------------------

def test_it_follows_the_app_palette_when_the_host_did_not_pick_a_colour(qtbot,
                                                                       qapp):
    """Without this, a dark-to-light switch leaves a black rectangle on a
    white page — exactly the bug the DNA rain had."""
    widget = AmbientWidget(seed=1)
    qtbot.addWidget(widget)
    widget.set_background_color("#000000")
    widget._background_explicit = False   # as if it had never been set
    from spacr.qt.theme import palette_for
    import spacr.qt.widgets.ambient as module
    original = module._theme_background
    module._theme_background = lambda: QColor(palette_for("light")["bg"])
    try:
        widget.changeEvent(QEvent(QEvent.ApplicationPaletteChange))
    finally:
        module._theme_background = original
    assert not widget.engine.dark
    assert widget.background_color() == QColor(palette_for("light")["bg"])


def test_a_host_that_chose_the_colour_keeps_it(qtbot):
    """app_screen re-applies the colour itself (it also has to re-resolve the
    wallpaper), so this widget must not fight it."""
    widget = make_widget(qtbot, background=DARK)
    widget.changeEvent(QEvent(QEvent.ApplicationPaletteChange))
    assert widget.background_color() == QColor(DARK)


def test_other_change_events_are_ignored(qtbot):
    widget = AmbientWidget(seed=1)
    qtbot.addWidget(widget)
    widget._background_explicit = False
    before = widget.background_color()
    widget.changeEvent(QEvent(QEvent.FontChange))
    assert widget.background_color() == before


def test_theme_background_falls_back_when_preferences_are_unreadable(
        monkeypatch):
    from spacr.qt import preferences
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    from spacr.qt.theme import palette_for
    assert amb._theme_background() == QColor(palette_for("dark")["bg"])


# ---------------------------------------------------------------------------
# The CPU guarantee — the reason this feature is allowed to exist
# ---------------------------------------------------------------------------

def test_timer_stops_when_hidden_and_restarts_when_shown(qtbot):
    widget = make_widget(qtbot)
    assert not widget.is_running(), "must not tick before it is on screen"
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_running()

    frames = widget.engine.frames
    widget.hide()
    assert not widget.is_running(), "hidden widget must not burn CPU"
    qtbot.wait(60)
    assert widget.engine.frames == frames, \
        "it kept asking for frames while hidden"

    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_running()


def test_the_timer_stops_when_the_host_screen_is_hidden(qtbot):
    """The real case: the user starts a pipeline and switches to another
    module. This screen stays constructed and must go quiet."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(600, 400)
    widget = install_ambient(host, theme="blobs", palette="spacr", seed=1)
    assert not widget.is_running()
    host.show()
    qtbot.waitExposed(host)
    assert widget.is_running()
    host.hide()
    assert not widget.is_running()


def test_timer_stops_when_the_window_is_minimised(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    widget = AmbientWidget(window, seed=5, background=DARK)
    layout.addWidget(widget)
    window.resize(400, 300)
    window.show()
    qtbot.waitExposed(window)
    assert widget.is_running()

    window.setWindowState(Qt.WindowMinimized)
    widget.eventFilter(window, QEvent(QEvent.WindowStateChange))
    assert not widget.is_running()

    window.setWindowState(Qt.WindowNoState)
    widget.eventFilter(window, QEvent(QEvent.WindowStateChange))
    assert widget.is_running()


def test_set_animating_pauses_without_destroying_anything(qtbot):
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_animating() and widget.is_running()
    engine = widget.engine
    time = widget.time()

    widget.set_animating(False)
    assert not widget.is_running()
    assert not widget.is_animating()
    assert widget.engine is engine, "pause must not destroy the engine"
    assert widget.time() == time

    widget.set_animating(True)
    assert widget.is_running()
    assert widget.engine is engine


def test_a_paused_widget_stays_paused_across_hide_and_show(qtbot):
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_animating(False)
    widget.hide()
    widget.show()
    qtbot.waitExposed(widget)
    assert not widget.is_running(), "showing it must not un-pause it"
    widget.set_animating(True)
    assert widget.is_running()


def test_set_animating_is_idempotent(qtbot):
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_animating(True)
    assert widget.is_running()
    widget.set_animating(False)
    widget.set_animating(False)
    assert not widget.is_running()


def test_start_and_stop_are_idempotent(qtbot):
    widget = make_widget(qtbot)
    widget.start()
    assert widget.is_running()
    widget.start()
    assert widget.is_running()
    widget.stop()
    assert not widget.is_running()
    widget.stop()
    assert not widget.is_running()


def test_fps_is_capped_and_becomes_a_real_interval(qtbot):
    widget = make_widget(qtbot)
    assert widget.fps() == amb.DEFAULT_FPS
    widget.set_fps(10_000)
    assert widget.fps() == amb.MAX_FPS
    widget.set_fps(-4)
    assert widget.fps() == amb.MIN_FPS
    assert widget._timer.interval() >= 1
    widget.set_fps(24)
    assert widget._timer.interval() == 41


def test_the_tick_clamps_a_long_stall(qtbot):
    """Two seconds of GPU contention must not teleport the animation."""
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    widget.stop()
    widget._clock.restart()
    qtbot.wait(5)
    widget.set_time(0.0)
    widget._on_tick()
    assert widget.time() <= amb.MAX_DT


def test_reparenting_moves_the_window_watch(qtbot):
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget._watched is widget

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    layout.addWidget(widget)
    host.resize(400, 300)
    host.show()
    qtbot.waitExposed(host)
    assert widget._watched is host
    assert widget.is_running()

    host.setWindowState(Qt.WindowMinimized)
    widget.eventFilter(host, QEvent(QEvent.WindowStateChange))
    assert not widget.is_running()
