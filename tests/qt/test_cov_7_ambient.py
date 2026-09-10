"""The ambient backdrop's controls, caches and refusals.

This paints behind every module screen while a pipeline runs, so nothing in it
may raise and nothing in it may grow without bound. The tests below drive the
parts that only a user turning a Preferences knob reaches: every setter that
drops a cache, the two caches that reset instead of growing, the blur that
resizes the buffer, and the reads that have to survive a theme module or a
preferences file that cannot be loaded.
"""

from __future__ import annotations

import math

import pytest

from PySide6.QtGui import QColor, QImage, QPainter

from spacr.qt.widgets import ambient as amb
from spacr.qt.widgets.ambient import (AURORA_PULSE_CACHE, AURORA_TILE_CACHE,
                                      AmbientWidget, drift_direction_label,
                                      make_engine)

pytestmark = pytest.mark.qt

DARK = "#101418"


def _engine(theme, palette, **kwargs):
    return make_engine(theme, palette, DARK, seed=7, **kwargs)


def _paint(engine, width=320, height=200) -> QImage:
    """Paint one frame exactly as the widget would."""
    image = QImage(width, height, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), QColor(DARK))
    engine.paint(painter, width, height)
    painter.end()
    return image


# ---------------------------------------------------------------------------
# Names and the theme colour
# ---------------------------------------------------------------------------

def test_an_unknown_drift_direction_is_named_in_the_error():
    """A stale preferences value must say what the valid ones are."""
    with pytest.raises(ValueError, match="unknown starfield direction"):
        drift_direction_label("sideways")


def test_the_backdrop_colour_falls_back_when_the_theme_cannot_be_read(
        monkeypatch):
    """A backdrop must never raise on its way to a screen.

    The flat fill under the animation is read from the live theme; if that
    read fails the dark page colour is used, because a screen with no
    backdrop at all is worse than one in the wrong palette.
    """
    from spacr.qt import theme

    def refuse():
        raise RuntimeError("no palette loaded")

    monkeypatch.setattr(theme, "active_page_colour", refuse)
    colour = amb._theme_background()

    assert isinstance(colour, QColor)
    assert colour == QColor(amb.page_colour("dark"))


def test_the_motion_settings_fall_back_when_preferences_cannot_be_read(
        monkeypatch):
    """Unreadable preferences give the shipped animation, not a crash."""
    import builtins

    real_import = builtins.__import__

    def no_preferences(name, *args, **kwargs):
        if "preferences" in name:
            raise ImportError("preferences are unreadable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_preferences)
    try:
        motion = amb.preferred_motion()
    finally:
        monkeypatch.undo()

    assert motion == amb.Motion(amb.DEFAULT_BLUR, amb.DEFAULT_SPEED,
                                amb.DEFAULT_SIZE, amb.DEFAULT_RESOLUTION,
                                amb.DEFAULT_DENSITY,
                                amb.DEFAULT_DRIFT_DIRECTION)


# ---------------------------------------------------------------------------
# The engine setters
# ---------------------------------------------------------------------------

def test_every_setter_takes_its_value_and_only_a_changed_one_costs_anything():
    """A slider that lands back where it was must not drop a cache.

    Each setter clamps, stores, and then tells the engine to drop whatever
    that setting sized. Re-setting the same value returns before the drop,
    which is what keeps a Preferences dialog cheap to drag.
    """
    engine = _engine("blobs", "spacr")

    engine.set_blur(2.0)
    engine.set_resolution(1.5)
    engine.set_size(2.0)
    assert (engine.blur, engine.resolution, engine.size) == (2.0, 1.5, 2.0)

    # Out of range in both directions, and clamped rather than refused.
    engine.set_blur(99.0)
    engine.set_resolution(-1.0)
    engine.set_size(0.0)
    assert engine.blur == amb.BLUR_RANGE[1]
    assert engine.resolution == amb.RESOLUTION_RANGE[0]
    assert engine.size == amb.SIZE_RANGE[0]


def test_a_direction_reaches_the_engine_and_an_unknown_one_does_not():
    """Every engine is told the direction; most have nothing to do with it."""
    engine = _engine("drift", "spacr", direction="up")
    assert engine.direction == "up"

    engine.set_direction("down")
    assert engine.direction == "down"

    engine.set_direction("sideways")
    assert engine.direction == "down"


# ---------------------------------------------------------------------------
# The buffered engines
# ---------------------------------------------------------------------------

def test_a_new_resolution_drops_the_buffer_rather_than_reusing_it():
    """A buffer sized for the old resolution would be painted at the wrong
    scale for one frame, which reads as a flicker on every slider drag."""
    engine = _engine("blobs", "spacr")
    engine.shade(320, 200)
    assert engine._buffer is not None

    engine.set_resolution(0.5)

    assert engine._buffer is None


def test_a_huge_canvas_is_shaded_at_a_coarser_scale_than_the_edge_asks_for():
    """The pixel ceiling, not the edge ratio, is what bounds the cost.

    An edge ratio does not know how big the display is. The aurora's own
    buffer edge at the top of the resolution range satisfies the ratio on a
    4K canvas and still asks for more pixels than the budget allows, so the
    scale is stepped up until the buffer fits.
    """
    engine = _engine("aurora", "borealis")
    engine.set_resolution(amb.RESOLUTION_RANGE[1])
    width, height = 3840, 2400

    from_the_ratio = math.ceil(max(width, height) / engine.resolution_edge())
    scale = engine.buffer_scale(width, height)
    buffer_width, buffer_height = engine.buffer_size(width, height)

    assert scale > from_the_ratio
    assert buffer_width * buffer_height <= amb.BUFFER_MAX_PIXELS


def test_the_blur_setting_shrinks_the_buffer_before_it_is_blitted_up():
    """The blur is one area-averaging pass, expressed as a downscale.

    At blur 0 the picture must be untouched; above it the buffer makes a
    smaller trip through the filter, and the blit carries it back up.
    """
    engine = _engine("blobs", "spacr")
    assert engine.blur_scale(320, 200) == 1.0

    engine.set_blur(3.0)
    assert engine.blur_scale(320, 200) > 1.0

    softened = engine.shade(320, 200)
    assert softened is not None
    assert softened.width() < 320


def test_an_empty_canvas_shades_nothing_and_blits_nothing():
    """A widget can be asked to paint before it has been given a size."""
    engine = _engine("blobs", "spacr")
    assert engine.shade(0, 200) is None
    assert engine.shade(320, 0) is None

    image = QImage(40, 40, QImage.Format_RGB32)
    painter = QPainter(image)
    engine.blit(painter, None, 40, 40)
    engine.blit(painter, QImage(), 40, 40)
    painter.end()

    assert not image.isNull()


# ---------------------------------------------------------------------------
# The aurora
# ---------------------------------------------------------------------------

def test_a_fold_slides_along_the_arc_instead_of_dragging_it_sideways():
    """The one property of this engine worth testing directly.

    Each fold component is a travelling wave in ``u``: at a fixed time it is
    a shape along the arc, and as the clock advances that shape moves along
    the arc while the arc itself stays where it is.
    """
    engine = _engine("aurora", "borealis")
    curtain = engine.curtains[0]

    still = [engine.fold(curtain, u / 20.0, 0.0) for u in range(21)]
    later = [engine.fold(curtain, u / 20.0, 3.0) for u in range(21)]

    assert any(abs(a - b) > 1e-9 for a, b in zip(still, later))
    assert all(math.isfinite(value) for value in still + later)


def test_a_surge_pulse_stays_inside_its_own_brightness_range():
    """The pulse is a brightness in 0..1; anything outside it is a bug.

    It is multiplied into the curtain's alpha, so a value above one would
    saturate the sheet and a negative one would punch a hole in it.
    """
    engine = _engine("aurora", "borealis")
    curtain = engine.curtains[0]

    values = [engine.pulse(curtain, u / 12.0, t / 4.0)
              for u in range(13) for t in range(9)]

    assert all(0.0 <= value <= 1.0 for value in values)
    assert max(values) > min(values)


def test_resizing_the_aurora_drops_both_of_its_pixel_caches():
    """Both caches are keyed on pixel sizes derived from the size setting."""
    engine = _engine("aurora", "borealis")
    _paint(engine)
    assert engine._tiles or engine._surges

    engine.set_size(2.0)

    assert engine._tiles == {}
    assert engine._surges == {}


def test_the_aurora_caches_reset_instead_of_growing_without_bound():
    """A long run of resizes must not accumulate textures forever.

    Each entry is a full image; keeping one per size the window has ever been
    is a leak measured in megabytes, so the cache starts again at its ceiling.
    """
    engine = _engine("aurora", "borealis")
    curtain = engine.curtains[0]
    engine._tiles = {("filler", i): None for i in range(AURORA_TILE_CACHE)}
    engine._surges = {("filler", i): None for i in range(AURORA_PULSE_CACHE)}

    tile = engine._tile(curtain, 1.0, 64, 48)
    surge = engine._surge(curtain, 1.0)

    assert not tile.isNull()
    assert not surge.isNull()
    assert len(engine._tiles) == 1
    assert len(engine._surges) == 1


def test_a_ray_at_full_length_is_not_cut_short(monkeypatch):
    """Only a ray that is breathing inward has anything taken off it."""
    engine = _engine("aurora", "borealis")
    curtain = engine.curtains[0]
    monkeypatch.setattr(
        type(engine), "ray_lengths",
        lambda self, c: tuple(1.0 for _ in amb.AURORA_TILE_RAYS))

    tile = engine._tile(curtain, 1.0, 64, 48)

    assert not tile.isNull()
    assert tile.size().width() == 64


def test_a_curtain_with_too_few_samples_is_skipped(monkeypatch):
    """A sheet needs two columns; one is not a shape that can be filled."""
    engine = _engine("aurora", "borealis")
    monkeypatch.setattr(type(engine), "geometry",
                        lambda self, width, height: ((0.0, 0.0),))

    image = _paint(engine)

    assert not image.isNull()


# ---------------------------------------------------------------------------
# The starfield
# ---------------------------------------------------------------------------

def test_every_starfield_setting_drops_the_pen_cache():
    """The pens are sized by blur, resolution and size all three.

    A pen kept across a change would draw the previous setting's dot for as
    long as that colour and layer stayed on screen.
    """
    engine = _engine("drift", "spacr")
    _paint(engine)
    assert engine._pens

    engine.set_blur(1.5)
    assert engine._pens == {}

    _paint(engine)
    engine.set_resolution(0.5)
    assert engine._pens == {}

    _paint(engine)
    engine.set_size(2.0)
    assert engine._pens == {}


def test_a_halo_is_the_dot_itself_until_the_blur_is_turned_up():
    """The default frame has to stay exactly what it was.

    The starfield has no buffer to soften, so its blur is a second, wider
    and fainter pass around each dot -- and at blur 0 that pass is the same
    size as the dot, which makes it invisible.
    """
    engine = _engine("drift", "spacr")
    assert engine.halo_size(0) == engine.dot_size(0)

    engine.set_blur(3.0)
    assert engine.halo_size(0) > engine.dot_size(0)
    assert engine.halo_size(0) <= amb.DRIFT_HALO_MAX_PX


def test_the_starfield_does_not_dim_itself_as_it_gets_denser():
    """A couple of hundred dots almost never land on each other.

    The buffered themes divide their alpha down because overlapping
    translucent fields pile up; doing that here would simply hide two thirds
    of the stars.
    """
    engine = _engine("drift", "spacr")
    engine.set_density(3.0)

    assert engine.alpha_scale() == 1.0


def test_every_speck_goes_its_own_way_in_the_random_direction():
    """``random`` is a wandering path, not one shared current with a wobble.

    Under ``up`` or ``down`` every particle shares a heading, so the field
    travels; under ``random`` the headings are isotropic and the field
    spreads and mixes instead.
    """
    engine = _engine("drift", "spacr", direction="random")
    engine.set_time(4.0)

    moved = engine.geometry(320, 200)
    engine.set_time(0.0)
    start = engine.geometry(320, 200)

    assert len(moved) == len(start)
    offsets = [(mx - sx, my - sy)
               for (mx, my, _s), (sx, sy, _t) in zip(moved, start)]
    angles = {round(math.atan2(dy, dx), 2) for dx, dy in offsets
              if abs(dx) + abs(dy) > 1e-6}
    assert len(angles) > 4


def test_the_halo_pass_is_painted_only_when_the_blur_is_on():
    """Two passes per frame is the cost of the blur, so blur 0 pays neither."""
    plain = _engine("drift", "spacr")
    plain.set_time(2.0)
    before = _paint(plain)

    blurred = _engine("drift", "spacr")
    blurred.set_blur(3.0)
    blurred.set_time(2.0)
    after = _paint(blurred)

    assert before != after


# ---------------------------------------------------------------------------
# The widget's own reads
# ---------------------------------------------------------------------------

def test_the_widget_reports_back_every_control_it_was_given(qtbot):
    """Preferences reads these to fill its sliders in.

    A getter that did not reflect the setter beside it would show the user a
    control in one position and animate in another.
    """
    widget = AmbientWidget(theme="drift", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.resize(240, 160)

    widget.set_blur(1.25)
    widget.set_resolution(0.75)
    widget.set_density(2.0)
    widget.set_speed(1.5)
    widget.set_size_scale(1.75)
    widget.set_direction("down")

    assert widget.blur() == 1.25
    assert widget.resolution() == 0.75
    assert widget.density() == 2.0
    assert widget.speed() == 1.5
    assert widget.size_scale() == 1.75
    assert widget.direction() == "down"


def test_an_unknown_direction_leaves_the_widget_where_it_was(qtbot):
    """A stale preferences value must not blank the starfield's setting."""
    widget = AmbientWidget(theme="drift", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.set_direction("down")

    widget.set_direction("sideways")

    assert widget.direction() == "down"
