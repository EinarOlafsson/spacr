"""The space flight's CPU maths, run as plain Python rather than as a jit.

Numba compiles ``sample_space`` and its neighbours to machine code, and a
compiled kernel executes no Python lines -- so the star field, the object
slots and the supersampler are invisible to a line tracer even while the
suite renders frames with them. Numba's own ``DISABLE_JIT`` switch turns the
decorators into no-ops, so the SAME source lines run as interpreted Python
and can be both traced and reasoned about one at a time.

Frames here are TINY. The kernels are interpreted in these tests, so a
backdrop-sized frame would be a wall-clock cost for nothing: every branch --
the hue field, all six parallax layers, all three object slots, the vignette
and the supersampler -- is evaluated per pixel, so a couple of hundred pixels
exercises the lot.

No GPU is touched anywhere in this file: the vispy path is a separate backend
and nothing here constructs a canvas.
"""
from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

pytest.importorskip("numba")

space = pytest.importorskip("spacr.qt.widgets.fractal_space")

SOURCE = space.__file__

#: A small frame that still lands stars: the field is sparse by design.
WIDTH, HEIGHT = 24, 16

#: Sentinel for "numba was not in sys.modules to begin with".
_MISSING = object()


def _execute_source(name: str, *, hide_numba: bool = False,
                    disable_jit: bool = False):
    """Run the shipped module source again under a private module name.

    Loading a second copy leaves ``spacr.qt.widgets.fractal_space`` in
    ``sys.modules`` exactly as it was, so nothing else in the session sees a
    module whose kernels are interpreted or whose numba is missing.
    """
    from numba import config as numba_config

    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    was_disabled = numba_config.DISABLE_JIT
    had_numba = sys.modules.get("numba", _MISSING)
    if disable_jit:
        numba_config.DISABLE_JIT = True
    if hide_numba:
        # ``None`` in sys.modules is the documented way to make an import of
        # that name fail, which is what a machine without numba looks like.
        sys.modules["numba"] = None
    try:
        spec.loader.exec_module(module)
    finally:
        numba_config.DISABLE_JIT = was_disabled
        if hide_numba:
            if had_numba is _MISSING:
                sys.modules.pop("numba", None)
            else:
                sys.modules["numba"] = had_numba
    return module


@pytest.fixture(scope="module")
def plain():
    """The module's own kernels, interpreted rather than compiled."""
    return _execute_source("_fractal_space_plain", disable_jit=True)


@pytest.fixture(scope="module")
def no_numba():
    """The module as it loads on a machine that has no numba at all."""
    return _execute_source("_fractal_space_no_numba", hide_numba=True)


# --------------------------------------------------------------------------
# The hash the whole field is built on


def test_the_field_hash_is_a_deterministic_fraction(plain):
    """Every star position, colour and object property is one of these."""
    values = [plain._hash2(x, y)
              for x in (0.0, 1.5, -3.25, 128.0) for y in (0.0, 7.9, -2.5)]
    assert all(0.0 <= v < 1.0 for v in values), values
    assert plain._hash2(1.5, 2.5) == plain._hash2(1.5, 2.5)
    assert plain._hash2(1.5, 2.5) != plain._hash2(2.5, 1.5)
    expected = math.sin(1.5 * 127.1 + 2.5 * 311.7) * 43758.5453123
    assert plain._hash2(1.5, 2.5) == pytest.approx(
        expected - math.floor(expected), abs=1e-12)


# --------------------------------------------------------------------------
# The three object slots


def _object_at(module, t, slot, speed=1.0):
    """Where slot ``slot`` has reached at ``t``, and what it looks like there.

    Recovers the object's own projected centre so a test can sample the pixel
    the object is actually on rather than hoping one of a few hundred pixels
    lands on it.
    """
    travel = t * (0.35 + 1.05 * speed) / 520.0 + 0.33 * slot
    epoch = math.floor(travel)
    phase = travel - epoch
    object_id = epoch * 3.0 + slot
    z = 7.2 + (0.16 - 7.2) * phase
    angle = 2.0 * math.pi * module._hash2(object_id + 2.2, 8.8)
    radius = 0.70 + 1.00 * module._hash2(object_id + 4.7, 5.6)
    x = radius * math.cos(angle) / max(z, 0.16)
    y = radius * math.sin(angle) / max(z, 0.16)
    #: Below 0.45 the slot holds a lit planet, above it a sun.
    kind = module._hash2(object_id + 7.1, 3.77)
    return phase, x, y, kind


def test_an_object_is_black_until_it_has_entered_the_frame(plain):
    """An object that popped into existence at full brightness would read as
    a glitch, so the first hundredth of a pass is a fade-in from nothing."""
    t = (1.0 + 0.005) * 520.0 / (0.35 + 1.05 * 1.0)
    phase, x, y, _kind = _object_at(plain, t, 0)
    assert phase < 0.02
    assert plain._object_color(x, y, t, 1.0, 0) == (0.0, 0.0, 0.0)


def test_an_object_is_black_again_once_it_has_swept_past(plain):
    """The fade-out is symmetric: an object does not blink out at the edge."""
    t = (2.0 + 0.999) * 520.0 / (0.35 + 1.05 * 1.0)
    phase, x, y, _kind = _object_at(plain, t, 0)
    assert phase > 0.90
    assert plain._object_color(x, y, t, 1.0, 0) == (0.0, 0.0, 0.0)


def test_a_sun_is_far_brighter_than_a_lit_planet(plain):
    """The two object kinds are drawn by different halves of the kernel: a
    planet is a shaded disc, a sun is a core with a halo and rays.

    A sun overdrives its channels so it clips to white in the frame, which is
    what makes it read as a light source rather than as a pale disc.
    """
    sun_t = 185.71428571428572
    planet_t = 63.14285714285714
    _phase, sun_x, sun_y, sun_kind = _object_at(plain, sun_t, 0)
    _phase, planet_x, planet_y, planet_kind = _object_at(plain, planet_t, 1)

    sun = plain._object_color(sun_x, sun_y, sun_t, 1.0, 0)
    planet = plain._object_color(planet_x, planet_y, planet_t, 1.0, 1)

    assert sun_kind >= 0.45, "the sun slot picked a planet"
    assert planet_kind < 0.45, "the planet slot picked a sun"
    assert max(sun) > 1.0, f"a sun that does not clip to white: {sun}"
    assert 0.0 < max(planet) < 1.0, f"a planet is a shaded disc, not a lamp: {planet}"
    assert max(sun) > 3.0 * max(planet)


# --------------------------------------------------------------------------
# One pixel of the flight


def test_a_sampled_pixel_is_three_clamped_channels(plain):
    """Every pixel the frame kernel writes comes through here, and the cast
    to uint8 is a bare ``int(255 * c)`` -- so a channel that came back above
    1.0 would be written into the frame as an out-of-range byte.

    A sun overdrives its channels well past 1.0 by design, so the clamp is
    tested where it actually bites: over the patch of sky one is crossing.
    """
    sun_t = 185.71428571428572
    clipped = 0
    for x in np.arange(-0.30, 0.15, 0.01):
        for y in np.arange(-0.65, -0.20, 0.01):
            pixel = plain.sample_space(float(x), float(y), sun_t, 1.0)
            assert len(pixel) == 3
            assert all(0.0 <= c <= 1.0 for c in pixel), (x, y, pixel)
            clipped += any(c == 1.0 for c in pixel)
    assert clipped, "no sample was bright enough for the clamp to matter"


def test_the_sky_is_dark_but_it_has_stars(plain):
    """The whole point of the pattern: what a user is reading stays the
    brightest thing on screen, and the field is still a field.

    A field, specifically: the bright pixels have to turn up in several parts
    of the frame. One bright patch is a passing object, which the scene also
    has -- so counting bright pixels alone cannot tell a star field from a
    single sun with no stars behind it at all.
    """
    wide, tall = 64, 40
    frame = plain.render_space_frame(wide, tall, 3.0, 1.0, 0.0, 0.0, 1)
    assert frame.shape == (tall, wide, 3)
    assert frame.dtype == np.uint8
    assert frame.mean() < 40, f"mean {frame.mean():.1f} is not a night sky"

    lit = frame.max(axis=2) > 90
    assert lit.sum() < 0.10 * wide * tall, "a lit frame is not a night sky"
    rows, cols = np.nonzero(lit)
    patches = {(row // 8, col // 8) for row, col in zip(rows, cols)}
    assert len(patches) >= 3, (
        f"every bright pixel is in {len(patches)} patch(es) of the frame, "
        f"which is an object rather than a field of stars")


def test_the_flight_is_a_pure_function_of_its_arguments(plain):
    """A dropped frame is then a repeat and never a jump."""
    once = plain.render_space_frame(WIDTH, HEIGHT, 4.2, 1.0, 0.0, 0.0, 1)
    twice = plain.render_space_frame(WIDTH, HEIGHT, 4.2, 1.0, 0.0, 0.0, 1)
    assert np.array_equal(once, twice)
    later = plain.render_space_frame(WIDTH, HEIGHT, 5.4, 1.0, 0.0, 0.0, 1)
    assert not np.array_equal(once, later)


def test_one_sample_a_pixel_is_the_pixel_the_kernel_sampled(plain):
    """The single-sample path maps pixel centres to scene coordinates."""
    t, speed = 3.0, 1.0
    frame = plain.render_space_frame(WIDTH, HEIGHT, t, speed, 0.0, 0.0, 1)
    denominator = float(min(WIDTH, HEIGHT))
    for row in range(HEIGHT):
        for col in range(WIDTH):
            x = (2.0 * (col + 0.5) - WIDTH) / denominator * 1.08
            y = (HEIGHT - 2.0 * (row + 0.5)) / denominator * 1.08
            r, g, b = plain.sample_space(x, y, t, speed)
            assert tuple(int(v) for v in frame[row, col]) == (
                int(255.0 * r), int(255.0 * g), int(255.0 * b)), (row, col)


def test_two_samples_a_side_average_the_four_corners_of_the_pixel(plain):
    """Supersampling is what keeps a sub-pixel star from crawling, so it has
    to be four samples of ONE instant averaged, not four instants blended."""
    t, speed = 3.6, 1.0
    frame = plain.render_space_frame(WIDTH, HEIGHT, t, speed, 0.0, 0.0, 2)
    denominator = float(min(WIDTH, HEIGHT))
    for row in range(HEIGHT):
        for col in range(WIDTH):
            totals = [0.0, 0.0, 0.0]
            for oy in (0.25, 0.75):
                for ox in (0.25, 0.75):
                    x = (2.0 * (col + ox) - WIDTH) / denominator * 1.08
                    y = (HEIGHT - 2.0 * (row + oy)) / denominator * 1.08
                    sampled = plain.sample_space(x, y, t, speed)
                    for channel, value in enumerate(sampled):
                        totals[channel] += value
            assert tuple(int(v) for v in frame[row, col]) == tuple(
                int(255.0 * total / 4.0) for total in totals), (row, col)


def test_a_pointer_offset_slides_the_field_rather_than_bending_it(plain):
    """The offsets are added to the scene coordinate, so the whole field
    translates and every star stays a point."""
    t, speed = 3.0, 1.0
    shifted = plain.render_space_frame(WIDTH, HEIGHT, t, speed, 0.3, -0.2, 1)
    still = plain.render_space_frame(WIDTH, HEIGHT, t, speed, 0.0, 0.0, 1)
    assert not np.array_equal(shifted, still)
    denominator = float(min(WIDTH, HEIGHT))
    x = (2.0 * (3 + 0.5) - WIDTH) / denominator * 1.08
    y = (HEIGHT - 2.0 * (4 + 0.5)) / denominator * 1.08
    r, g, b = plain.sample_space(x + 0.3, y - 0.2, t, speed)
    assert tuple(int(v) for v in shifted[4, 3]) == (
        int(255.0 * r), int(255.0 * g), int(255.0 * b))


# --------------------------------------------------------------------------
# The engine around the kernels


def test_the_supersampler_is_dropped_on_a_backdrop_sized_frame():
    """Four samples a pixel on a full-screen frame costs four times as much
    for a backdrop nobody looks straight at."""
    assert space.SpaceEngine._samples(320, 200) == 2
    assert space.SpaceEngine._samples(1920, 1080) == 1
    assert space.SpaceEngine._samples(800, 400) == 2
    assert space.SpaceEngine._samples(800, 401) == 1


def test_a_thread_cap_that_cannot_be_set_still_draws(monkeypatch):
    """A numba that will not take a thread count is a slower frame, never a
    backdrop that fails to appear."""
    import numba

    def refuse(_count):
        raise RuntimeError("cannot set the thread count after a parallel call")

    monkeypatch.setattr(numba, "set_num_threads", refuse)
    engine = space.SpaceEngine(3)
    assert engine.thread_count == 3
    frame = engine.render(8, 6, 3.0, 1.0)
    assert frame.shape == (6, 8, 3)


def test_a_thread_count_below_one_is_raised_to_one():
    assert space.SpaceEngine(0).thread_count == 1
    assert space.SpaceEngine(-4).thread_count == 1


def test_a_pull_and_a_push_that_cancel_leave_the_flight_where_it_was():
    """The pointer contributes ``pull * 0.22 - push * 0.35`` per axis, so a
    click that exactly answers the pull is the same frame as no pointer."""
    engine = space.SpaceEngine(2)
    neutral = engine.render(16, 12, 4.0, 1.0)
    cancelled = engine.render(16, 12, 4.0, 1.0, pointer_x=0.8, pointer_y=-0.5,
                              pull=0.35, push=0.22)
    pulled = engine.render(16, 12, 4.0, 1.0, pointer_x=0.8, pointer_y=-0.5,
                           pull=0.35, push=0.0)
    assert np.array_equal(neutral, cancelled)
    assert not np.array_equal(neutral, pulled)


# --------------------------------------------------------------------------
# The two backends, and the machine that has neither


def test_the_compiled_kernel_and_the_interpreted_one_draw_the_same_frame(plain):
    """What the tests above measure is what a user's machine renders."""
    compiled = space.render_space_frame(WIDTH, HEIGHT, 3.0, 1.0, 0.0, 0.0, 2)
    interpreted = plain.render_space_frame(WIDTH, HEIGHT, 3.0, 1.0,
                                           0.0, 0.0, 2)
    assert compiled.shape == interpreted.shape
    difference = np.abs(compiled.astype(int) - interpreted.astype(int)).max()
    assert difference <= 1, f"the two backends disagree by {difference}/255"


def test_without_numba_the_module_still_imports(no_numba):
    """The GPU backend does not need numba, so a machine without it must
    still get a module with a shader in it rather than an ImportError."""
    assert no_numba.njit is None
    assert no_numba.prange is range
    assert "render_sample" in no_numba.FRAGMENT_SHADER
    assert no_numba.OBJECT_SLOTS == 3
    assert no_numba.STAR_LAYERS == 6


def test_without_numba_the_cpu_renderer_says_what_is_missing(no_numba):
    """A frame of black with no explanation is the failure this replaces."""
    with pytest.raises(RuntimeError,
                       match="numba is required for the CPU space renderer"):
        no_numba.sample_space(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(RuntimeError,
                       match="numba is required for the CPU space renderer"):
        no_numba.render_space_frame(4, 4, 1.0, 1.0, 0.0, 0.0, 1)
