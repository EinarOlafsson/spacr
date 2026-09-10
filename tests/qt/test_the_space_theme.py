"""The space theme: forward flight through a dark star field.

Frames here are deliberately TINY. The renderer is a numba kernel compiled
with `parallel=True`, and compiling it at a real backdrop size is expensive
enough to matter on a shared machine; 96x60 exercises every branch -- the hue
field, all six parallax layers, all three object slots and the vignette --
because each is evaluated per pixel rather than per region.
"""
from __future__ import annotations

import numpy as np
import pytest

space = pytest.importorskip("spacr.qt.widgets.fractal_space")

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

WIDTH, HEIGHT = 96, 60


@pytest.fixture(scope="module")
def engine():
    return space.SpaceEngine(2)


def test_it_is_offered_as_a_pattern():
    from spacr.qt.widgets.fractal_travel import PATTERN_LABELS, PATTERNS
    from spacr.qt.preferences import FRACTAL_PATTERNS

    assert "space" in PATTERNS
    assert "space" in FRACTAL_PATTERNS
    assert PATTERN_LABELS["space"]


def test_both_backends_exist():
    """A pattern with only one backend is the only one that cannot run on
    a machine that lacks the other."""
    assert "render_sample" in space.FRAGMENT_SHADER
    assert "space_star_field" in space.FRAGMENT_SHADER
    assert "space_object" in space.FRAGMENT_SHADER
    assert callable(space.render_space_frame)


def test_a_frame_is_the_shape_the_widget_blits(engine):
    frame = engine.render(WIDTH, HEIGHT, 3.0, 1.0)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8


def test_space_is_mostly_black(engine):
    """The point of this pattern: what you are reading stays the brightest
    thing on screen."""
    frame = engine.render(WIDTH, HEIGHT, 3.0, 1.0)
    assert frame.mean() < 40, f"mean brightness {frame.mean():.1f} is not sky"


def test_but_it_has_stars(engine):
    """Mostly black must not become entirely black."""
    frame = engine.render(WIDTH, HEIGHT, 3.6, 1.0)
    assert frame.max() > 120, "nothing in the field is bright enough to be a star"


def test_the_flight_moves(engine):
    """A backdrop that repeats one frame is a picture, not a flight."""
    first = engine.render(WIDTH, HEIGHT, 3.0, 1.0)
    later = engine.render(WIDTH, HEIGHT, 3.6, 1.0)
    assert not np.array_equal(first, later)


def test_a_frame_is_a_pure_function_of_its_arguments(engine):
    """Same time, same speed, same frame -- so a dropped frame is a repeat
    and never a jump."""
    once = engine.render(WIDTH, HEIGHT, 4.2, 1.0)
    twice = engine.render(WIDTH, HEIGHT, 4.2, 1.0)
    assert np.array_equal(once, twice)


def test_the_pointer_steers_rather_than_warps(engine):
    """Bending a star field would curve the stars, which reads as a fault.

    Here the pointer nudges the heading, so the field slides the way a
    camera pans and every star stays a point.
    """
    still = engine.render(WIDTH, HEIGHT, 3.0, 1.0, pull=0.0)
    pulled = engine.render(WIDTH, HEIGHT, 3.0, 1.0,
                           pointer_x=0.6, pointer_y=0.4, pull=1.0)
    assert not np.array_equal(still, pulled)

    # Still a star field: the brightest pixels stay isolated points rather
    # than smearing into streaks.
    bright = (pulled.max(axis=2) > 120).sum()
    assert bright < pulled.shape[0] * pulled.shape[1] * 0.05


def test_it_takes_the_same_call_as_the_other_engines(engine):
    """The widget builds and drives all three patterns identically."""
    import inspect

    from spacr.qt.widgets.fractal_cascade import CascadeEngine

    theirs = inspect.signature(CascadeEngine.render).parameters
    ours = inspect.signature(space.SpaceEngine.render).parameters
    for name in list(theirs)[:6]:
        assert name in ours, f"the space engine cannot be called with {name}"


def test_speed_changes_the_flight(engine):
    slow = engine.render(WIDTH, HEIGHT, 5.0, 0.4)
    fast = engine.render(WIDTH, HEIGHT, 5.0, 2.0)
    assert not np.array_equal(slow, fast)
