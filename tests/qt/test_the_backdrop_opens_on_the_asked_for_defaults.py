"""What spaceout draws on a machine that has never been configured.

Asked for on 2026-09-01: "orbit fold 2 supersampling backend auto scale
0.5 speed 1 should be the default".

The pattern default matters more than it looks. It was ``mandelbrot``,
which is the one pattern with NO CPU renderer -- so on every machine
without a usable GPU the stated default was a pattern that could not be
drawn, and ``pattern_for_this_machine`` quietly substituted the orbit
fold anyway. The default now says what actually happens.
"""
from __future__ import annotations

import pytest

from spacr.qt import fractal_defaults as F


def test_the_pattern_is_the_orbit_fold():
    assert F.DEFAULT_PATTERN == "orbit"


def test_the_orbit_fold_is_a_pattern_that_exists():
    """A default naming a pattern the renderer does not know is drawn as
    the fallback, silently."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.fractal_travel import PATTERNS, PATTERN_LABELS

    assert F.DEFAULT_PATTERN in PATTERNS
    assert "Orbit fold" in PATTERN_LABELS[F.DEFAULT_PATTERN]


def test_the_default_pattern_can_be_drawn_without_a_gpu():
    """THE REASON FOR THE CHANGE, stated as the property it fixes.

    The Mandelbrot needs a texture of its reference orbit and has no CPU
    renderer. A default that cannot run on a machine with no GPU is a
    default in name only.
    """
    assert F.DEFAULT_PATTERN != "mandelbrot"
    assert F.DEFAULT_PATTERN == F.FALLBACK_PATTERN, (
        "the default and the no-GPU fallback disagree again, so what a "
        "machine draws still depends on what it turned out to have")


def test_the_backend_is_auto():
    assert F.DEFAULT_BACKEND == "auto"


def test_the_scale_is_a_half():
    assert F.DEFAULT_SCALE == 0.5


def test_the_speed_is_one():
    assert F.DEFAULT_SPEED == 1.0


def test_supersampling_is_one():
    """CHANGED 2026-09-01: "default to computationally easy settings like
    supersampling 1 and scale 0.5 and speed 1".

    It squares the cost -- 2 is FOUR samples per pixel, not two -- so on
    a backdrop it is the single most expensive setting to have on by
    default. The picture is softer, which is the right trade for
    something drawn behind the interface.
    """
    assert F.DEFAULT_SUPERSAMPLING == 1


def test_the_easy_defaults_are_all_easy():
    """The three asked for together, so raising one alone is visible."""
    assert F.DEFAULT_SUPERSAMPLING == 1
    assert F.DEFAULT_SCALE == 0.5
    assert F.DEFAULT_SPEED == 1.0

    # Supersampling squares: this is what the default actually costs
    # relative to one sample per pixel at native resolution.
    cost = (F.DEFAULT_SUPERSAMPLING ** 2) * (F.DEFAULT_SCALE ** 2)
    assert cost <= 0.25, (
        f"the default shades {cost:.2f}x a native single-sampled frame")


def test_every_default_is_inside_its_own_limits():
    """A default outside its clamp is silently replaced on first read,
    which makes the constant a lie."""
    pytest.importorskip("PySide6")
    from spacr.qt.preferences import FRACTAL_LIMITS

    for name, value in (("scale", F.DEFAULT_SCALE),
                        ("speed", F.DEFAULT_SPEED),
                        ("supersampling", F.DEFAULT_SUPERSAMPLING)):
        low, high, why = FRACTAL_LIMITS[name]
        assert value >= low, f"{name} default {value} is below {low}: {why}"
        if high is not None:
            assert value <= high, f"{name} default {value} is above {high}"


def test_preferences_reads_these_and_not_a_second_copy():
    """The settings layer must resolve to the same numbers, or the
    constants above document something nothing uses."""
    pytest.importorskip("PySide6")
    from spacr.qt.preferences import get_fractal_settings

    values = get_fractal_settings()
    assert values["pattern"] == F.DEFAULT_PATTERN
    assert values["backend"] == F.DEFAULT_BACKEND
    assert values["scale"] == F.DEFAULT_SCALE
    assert values["speed"] == F.DEFAULT_SPEED
