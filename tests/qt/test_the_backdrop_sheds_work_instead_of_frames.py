"""Instruction 327 (1): why fullscreen was choppy and the backdrop was not.

Reported 2026-08-31: "the fullscreen mode fractal is super choppy while
the normal background one is super smoothe."

MEASURED, as the instruction requires, before anything was changed.

The cost IS the area -- at render_scale 0.5 a 2560x1440 fullscreen shades
5.12x the pixels of a 900x600 backdrop panel, and 11.53x at 4K. But area
alone is not the defect, because there is already a loop meant to absorb
it: ``_adapt_resolution`` measures the render time, compares it with the
frame budget, and computes a new scale between 0.58 and 1.35.

ALL OF THAT RAN AND NONE OF IT REACHED THE RENDERER. The ``render_scale``
branch REPLACED the requested pixel count instead of multiplying into it,
and ``render_scale`` defaults above zero -- so on every launch the
adaptive scale was computed, clamped, and thrown away. Sweeping it across
its whole range left the shaded size at 1280x720 every time.

So the frame rate fell instead of the resolution, which is exactly what
"choppy" means.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fractal_travel import target_render_size

PANEL = (900, 600)
FULLSCREEN = (2560, 1440)
UHD = (3840, 2160)


def _pixels(size, render_scale=0.5, adaptive=1.0):
    width, height = target_render_size(size[0], size[1], 1.0, render_scale,
                                       base_pixels=1_250_000,
                                       adaptive_scale=adaptive)
    return width * height


# ---------------------------------------------------------------------------
# The measurement that identified the candidate
# ---------------------------------------------------------------------------

def test_fullscreen_really_does_shade_far_more_than_a_panel():
    """The AREA candidate, quantified. Kept because the fix below is only
    worth having while this is true."""
    panel = _pixels(PANEL)
    full = _pixels(FULLSCREEN)
    uhd = _pixels(UHD)

    assert full / panel > 4.0, f"fullscreen is only {full / panel:.2f}x a panel"
    assert uhd / panel > 10.0, f"4K is only {uhd / panel:.2f}x a panel"


# ---------------------------------------------------------------------------
# THE DEFECT: the adaptive scale must reach the renderer
# ---------------------------------------------------------------------------

def test_a_lower_adaptive_scale_shades_fewer_pixels():
    """THE FIX. This is what was broken: every value gave the same size."""
    comfortable = _pixels(FULLSCREEN, adaptive=1.0)
    struggling = _pixels(FULLSCREEN, adaptive=0.58)

    assert struggling < comfortable, (
        "the adaptive scale does not reach the renderer; the frame rate "
        "falls instead of the resolution, which is what 'choppy' means")
    assert struggling / comfortable < 0.5, (
        f"a machine at the bottom of the range sheds only "
        f"{(1 - struggling / comfortable) * 100:.0f}% of the work")


@pytest.mark.parametrize("size", [PANEL, FULLSCREEN, UHD])
def test_the_response_is_monotonic_at_every_window_size(size):
    """A loop whose output is not monotonic cannot converge."""
    seen = [_pixels(size, adaptive=a) for a in (0.58, 0.8, 1.0, 1.2, 1.35)]
    assert seen == sorted(seen), f"not monotonic: {seen}"


def test_the_users_own_scale_still_means_what_it_says():
    """The adaptive term must not take the setting over.

    At a comfortable frame rate the scale the user chose is what they
    get: 0.5 is half native in each axis, so a quarter of the pixels.
    """
    native = FULLSCREEN[0] * FULLSCREEN[1]
    half = _pixels(FULLSCREEN, render_scale=0.5, adaptive=1.0)
    full = _pixels(FULLSCREEN, render_scale=1.0, adaptive=1.0)

    assert full == pytest.approx(native, rel=0.01)
    assert half == pytest.approx(native * 0.25, rel=0.02)


def test_a_higher_scale_is_not_silently_capped():
    """The reason the old fixed ceiling was removed: past a point,
    raising the scale did nothing and there was no way to tell."""
    assert _pixels(FULLSCREEN, render_scale=1.0) > _pixels(
        FULLSCREEN, render_scale=0.75)
    assert _pixels(FULLSCREEN, render_scale=0.75) > _pixels(
        FULLSCREEN, render_scale=0.5)


# ---------------------------------------------------------------------------
# The bounds the loop relies on
# ---------------------------------------------------------------------------

def test_nothing_shades_more_than_the_widget_has():
    """Shading more pixels than the display can show is pure waste."""
    for size in (PANEL, FULLSCREEN, UHD):
        native = size[0] * size[1]
        assert _pixels(size, render_scale=1.0, adaptive=1.35) <= native


def test_a_floor_stops_it_collapsing_to_nothing():
    """A scale of zero or a tiny window must still produce a picture."""
    width, height = target_render_size(320, 180, 1.0, 0.01,
                                       adaptive_scale=0.58)
    assert width >= 320 and height >= 180


def test_the_size_stays_even_and_keeps_its_aspect():
    """Odd dimensions break the 2x2 supersampling this renderer uses."""
    for size in (PANEL, FULLSCREEN, UHD):
        width, height = target_render_size(size[0], size[1], 1.0, 0.5)
        assert width % 2 == 0 and height % 2 == 0
        assert width / height == pytest.approx(size[0] / size[1], rel=0.02)
