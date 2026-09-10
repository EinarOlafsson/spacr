"""Measuring and zooming a packaged setting animation that has no content.

The zoom crops every animation so its scene covers a fixed share of the
square. A blank animation -- one whose every pixel is background, or whose
only marks are the well chrome -- has no bounds to crop to, and the module
has to show it as generated rather than dividing by an extent of zero or
cropping to a one-pixel scene.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets import animation_zoom as az

SIDE = az.SOURCE_SIZE


def _blank(n=2):
    return [np.zeros((SIDE, SIDE, 3), dtype=np.uint8) for _ in range(n)]


def _lit(box=(150, 150, 210, 210), n=2):
    left, top, right, bottom = box
    frames = []
    for _ in range(n):
        frame = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
        frame[top:bottom, left:right, :] = 255
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Measurement with nothing to measure
# ---------------------------------------------------------------------------

def test_no_frames_at_all_measure_as_an_empty_mask():
    """An empty sequence has no pixels, and the mask has to say so by shape.

    Every caller tests ``mask.size`` before using it; a mask of the source
    size full of False would claim the animation was measured and found
    blank, which is a different fact from "there was nothing to measure".
    """
    mask = az.content_mask([])

    assert mask.shape == (0, 0)
    assert mask.dtype == bool


def test_no_frames_have_no_content_bounds():
    """Bounds over nothing are None, not a zero-sized rectangle.

    ``(0, 0, 0, 0)`` would be indistinguishable from a single lit pixel in
    the corner, and the zoom would crop the animation to it.
    """
    assert az.content_bounds([]) is None


def test_an_extent_over_no_frames_is_zero():
    """A share of the square needs a square; with no frames there is none."""
    assert az.content_extent([]) == 0.0


def test_a_blank_animation_has_no_extent_to_zoom_towards():
    """A frame that is all background covers none of the square.

    Reporting anything else would make the zoom scale a blank picture up to
    the target fill, which is a magnified view of nothing.
    """
    assert az.content_extent(_blank()) == 0.0


def test_a_blank_animation_is_shown_as_generated(tmp_path):
    """With no content bounds the frames are rescaled and nothing is cropped.

    The crop is reported as the whole source square and the extent as zero,
    so a caller can tell "this could not be zoomed" from "this was zoomed to
    the target".
    """
    frames, crop, source_extent, fill, shows_field = az.zoom_frames(
        _blank(), size=60)

    assert len(frames) == 2
    assert frames[0].shape == (60, 60, 3)
    assert crop == (0, 0, SIDE)
    assert source_extent == 0.0
    assert fill == 0.0
    assert shows_field is True


def test_an_animation_that_decodes_to_no_frames_is_not_an_animation(monkeypatch):
    """A file Pillow opens but yields nothing from cannot be measured or shown.

    Both entry points answer with the "no animation" value rather than
    indexing frame zero, so a corrupt asset degrades to a text-only tooltip
    instead of raising into the event loop.
    """
    monkeypatch.setattr(az, "read_frames", lambda path: ((), ()))

    assert az.source_content_extent("anything.gif") == 0.0
    assert az.zoomed_animation("no-such-animation.gif", 60) is None


# ---------------------------------------------------------------------------
# Re-measuring the produced frames
# ---------------------------------------------------------------------------

def test_a_zoomed_animation_measures_itself_rather_than_trusting_the_maths():
    """The produced frames are re-measured through the same rule as the source.

    The target fill is a claim about the output, so it is checked against the
    output; deriving it from the crop arithmetic would make the number true
    by construction and useless as a check.
    """
    zoomed = az.zoom_frames(_lit(), size=120)
    frames, crop, source_extent, fill, shows_field = zoomed
    animation = az.ZoomedAnimation(
        path="synthetic", size=120, frames=frames, delays=(80, 80),
        source_extent=source_extent, fill=fill, crop=crop,
        shows_field=shows_field)

    measured = animation.measured_fill()

    assert measured == pytest.approx(az.TARGET_FILL, abs=0.06)
    assert measured == pytest.approx(animation.fill, abs=0.05)
