"""`align_field`'s refusals: a channel too far out, and cycles it cannot use.

372 PART 14-L found a cycle's four base channels up to 14 px apart, and
PART 14-M added the within-cycle step with a channel tolerance of 16 px. A
channel further out than that is REFUSED -- left where it was and reported --
rather than dragged onto a coincidental peak. No test showed that refusal,
nor the two errors for a field with nothing to align, nor what a shift wider
than the image leaves behind.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_cycles import _CHANNEL_TOLERANCE, _translate, align_field

pytest.importorskip("scipy")


def _background(size=128, seed=7):
    """A smooth textured cell background, which is what the channel step
    registers on once the spots are clipped away."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    return (ndimage.gaussian_filter(rng.normal(size=(size, size)), 3)
            * 100.0 + 500.0).astype(np.float32)


def test_a_channel_beyond_the_tolerance_is_refused_and_left_where_it_was():
    background = _background()
    near = np.roll(background, 10, axis=1)
    far = np.roll(background, 25, axis=1)
    assert 10 < _CHANNEL_TOLERANCE < 25

    field = align_field({1: [background, near, far, background]}, gpu=False)

    assert field.channel_shifts[1] == ((0, 0), (0, -10), (0, 0), (0, 0))
    assert field.refused_channels == ((1, 2),)
    assert np.array_equal(field.stack[0, 2], far)
    assert field.kept == (1,)


def test_a_field_with_no_cycle_to_align_is_refused():
    with pytest.raises(ValueError, match="no cycle of this field has planes"):
        align_field({1: None, 2: [], 3: [None]}, gpu=False)


def test_a_reference_cycle_that_was_not_read_is_refused():
    background = _background(64)

    with pytest.raises(ValueError, match="reference cycle 2 has no planes"):
        align_field({1: [background] * 4, 2: None}, reference=2, gpu=False)


def test_a_shift_wider_than_the_image_leaves_nothing_in_the_frame():
    """Zero-filled by design, so no intensity is invented at the edge."""
    image = np.ones((2, 6, 6), np.uint16)

    moved = _translate(image, 6, 0)

    assert moved.shape == image.shape and moved.dtype == image.dtype
    assert not moved.any()
    assert not _translate(image, 0, -7).any()
