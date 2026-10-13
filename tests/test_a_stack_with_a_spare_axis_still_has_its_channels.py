"""A field stack is promoted to (H, W, C) whatever axes it was saved with.

Both ends of the mask pass depend on the rank: Cellpose is called with
``channel_axis=-1`` and refuses a 2-D image outright, and the write-back
concatenates the mask onto the same array, which numpy refuses when the two
disagree. Promoting once at load keeps one convention for the whole pass.
"""
from __future__ import annotations

import numpy as np

from spacr.pipeline_v2 import _as_hwc


def test_a_bare_plane_gains_a_channel_axis():
    assert _as_hwc(np.zeros((6, 5), dtype=np.uint16)).shape == (6, 5, 1)


def test_a_stack_that_already_has_channels_is_left_alone():
    arr = np.zeros((6, 5, 3), dtype=np.uint16)

    assert _as_hwc(arr) is arr


def test_a_leading_singleton_axis_is_squeezed_away_not_kept():
    """A stack written as (1, H, W, C) is one field, not a batch of one."""
    arr = np.zeros((1, 6, 5, 3), dtype=np.uint16)

    promoted = _as_hwc(arr)

    assert promoted.shape == (6, 5, 3), (
        "the channel axis has to end up last for channel_axis=-1 to mean it")


def test_a_plane_wrapped_in_two_singletons_still_ends_up_hwc():
    assert _as_hwc(np.zeros((1, 6, 5, 1), dtype=np.uint16)).shape == (6, 5, 1)
