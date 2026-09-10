"""Three small array helpers in utils, on the input they leave alone.

Each is a loop or a shape test whose "no" side had never run, and in each case
the no is what keeps a mask correct: a merge that did not happen, a background
label that was not relabelled, an activation map that was not transposed.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# merge_touching_objects — a pair that touches too little
# ---------------------------------------------------------------------------

def test_objects_sharing_too_little_boundary_stay_separate():
    """Arc 6301 -> 6300: the loop goes round without merging.

    The threshold defaults to 0.25, and the comment beside it says "more than
    25% of their boundary". A merge rule with no floor would fuse every
    adjacent pair in a confluent field into one object, which is the failure
    the fraction exists to prevent.
    """
    from spacr.utils import merge_touching_objects

    mask = np.zeros((24, 24), dtype=np.int32)
    mask[2:22, 2:11] = 1
    mask[2:22, 13:22] = 2
    mask[11, 11] = 1                          # a single-pixel bridge
    mask[11, 12] = 2

    out = merge_touching_objects(mask.copy(), threshold=0.9)

    assert set(np.unique(out)) == {0, 1, 2}


def test_objects_sharing_most_of_a_boundary_are_merged():
    """The taken side, on the same pair at a threshold it clears."""
    from spacr.utils import merge_touching_objects

    mask = np.zeros((24, 24), dtype=np.int32)
    mask[2:22, 2:12] = 1
    mask[2:22, 12:22] = 2

    out = merge_touching_objects(mask.copy(), threshold=0.01)

    assert len(set(np.unique(out)) - {0}) == 1


# ---------------------------------------------------------------------------
# _relabel_parent_with_child_labels — the background label
# ---------------------------------------------------------------------------

def test_the_background_is_never_relabelled():
    """Arc 6385 -> 6384: label 0 is skipped.

    Zero is not an object, and relabelling it would give every background
    pixel a child's label -- turning the whole field into one enormous object
    and every downstream area measurement into nonsense.
    """
    from spacr.utils import _relabel_parent_with_child_labels

    parent = np.zeros((12, 12), dtype=np.int32)
    parent[2:6, 2:6] = 1
    parent[7:11, 7:11] = 2

    # The first child spills OFF its parent onto background, so 0 is among
    # the parent labels it overlaps -- which is the only way to reach the
    # skip. A nucleus segmented slightly larger than its cell does exactly
    # this, and it is common at a cell boundary.
    child = np.zeros((12, 12), dtype=np.int32)
    child[3:8, 3:8] = 10
    child[8:10, 8:10] = 20

    out = _relabel_parent_with_child_labels(parent, child)
    out = out[0] if isinstance(out, tuple) else out

    assert out[0, 0] == 0, "the background must still be background"
    assert (out == 0).sum() >= (parent == 0).sum() - parent.size * 0.5


# ---------------------------------------------------------------------------
# _activation_map_to_2d — a stack that is neither one plane nor three
# ---------------------------------------------------------------------------

def test_a_multi_plane_activation_map_is_returned_unchanged():
    """Arc 6866 -> 6868: neither the single-plane nor the RGB case.

    A map with, say, eight planes is not something this helper can flatten
    without choosing which planes to keep -- and choosing silently would show
    the user a picture of three arbitrary channels labelled as the map.
    Returning it whole leaves that decision with the caller.
    """
    from spacr.utils import _activation_map_to_2d

    stack = np.zeros((8, 16, 16), dtype=np.float32)

    out = _activation_map_to_2d(stack)

    assert out.shape == (8, 16, 16)


def test_a_single_plane_map_is_unwrapped():
    """The first taken side."""
    from spacr.utils import _activation_map_to_2d

    assert _activation_map_to_2d(np.zeros((1, 16, 16))).shape == (16, 16)


def test_a_three_plane_map_is_moved_to_channels_last():
    """The second, which is what makes it drawable as RGB."""
    from spacr.utils import _activation_map_to_2d

    assert _activation_map_to_2d(np.zeros((3, 16, 16))).shape == (16, 16, 3)


def test_a_map_that_is_already_two_dimensional_is_returned_as_is():
    """The guard above all three."""
    from spacr.utils import _activation_map_to_2d

    assert _activation_map_to_2d(np.zeros((16, 16))).shape == (16, 16)
