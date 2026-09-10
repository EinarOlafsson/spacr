"""A mask dim of None must not silently become the whole stack.

`stack[:, :, None]` does NOT raise. None is np.newaxis, so the expression
returns every channel with an axis inserted rather than one channel -- and
the mask-cleanup helpers then treated the entire stack as the nucleus or the
pathogen mask. Zeroing a label reached every channel that happened to share
the number.

Measured before the fix on an 8x8 stack: a cell labelled 5, nowhere near
the pathogen, was erased in full when nucleus_dim was None.
"""

from __future__ import annotations

import numpy as np

from spacr.utils import _remove_outside_objects, _remove_multiobject_cells


def _stack():
    """A cell and a pathogen that SHARE A LABEL and do not overlap.

    The shared number is the whole point: independent label spaces mean 5 in
    the cell channel and 5 in the pathogen channel are different objects.
    """
    stack = np.zeros((8, 8, 3), dtype=np.uint16)
    stack[1:4, 1:4, 0] = 5          # a cell
    stack[6:8, 6:8, 2] = 5          # an uncovered pathogen, same label
    stack[6:8, 6:8, 1] = 3          # its nucleus
    return stack


def test_none_is_newaxis_not_a_channel():
    """The property the bug rests on, stated once so the rest reads."""
    stack = _stack()
    assert stack[:, :, None].shape == (8, 8, 1, 3)
    assert stack[:, :, 1].shape == (8, 8)


def test_a_cell_survives_a_missing_nucleus_channel():
    out = _remove_outside_objects(_stack(), cell_dim=0, nucleus_dim=None,
                                  pathogen_dim=2)
    assert (out[:, :, 0] == 5).any(), "an unrelated cell was erased"
    assert int((out[:, :, 0] == 5).sum()) == 9


def test_the_uncovered_pathogen_still_goes():
    """The fix must not buy safety by doing nothing."""
    out = _remove_outside_objects(_stack(), cell_dim=0, nucleus_dim=None,
                                  pathogen_dim=2)
    assert not (out[:, :, 2] == 5).any()


def test_the_three_channel_path_is_unchanged():
    out = _remove_outside_objects(_stack(), cell_dim=0, nucleus_dim=1,
                                  pathogen_dim=2)
    assert not (out[:, :, 2] == 5).any(), "the uncovered pathogen stayed"
    assert not (out[:, :, 1] == 3).any(), "its nucleus stayed"
    assert (out[:, :, 0] == 5).any(), "the cell was taken with it"


def test_a_covered_pathogen_is_kept():
    stack = np.zeros((8, 8, 3), dtype=np.uint16)
    stack[1:6, 1:6, 0] = 5
    stack[2:4, 2:4, 2] = 5          # inside the cell
    out = _remove_outside_objects(stack, cell_dim=0, nucleus_dim=1,
                                  pathogen_dim=2)
    assert (out[:, :, 2] == 5).any()


def test_no_pathogen_channel_removes_nothing():
    out = _remove_outside_objects(_stack(), cell_dim=0, nucleus_dim=1,
                                  pathogen_dim=None)
    assert (out[:, :, 0] == 5).any()
    assert (out[:, :, 2] == 5).any()


def test_multiobject_survives_a_missing_nucleus_channel():
    stack = np.zeros((8, 8, 3), dtype=np.uint16)
    stack[1:6, 1:6, 0] = 5          # one cell
    stack[2:3, 2:3, 2] = 1          # two objects inside it
    stack[4:5, 4:5, 2] = 2
    out = _remove_multiobject_cells(stack.copy(), mask_dim=0, cell_dim=0,
                                    nucleus_dim=None, pathogen_dim=2,
                                    object_dim=2)
    assert not (out[:, :, 0] == 5).any(), "the multi-object cell was kept"


def test_multiobject_keeps_a_single_object_cell():
    stack = np.zeros((8, 8, 3), dtype=np.uint16)
    stack[1:6, 1:6, 0] = 5
    stack[2:3, 2:3, 2] = 1
    out = _remove_multiobject_cells(stack.copy(), mask_dim=0, cell_dim=0,
                                    nucleus_dim=None, pathogen_dim=2,
                                    object_dim=2)
    assert (out[:, :, 0] == 5).any()


def test_multiobject_without_an_object_channel_is_a_no_op():
    stack = _stack()
    out = _remove_multiobject_cells(stack.copy(), mask_dim=0, cell_dim=0,
                                    nucleus_dim=1, pathogen_dim=2,
                                    object_dim=None)
    assert np.array_equal(out, stack)
