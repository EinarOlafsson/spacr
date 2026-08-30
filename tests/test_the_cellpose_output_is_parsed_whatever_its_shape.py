"""Unpacking what Cellpose 4 returns, whose shape depends on the call.

``parse_cellpose4_output`` normalises a single-image result and a batched one
into the same shape, because everything downstream iterates. The comment above
the first check explains the ordering and it is load-bearing: "The check goes
FIRST because a 2-D mask is one image whatever the flows look like."

Getting this wrong does not raise -- it produces a mask list of the wrong
length, and the run measures the wrong field's objects.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_a_single_two_dimensional_mask_is_wrapped_as_one_image():
    """The first check, taken.

    A 2-D mask is one image no matter what the flows look like, which is why
    this is decided on the mask alone and before anything inspects the flows.
    """
    from spacr.spacr_cellpose import parse_cellpose4_output

    mask = np.zeros((8, 8), dtype=np.int32)
    flows = [np.zeros((2, 8, 8)), np.zeros((8, 8)), np.zeros((8, 8))]

    masks, f1, f2, f3, f4 = parse_cellpose4_output((mask, flows))

    assert masks is mask
    assert len(f1) == len(f2) == len(f3) == len(f4) == 1


def test_a_single_mask_with_fewer_flows_pads_with_none():
    """The generator's ``if i < len(items) else None``.

    Cellpose returns three or four flow entries depending on version and
    call, and the caller unpacks four. Padding is what keeps a three-entry
    result from raising in the unpack.
    """
    from spacr.spacr_cellpose import parse_cellpose4_output

    mask = np.zeros((4, 4), dtype=np.int32)

    _masks, f1, f2, f3, f4 = parse_cellpose4_output((mask, [np.zeros((4, 4))]))

    assert f1[0] is not None
    assert f2 == [None] and f3 == [None] and f4 == [None]


def test_a_single_mask_with_no_flows_at_all_still_unpacks():
    """The same padding at its limit, which a masks-only call produces."""
    from spacr.spacr_cellpose import parse_cellpose4_output

    mask = np.zeros((4, 4), dtype=np.int32)

    masks, f1, f2, f3, f4 = parse_cellpose4_output((mask, []))

    assert masks is mask
    assert f1 == [None] and f4 == [None]


def test_a_batched_stack_is_not_taken_for_one_image():
    """The comment's own reasoning: an (N, H, W) stack has ndim 3.

    Treating a batch as one image would return every field's mask as a single
    array, and the caller would write one field's objects under every field's
    name.
    """
    from spacr.spacr_cellpose import parse_cellpose4_output

    stack = np.zeros((3, 8, 8), dtype=np.int32)
    # Per-image flows: one entry per image, which is Case B.
    flows = [[np.zeros((2, 8, 8)), np.zeros((8, 8))] for _ in range(3)]

    masks, f1, _f2, _f3, _f4 = parse_cellpose4_output((stack, flows))

    assert len(masks) == 3
    assert len(f1) == 3


def test_a_list_of_two_dimensional_masks_is_not_taken_for_one_image():
    """The other half of the comment: a list fails the isinstance.

    A list of per-field masks is the commonest batched return, and it must not
    hit the single-image path even though each element is 2-D.
    """
    from spacr.spacr_cellpose import parse_cellpose4_output

    masks_in = [np.zeros((8, 8), dtype=np.int32) for _ in range(2)]
    flows = [[np.zeros((2, 8, 8)), np.zeros((8, 8))] for _ in range(2)]

    masks, f1, _f2, _f3, _f4 = parse_cellpose4_output((masks_in, flows))

    assert len(masks) == 2
    assert len(f1) == 2


def test_the_four_array_batched_format_is_split_per_image():
    """Case A: four stacked arrays, one per flow kind.

    The second is indexed on its SECOND axis while the others are indexed on
    the first -- Cellpose stacks the vector field differently -- so a uniform
    split would hand every image the same slice.
    """
    from spacr.spacr_cellpose import parse_cellpose4_output

    n = 3
    flows = [np.zeros((n, 8, 8)), np.zeros((2, n, 8, 8)),
             np.zeros((n, 8, 8)), np.zeros((n, 8, 8))]
    masks = np.zeros((n, 8, 8), dtype=np.int32)

    _masks, f0, f1, f2, f3 = parse_cellpose4_output((masks, flows))

    assert len(f0) == len(f1) == len(f2) == len(f3) == n
    assert f1[0].shape == (2, 8, 8)


def test_a_flow_structure_that_matches_neither_case_is_refused():
    """The raise, which names the type and length it could not read.

    Silently returning empty flow lists would draw a montage with no flows and
    no explanation; the message is what tells a maintainer which Cellpose
    version changed under them.
    """
    from spacr.spacr_cellpose import parse_cellpose4_output

    masks = np.zeros((3, 8, 8), dtype=np.int32)

    with pytest.raises(ValueError) as excinfo:
        parse_cellpose4_output((masks, [np.zeros((8, 8))] * 2))

    assert "Unrecognized Cellpose flows format" in str(excinfo.value)
    assert "len=2" in str(excinfo.value)
