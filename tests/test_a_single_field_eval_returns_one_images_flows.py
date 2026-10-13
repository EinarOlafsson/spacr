"""A bare 2-D Cellpose eval is one image, whatever its flows look like.

``CellposeModel.eval`` handed a single ``(H, W)`` array returns a 2-D mask
and a flat three-entry flows list -- the RGB rendering, the vectors and the
probability map for that ONE image -- rather than a list with one entry per
image. Read as a batch, ``len(masks)`` is the image height and a field that
segmented perfectly goes looking for H entries in a list of three.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.spacr_cellpose import parse_cellpose4_output


def _one_field():
    mask = np.zeros((6, 5), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    rgb = np.zeros((6, 5, 3), dtype=np.uint8)
    vectors = np.zeros((2, 6, 5), dtype=np.float32)
    probability = np.zeros((6, 5), dtype=np.float32)
    return mask, [rgb, vectors, probability]


def test_a_two_dimensional_mask_is_read_as_one_image_not_as_h_images():
    mask, flows = _one_field()

    masks, flows0, flows1, flows2, flows3 = parse_cellpose4_output(
        (mask, flows))

    assert masks is mask
    assert len(flows0) == len(flows1) == len(flows2) == len(flows3) == 1, (
        "one image in means one entry per flow list out")
    assert flows0[0] is flows[0]
    assert flows1[0] is flows[1]
    assert flows2[0] is flows[2]


def test_a_short_flow_list_pads_with_none_rather_than_raising():
    mask, flows = _one_field()

    _masks, _f0, _f1, _f2, flows3 = parse_cellpose4_output(
        (mask, flows[:2]))

    assert flows3 == [None], (
        "Cellpose omits the fourth flow; the slot stays empty")


def test_flows_that_are_not_a_sequence_name_the_type_they_were():
    with pytest.raises(ValueError, match="Unrecognized Cellpose flows type"):
        parse_cellpose4_output((np.zeros((4, 4)), object()))
