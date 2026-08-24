"""Coverage-fill for spacr.spacr_cellpose — the CPU-reachable parse error
paths. The GPU model-build functions are exercised by the @gpu suites.

The mask-comparison cases that used to sit here went with the functions they
covered: ``compare_cellpose_masks``, its ``compare_mask`` worker and its
``save_results_and_figure`` writer had no caller, and the sibling-directory
layout they compared can no longer hold the two condition folders they need.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import spacr_cellpose as SC


# ---------------------------------------------------------------------------
# parse_cellpose4_output — error branches
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_flows_wrong_type_raises(self):
        # flows not list/tuple → ValueError (line 32).
        with pytest.raises(ValueError):
            SC.parse_cellpose4_output(([np.zeros((4, 4))], "not-a-list"))

    def test_masks_length_undeterminable_raises(self):
        # masks with no len() → ValueError (lines 37-38).
        class _NoLen:
            pass
        with pytest.raises(ValueError):
            SC.parse_cellpose4_output((_NoLen(), [[], [], []]))

    def test_per_image_non_list_non_array_item(self):
        # A per-image flows item that is neither list nor ndarray → all
        # None branch (line 65).
        masks = [np.zeros((4, 4), dtype=np.int32)]
        flows = ["scalar-item"]   # len == num_images (1), item is str
        out = SC.parse_cellpose4_output((masks, flows))
        # returns (masks, f0, f1, f2, f3); f0 for the str item is None.
        assert out[0] is masks
        assert out[1] == [None]

    def test_unrecognized_structure_raises(self):
        # flows len != 4 and != num_images → ValueError (line 75).
        masks = [np.zeros((4, 4)), np.zeros((4, 4))]  # num_images = 2
        flows = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]  # len 3
        with pytest.raises(ValueError):
            SC.parse_cellpose4_output((masks, flows))

    def test_per_image_list_item_partial(self):
        # A per-image list item with <4 entries fills the rest with None.
        masks = [np.zeros((4, 4), dtype=np.int32)]
        flows = [[np.zeros((4, 4))]]   # one entry → f1,f2,f3 = None
        out = SC.parse_cellpose4_output((masks, flows))
        assert out[2] == [None]


def test_parse_batched_four_array_format():
    # Batched format: exactly 4 ndarray flows over the batch (lines 42-49).
    n = 2
    masks = np.zeros((n, 8, 8), dtype=np.int32)
    flow0 = np.zeros((n, 8, 8), dtype=np.float32)
    flow1 = np.zeros((3, n, 8, 8), dtype=np.float32)
    flow2 = np.zeros((n, 8, 8), dtype=np.float32)
    flow3 = np.zeros((n, 8, 8), dtype=np.float32)
    out = SC.parse_cellpose4_output((masks, [flow0, flow1, flow2, flow3]))
    m, f0, f1, f2, f3 = out
    assert len(f0) == n and len(f1) == n
