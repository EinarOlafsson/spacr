"""The convex ring fill on an empty input.

The classical network / irregular / ring back-ends are tested in
``tests/test_object_classical_segmentation.py``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np


def test_fill_rings_convex_on_empty_input_returns_empty():
    from spacr.object import _fill_rings_convex

    empty = np.zeros((16, 16), dtype=bool)
    out = _fill_rings_convex(empty)
    assert out.shape == (16, 16)
    assert not out.any()
