"""One object in a field, which every sparse plate edge produces.

``tree.query(coords, k=1)`` returns a 1-D array of self-distances while the
column indexing below it assumes 2-D, so the single-object case takes a reshape
that nothing else does. A wrong turn there does not raise -- it indexes the row
wrongly -- and the module's own docstring promises the result NEVER contains
NaN, which is what makes a silently wrong row expensive.

``_with_distances`` beside it is a NESTED function inside the measurement
driver and cannot be reached without running the whole driver, so its two
guards are not covered here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _spatial_measurements — a field holding exactly one object
# ---------------------------------------------------------------------------

def test_a_single_object_still_gets_a_full_spatial_row():
    """The ``k == 1`` reshape.

    ``tree.query(coords, k=1)`` returns a 1-D array of self-distances, and the
    column indexing below it assumes 2-D. Without the reshape the row would be
    indexed wrongly rather than raising -- and a field with one object is what
    every sparse plate edge produces.

    The module's own docstring promises the result NEVER contains NaN, which is
    the property this test pins for the smallest possible input.
    """
    from spacr.measure import _spatial_measurements

    mask = np.zeros((16, 16), dtype=np.int32)
    mask[4:8, 4:8] = 1

    out = _spatial_measurements(mask)

    assert len(out) == 1
    assert "label" in out.columns
    assert out["label"].tolist() == [1]
    assert not out.drop(columns=["label"]).isna().any().any()


def test_several_objects_get_their_neighbour_distances():
    """The ``k >= 2`` side, so the reshape above is visibly the other case."""
    from spacr.measure import _spatial_measurements

    mask = np.zeros((32, 32), dtype=np.int32)
    mask[2:6, 2:6] = 1
    mask[2:6, 20:24] = 2
    mask[20:24, 2:6] = 3

    out = _spatial_measurements(mask)

    assert len(out) == 3
    assert sorted(out["label"].tolist()) == [1, 2, 3]
    assert not out.drop(columns=["label"]).isna().any().any()


def test_an_empty_mask_measures_nothing_without_raising():
    """A field where segmentation found nothing, which is common at a plate edge."""
    from spacr.measure import _spatial_measurements

    out = _spatial_measurements(np.zeros((16, 16), dtype=np.int32))

    assert len(out) == 0
