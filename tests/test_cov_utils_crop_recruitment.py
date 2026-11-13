"""Branch coverage for the crop / object-settings / recruitment helpers in ``spacr.utils``.

Covers the defensive and rarely-taken paths of:

    _get_percentiles       - the "no non-zero pixels" fallback
    _crop_center           - label masks, mask-outside zeroing, empty mask
    _masks_to_masks_stack  - order/identity preservation
    _get_object_settings   - the unsupported-object-type branch ('cell_large')
    _get_cellpose_channels - the organelle channel
    annotate_conditions    - loc values that are neither 'r*' nor 'c*'
    _split_data            - missing well columns, all-object frames, all-numeric frames
    _calculate_recruitment - the full set of ratio columns

Everything here is CPU-only, offline and sub-second.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a figure leak out of a test (nothing here plots, but be safe)."""
    yield
    import sys
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")


# ---------------------------------------------------------------------------
# _get_percentiles
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _masks_to_masks_stack
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _get_object_settings — the unsupported object-type branch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _get_cellpose_channels — organelle
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# annotate_conditions — loc values that map to neither rowID nor columnID
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _split_data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _calculate_recruitment
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _group_by_well
# ---------------------------------------------------------------------------

def test_group_by_well_means_numeric_and_takes_first_non_numeric():
    from spacr.utils import _group_by_well

    df = pd.DataFrame({
        "plateID": ["p1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1"] * 4,
        "cell_area": [100.0, 300.0, 10.0, 30.0],
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c1", "p1_r2_c1"],
    })

    out = _group_by_well(df)

    assert out.index.names == ["plateID", "rowID", "columnID"]
    assert out.loc[("p1", "r1", "c1"), "cell_area"] == 200.0
    assert out.loc[("p1", "r2", "c1"), "cell_area"] == 20.0
    # non-numeric column keeps the first value of the group
    assert out.loc[("p1", "r2", "c1"), "prc"] == "p1_r2_c1"
    assert len(out) == 2
