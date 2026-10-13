"""A plate heatmap paints only wells that were measured.

The module exists to undo the invented zero: an absent well, a well below
min_count and a well whose every row holds a non-numeric value all reach the
grid as NaN in the underlying plotter and are filled with 0, which paints
them as a real measurement at the bottom of the shared colour scale. These
tests drive the paths where nothing at all survives that rule -- a plate with
no readable numbers, identifiers that name no well, an empty set of matrices
-- and check the colour scale honours a caller's explicit limits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.figures import plates


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def test_no_measured_wells_is_no_grid():
    """Zero is the caller's signal that there is nothing to draw; a 1x1 grid
    would put a single empty tile on the screen instead."""
    assert plates.full_plate_grid([], []) == (0, 0)
    assert plates.full_plate_grid([1, 2], []) == (0, 0)
    assert plates.full_plate_grid([], [3]) == (0, 0)


def test_identifiers_that_name_no_well_produce_no_matrices():
    """A `prc` that is not plate_row_column places no well. Drawing the
    plate anyway would show an empty grid that looks like an unpipetted
    plate rather than an unreadable identifier."""
    frame = pd.DataFrame({"prc": ["x_y_z", "x_y_w"], "v": [1.0, 2.0]})

    names, matrices, shape = plates.well_matrices(frame, "v")

    assert names == ["x"], "the plate is still named, so the caller can say so"
    assert matrices == []
    assert shape == (0, 0)


def test_a_plate_with_no_numeric_value_anywhere_is_drawn_as_absent():
    """Its wells have rows, so the count map says they are present. Painting
    them at the bottom of a scale shared with a real plate would invent a
    whole plate of zeros."""
    frame = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c2", "p2_r1_c1", "p2_r1_c2"],
        "v": [1.0, 2.0, "n/a", ""],
    })

    names, matrices, shape = plates.well_matrices(frame, "v")

    assert names == ["p1", "p2"]
    assert shape[0] >= 1 and shape[1] >= 2
    good, unreadable = matrices[0], matrices[1]
    assert np.isfinite(good).any(), "the readable plate lost its wells"
    assert not np.isfinite(unreadable).any(), (
        "a plate with no numbers was painted as a plate of zeros")


# ---------------------------------------------------------------------------
# The shared colour scale
# ---------------------------------------------------------------------------

def test_a_scale_over_nothing_is_a_unit_scale():
    """No finite well anywhere. The limits still have to be usable, and
    degenerate limits crash the colour normaliser."""
    assert plates.shared_limits([]) == (0.0, 1.0)
    assert plates.shared_limits([np.array([np.nan, np.nan])]) == (0.0, 1.0)


def test_fractional_limits_are_quantiles_and_whole_ones_are_absolute():
    """This is `generate_plate_heatmap`'s spec, and a caller's existing
    setting has to keep its meaning: `[0, 10]` is a range of the measurement,
    `[0.1, 0.9]` is the 10th to 90th percentile of it."""
    matrices = [np.array([[1.0, 2.0], [3.0, 100.0]])]

    low, high = plates.shared_limits(matrices, min_max=[0.1, 0.9])
    assert low > 1.0 and high < 100.0, (low, high)

    assert plates.shared_limits(matrices, min_max=[0, 10]) == (0.0, 10.0)


# ---------------------------------------------------------------------------
# Laying the plates out
# ---------------------------------------------------------------------------

def test_no_plates_needs_no_arrangement():
    """A composite of zero plates has no rows and no columns; returning
    (1, 0) would make the figure code divide by zero."""
    assert plates.small_multiple_layout(0, 1.5, 1.6) == (0, 0)
    assert plates.small_multiple_layout(-3, 1.5, 1.6) == (0, 0)


def test_a_colormap_object_is_taken_as_given_and_a_name_is_looked_up():
    """Callers pass either spelling, and both must end up masking NaN.

    ``set_bad("none")`` is why this helper exists: an unmeasured well reaches
    the grid as NaN, and a colormap that paints NaN with a real colour puts a
    solid square where there was no measurement -- the invented zero this
    module exists to undo, wearing a different hat.

    The caller's colormap is given a VISIBLE bad colour first, so the
    assertions can tell three things apart that a default-configured map
    cannot: that the returned map masks NaN, that it is a copy, and that the
    original still has the red it came in with. Matplotlib's viridis already
    defaults to a transparent bad, so passing that would have asserted
    nothing.
    """
    from matplotlib import colormaps

    mine = colormaps["viridis"].copy()
    mine.set_bad("red")

    from_object = plates._named(mine)
    from_name = plates._named("viridis")

    for ramp in (from_object, from_name):
        assert np.allclose(ramp.get_bad(), (0.0, 0.0, 0.0, 0.0)), (
            "an unmeasured well would be painted a real colour")

    assert from_object is not mine, "the helper handed back the caller's map"
    assert np.allclose(mine.get_bad(), (1.0, 0.0, 0.0, 1.0)), (
        "the caller's colormap was mutated, so every other figure it is used "
        "for now hides its NaNs too")
