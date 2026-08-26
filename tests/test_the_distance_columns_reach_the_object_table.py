"""The object-distance families arrive under the names a user reads.

Instruction 241 closes on two numbers measured on real plate1 fields:
`cell_surface_to_nucleus_surface` with a median of 4.3 px and
`cell_nucleus_overlap_fraction` with a median of 0.31 -- "a nucleus inside
its cell, which is what those two numbers mean when the segmentation is
right".

NEITHER NAME IS WRITTEN ANYWHERE IN THE SOURCE. `object_distances` emits
`surface_to_nucleus_surface` bare, on purpose, and `measure` prefixes every
column of a props frame with its object type, so the name a user meets in
measurements.db is composed at run time out of two halves that live in
different modules. The existing coverage stops either side of the join:
tests/test_every_distance_worth_measuring.py drives the bare frame and
asserts the prefix is ABSENT, and the measure-level test asks only that some
column matching "distance" or "nearest" appeared -- which neither
`surface_to_nucleus_surface` nor `nucleus_overlap_fraction` does. Both
between-type families could drop out of the merge without a red test.

So this file asserts the composed names, which is the contract every
downstream reader of the table actually depends on.
"""

import numpy as np
import pytest

from spacr import measure


def _square(shape, y, x, size, label=1, into=None):
    mask = np.zeros(shape, dtype=np.uint16) if into is None else into
    mask[y:y + size, x:x + size] = label
    return mask


@pytest.fixture
def a_nucleus_inside_its_cell():
    """One 30 px cell holding a 10 px nucleus, 8 px in from the cell's rim."""
    cell = _square((80, 80), 10, 10, 30, label=1)
    nucleus = _square((80, 80), 18, 18, 10, label=1)
    pathogen = _square((80, 80), 60, 5, 6, label=1)
    return cell, nucleus, pathogen


def _settings(**over):
    base = {"cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
            "organelle_mask_dim": None, "cytoplasm": False,
            "channels": [0], "spatial_measurements": False,
            "object_distances": True}
    base.update(over)
    return base


def _frames(masks, **over):
    cell, nucleus, pathogen = masks
    images = np.full((*cell.shape, 1), 100, np.uint16)
    cell_df, nucleus_df, pathogen_df, _org, _cyt = \
        measure._morphological_measurements(
            cell, nucleus, pathogen, None, None, _settings(**over),
            zernike=False, channel_arrays=images)
    return cell_df, nucleus_df, pathogen_df


def test_the_two_columns_the_instruction_verified_are_there(
        a_nucleus_inside_its_cell):
    """The exact names instruction 241 reports medians for."""
    cell_df, _n, _p = _frames(a_nucleus_inside_its_cell)

    assert "cell_surface_to_nucleus_surface" in cell_df.columns
    assert "cell_nucleus_overlap_fraction" in cell_df.columns


def test_the_families_are_measured_in_both_directions(
        a_nucleus_inside_its_cell):
    """Centre-to-surface is not symmetric, so both orderings are columns.

    A frame that only carried the cell's view of the nucleus would answer
    "how far is the nucleus from this cell" and never "where does this
    nucleus sit in its cell", which is half of what the item asked for.
    """
    cell_df, nucleus_df, _p = _frames(a_nucleus_inside_its_cell)

    assert "cell_centre_to_nucleus_surface" in cell_df.columns
    assert "nucleus_centre_to_cell_surface" in nucleus_df.columns
    assert "nucleus_surface_to_cell_surface" in nucleus_df.columns
    assert "nucleus_cell_overlap_fraction" in nucleus_df.columns


def test_an_object_entirely_inside_another_overlaps_completely(
        a_nucleus_inside_its_cell):
    """The item's own acceptance criterion, read off the composed column."""
    _c, nucleus_df, _p = _frames(a_nucleus_inside_its_cell)

    assert nucleus_df["nucleus_cell_overlap_fraction"].iloc[0] == \
        pytest.approx(1.0)


def test_the_gap_reproduces_a_known_separation(a_nucleus_inside_its_cell):
    """The nucleus starts 8 px in from the cell's rim, so the gap is 8."""
    cell_df, _n, _p = _frames(a_nucleus_inside_its_cell)

    assert cell_df["cell_surface_to_nucleus_surface"].iloc[0] == \
        pytest.approx(8.0, abs=1.0)


def test_the_object_type_is_written_exactly_once(a_nucleus_inside_its_cell):
    """`cell_cell_surface_to_...` is what a second prefix would produce.

    The bare frame deliberately carries no object name so that measure can
    add it; a module that added its own would double it, which is the defect
    the blur column already hit once.
    """
    cell_df, _n, _p = _frames(a_nucleus_inside_its_cell)

    doubled = [c for c in cell_df.columns if c.startswith("cell_cell_")]
    assert not doubled, doubled


def test_nothing_is_measured_unless_it_is_asked_for(a_nucleus_inside_its_cell):
    """Opt-in, and off by default: real time on a 3-D field."""
    cell_df, _n, _p = _frames(a_nucleus_inside_its_cell,
                              object_distances=False)

    assert "cell_surface_to_nucleus_surface" not in cell_df.columns
    assert "cell_nucleus_overlap_fraction" not in cell_df.columns
