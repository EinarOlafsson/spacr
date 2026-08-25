"""Malformed screen designs are refused before a simulation is run on them.

A power simulation produces numbers that go straight into a grant or a plate
order, so a design that is quietly wrong is worse than one that fails. Each of
these inputs is a real way a plate frame arrives broken -- filtered down to
nothing, ragged, or duplicated -- and each has to name what is wrong rather than
produce a table of NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.power_simulate import (MalformedPlateError, ScreenDesignError,
                                  _plate_grid, _require_columns,
                                  rdirichlet_stable,
                                  simulate_sequencing_plate,
                                  simulate_spot_plate)


def test_a_multi_dimensional_alpha_is_not_a_concentration_vector():
    """A 2-D alpha is refused instead of being flattened into a guess.

    A matrix arrives here when a caller passes a whole per-well table by
    mistake; flattening it would silently simulate a library with the wrong
    number of categories.
    """
    with pytest.raises(ScreenDesignError) as caught:
        rdirichlet_stable(np.ones((2, 3)), seed=1)
    assert 'scalar or a 1-D array' in str(caught.value)


def test_an_empty_alpha_has_no_categories_to_draw():
    """A zero-length concentration vector is refused, not returned empty.

    An empty library is the result of a filter that removed everything; a
    zero-length abundance vector would divide by zero in the next stage.
    """
    with pytest.raises(ScreenDesignError) as caught:
        rdirichlet_stable(np.array([]), seed=1)
    assert 'at least one category' in str(caught.value)


def test_an_empty_gene_library_is_refused_rather_than_simulated():
    """A library with no rows means an upstream filter removed everything.

    Building a plate from it would produce an empty spot plate that reads as a
    successful simulation of a screen with no genes in it.
    """
    empty = pd.DataFrame({'gene': [], 'gene_abundance': []})
    with pytest.raises(MalformedPlateError) as caught:
        simulate_spot_plate(empty, 4, 4.6, 1.0, seed=3)
    assert 'gene_library is empty' in str(caught.value)


def test_a_plate_that_is_not_a_frame_is_named_in_the_refusal():
    """The message says which parameter was wrong and what arrived instead.

    These helpers are called with several frames in a row; "must be a DataFrame"
    on its own leaves the caller guessing which argument to look at.
    """
    with pytest.raises(MalformedPlateError) as caught:
        _require_columns({'gene': [1]}, ('gene',), 'spot_plate')
    assert 'spot_plate must be a pandas DataFrame, got dict' in str(
        caught.value)


def test_a_plate_with_no_rows_says_so_before_the_pivot():
    """An empty tidy frame is refused where it is detected, by name.

    The gene-by-well pivot of an empty frame is a zero-sized matrix, and every
    later stage would report zeros rather than an error.
    """
    empty = pd.DataFrame({'gene': [], 'well': []})
    with pytest.raises(MalformedPlateError) as caught:
        _plate_grid(empty, 'spot_plate')
    assert 'spot_plate has no rows' in str(caught.value)


def test_a_duplicated_gene_well_pair_is_refused_by_the_pivot():
    """Two rows for one (gene, well) pair cannot both survive the pivot.

    The row count still matches the rectangle, so this is the corruption a shape
    check misses: one pair appears twice and another not at all, and the counts
    end up attributed to the wrong gene.
    """
    ragged = pd.DataFrame({
        'gene': ['g1', 'g1', 'g2', 'g2'],
        'well': ['w1', 'w1', 'w1', 'w2'],
    })
    with pytest.raises(MalformedPlateError) as caught:
        _plate_grid(ragged, 'spot_plate')
    assert 'duplicate (gene, well) pairs' in str(caught.value)


def test_a_non_finite_pcr_factor_is_refused_before_amplification():
    """An infinite log-mean amplification is refused, not exponentiated.

    ``exp(inf)`` gives an infinite barcode pool, and the hypergeometric draw
    that follows turns that into NaN read counts spread over every gene.
    """
    plate = pd.DataFrame({
        'gene': ['g1', 'g1', 'g2', 'g2'],
        'well': ['w1', 'w2', 'w1', 'w2'],
        'gene_in_well': [1, 0, 1, 1],
    })
    with pytest.raises(ScreenDesignError) as caught:
        simulate_sequencing_plate(plate, 100.0, float('inf'), 0.1, 1e5,
                                  seed=5)
    assert 'pcr_factor_mu must be finite' in str(caught.value)
