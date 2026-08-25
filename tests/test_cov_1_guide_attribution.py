"""Wells the attribution model cannot speak about, and the cells it cannot place.

A pooled screen delivers wells with no usable sequencing and cells whose score
sits nowhere near any guide's expectation. Every branch below is one of those:
the answer has to be an explicit "no call", never a confident guide name
produced from a prior the data never supported.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.guide_attribution import (
    AMBIGUOUS,
    assign_well,
    attributable,
    attribute_well,
    posterior,
    posterior_multivariate,
)


# ---------------------------------------------------------------------------
# Nothing to attribute
# ---------------------------------------------------------------------------

def test_a_well_with_no_cells_has_an_empty_posterior():
    """Zero cells still returns the guide names, with a (0, G) matrix.

    Callers index the columns by guide, so the shape has to be right even when
    the well was empty -- a (0,) array would break the very next lookup.
    """
    r, guides = posterior([], {"g1": 0.5, "g2": 0.5}, {})
    assert guides == ("g1", "g2")
    assert r.shape == (0, 2)


def test_a_well_with_no_sequencing_calls_every_cell_ambiguous():
    """No usable guide fractions means no guide may be named.

    A well whose sequencing failed still has cells. Attributing them from a
    uniform prior would put invented guide names into the annotation table.
    """
    calls = attribute_well([0.1, 0.4, 2.0], {}, {})

    assert len(calls) == 3
    assert [c.guide for c in calls] == [AMBIGUOUS] * 3
    assert all(c.ambiguous and c.probability == 0.0 for c in calls)
    assert all(c.entropy == 0.0 for c in calls)


def test_a_well_with_no_sequencing_assigns_no_guide():
    """The one-guide-per-cell assignment refuses the same well.

    ``cost`` is infinite rather than 0 so that a caller comparing assignments
    cannot mistake "no information" for "a perfect fit".
    """
    result = assign_well([0.1, 0.4], {}, {})

    assert result.guides == (AMBIGUOUS, AMBIGUOUS)
    assert result.cost == float("inf")
    assert result.counts == {}
    assert result.decisive is False


# ---------------------------------------------------------------------------
# Cells no guide explains
# ---------------------------------------------------------------------------

def test_a_score_no_guide_can_explain_falls_back_to_the_prior():
    """An impossible score gets the sequencing prior, not a dropped row.

    Far enough from every guide's expectation the normal density underflows to
    exactly zero for all of them. The cell is still a cell that carries a
    guide, so the honest posterior is the prior -- and the row must still sum
    to one rather than to nothing.
    """
    priors = {"g1": 0.75, "g2": 0.25}
    r, guides = posterior([0.0, 1e6], priors, {"g1": 0.0, "g2": 1.0},
                          centre=0.0, scale=1.0)

    assert guides == ("g1", "g2")
    np.testing.assert_allclose(r.sum(axis=1), [1.0, 1.0])
    # The dead row keeps the sequencing proportions; the live one is dominated
    # by the guide whose effect matches the score.
    assert r[1, 0] > r[1, 1]
    assert r[0, 0] > r[0, 1]


def test_a_one_dimensional_measurement_is_read_as_one_column():
    """A plain score vector is the same call as a single-column matrix.

    Callers with one measurement per cell pass a 1-D array; reshaping it here
    is what stops ``n_cells`` being read as ``n_measurements``.
    """
    scores = np.array([0.0, 0.5, 1.0])
    priors = {"g1": 0.5, "g2": 0.5}
    effects = {"g1": [0.0], "g2": [1.0]}

    flat, guides, report = posterior_multivariate(scores, priors, effects)
    column, _, _ = posterior_multivariate(scores[:, None], priors, effects)

    assert flat.shape == (3, 2)
    assert report["n_measurements"] == 1.0
    np.testing.assert_allclose(flat, column)


# ---------------------------------------------------------------------------
# Whether a guide is attributable at all
# ---------------------------------------------------------------------------

def test_a_guide_no_cell_carries_is_not_attributable():
    """A prior of zero can never reach the threshold, at any effect size.

    A guide absent from a well's sequencing must come back False without the
    ceiling search: with no prior mass there is no score at which a cell would
    be called for it.
    """
    ok, ceiling = attributable(effect=10.0, scale=1.0, prior=0.0)
    assert ok is False
    assert ceiling == 0.0


def test_the_only_guide_in_a_well_is_always_attributable():
    """A prior of one is certainty before any score is seen.

    A singly-infected well leaves nothing to compete, so the ceiling is 1 and
    the answer does not depend on the effect size at all.
    """
    ok, ceiling = attributable(effect=0.0, scale=1.0, prior=1.0)
    assert ok is True
    assert ceiling == 1.0
