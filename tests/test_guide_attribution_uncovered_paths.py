"""A saturated guide, and the apportionment that gives every cell a slot.

Two things the attribution has to survive that nothing else pins down:

* a guide whose fitted effect is large enough to push the beta likelihood's
  mean off the end of the logit scale -- which used to raise
  ``OverflowError`` out of the middle of a well rather than attributing it;
* the rounding that turns read fractions into an integer number of cells.
  ``assign_well`` is only an assignment because those slots sum to exactly N,
  and the tests below check the counts it actually produces, including the
  awkward well where every guide's share rounds down and each one has to be
  topped back up.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spacr.guide_attribution import (
    AMBIGUOUS,
    assign_well,
    attribute_well,
    normalise_fractions,
    posterior,
)

#: Scores on the [0, 1] scale the beta likelihood is meant for: both extremes
#: and the well's own centre.
BETA_SCORES = [0.02, 0.2, 0.5, 0.8, 0.98]


def _largest_remainder(fractions, n):
    """The slot count each guide is owed, worked out independently here.

    A second implementation of the apportionment ``assign_well`` performs, so
    that the test compares its counts against an expectation rather than
    against itself.
    """
    priors = normalise_fractions(fractions)
    exact = {guide: priors[guide] * n for guide in priors}
    counts = {guide: int(math.floor(value)) for guide, value in exact.items()}
    short = n - sum(counts.values())
    by_remainder = sorted(exact.items(),
                          key=lambda item: -(item[1] - math.floor(item[1])))
    for guide, _ in by_remainder[:short]:
        counts[guide] += 1
    return counts


# ---------------------------------------------------------------------------
# A guide whose effect saturates the likelihood
# ---------------------------------------------------------------------------

def test_a_guide_whose_effect_saturates_the_logit_still_attributes_the_well():
    """A large negative effect is a saturated guide, not an error.

    The beta mean is moved on the logit scale, and ``1 / (1 + exp(-x))``
    raises ``OverflowError`` once the shifted logit falls below about -709.
    The whole well died on that -- no attribution at all, from a guide the
    regression merely fitted a large negative coefficient for.
    """
    r, guides = posterior(BETA_SCORES, {"flat": 0.5, "off": 0.5},
                          {"flat": 0.0, "off": -1e6},
                          likelihood="beta", centre=0.5, scale=0.15)

    assert guides == ("flat", "off")
    assert r.shape == (5, 2)
    assert np.all(np.isfinite(r))
    np.testing.assert_allclose(r.sum(axis=1), np.ones(5))
    # The sequencing constraint still holds: each guide keeps half the well.
    np.testing.assert_allclose(r.sum(axis=0), [2.5, 2.5], atol=1e-6)
    # The cell sitting exactly on the unshifted guide's mean belongs to it.
    assert r[2, 0] > r[2, 1]


def test_a_saturated_effect_gives_exactly_what_a_merely_clamped_one_gives():
    """Saturating is not an approximation of the arithmetic it replaces.

    The beta mean is clamped to 1e-6 already, and every shifted logit below
    about -14 lands on that clamp. An effect of -20 reaches it through the
    ordinary expression and an effect of -1e6 through the saturating branch,
    so the two wells must be attributed identically, to the bit.
    """
    clamped, _ = posterior(BETA_SCORES, {"flat": 0.5, "off": 0.5},
                           {"off": -20.0},
                           likelihood="beta", centre=0.5, scale=0.15)
    saturated, _ = posterior(BETA_SCORES, {"flat": 0.5, "off": 0.5},
                             {"off": -1e6},
                             likelihood="beta", centre=0.5, scale=0.15)

    np.testing.assert_array_equal(saturated, clamped)


def test_the_one_guide_per_cell_assignment_survives_a_saturated_guide():
    """``assign_well`` reads the same likelihood and used to die with it.

    The counts still come from sequencing, every cell still gets a guide, and
    the cell at the unshifted mean is not handed to the saturated guide.
    """
    got = assign_well(BETA_SCORES, {"flat": 0.6, "off": 0.4},
                      {"flat": 0.0, "off": -1e6},
                      likelihood="beta", centre=0.5, scale=0.15)

    assert AMBIGUOUS not in got.guides
    assert got.counts == {"flat": 3, "off": 2}
    assert got.guides[2] == "flat"
    assert math.isfinite(got.cost)


def test_a_saturated_positive_effect_is_answered_too():
    """The positive tail saturates by underflow rather than overflow.

    It never raised, and it must keep giving the same answer the clamp at
    the other end of the mean gives.
    """
    clamped = attribute_well(BETA_SCORES, {"flat": 0.5, "on": 0.5},
                             {"on": 20.0},
                             likelihood="beta", centre=0.5, scale=0.15)
    saturated = attribute_well(BETA_SCORES, {"flat": 0.5, "on": 0.5},
                               {"on": 1e6},
                               likelihood="beta", centre=0.5, scale=0.15)

    assert [c.guide for c in saturated] == [c.guide for c in clamped]
    assert [c.probability for c in saturated] == [c.probability
                                                  for c in clamped]


# ---------------------------------------------------------------------------
# The apportionment: read fractions into an integer number of cells
# ---------------------------------------------------------------------------

def test_the_spare_cell_goes_to_the_guide_with_the_largest_remainder():
    """Seven cells split 0.5 / 0.3 / 0.2 owe 3.5, 2.1 and 1.4 slots.

    Three, two and one is six; the seventh cell is the standard
    largest-remainder apportionment's, and 0.5 is the largest remainder.
    """
    got = assign_well(list(range(7)), {"a": 0.5, "b": 0.3, "c": 0.2},
                      {"a": 1.0, "b": 0.0, "c": -1.0})

    assert got.counts == {"a": 4, "b": 2, "c": 1}
    assert sum(got.counts.values()) == 7
    assert AMBIGUOUS not in got.guides


def test_every_guide_is_topped_up_when_all_of_them_round_down():
    """Fractions of 1e-6 and 1e-7 over eleven cells owe 10 and 1 slots.

    Both products land a fraction of a cell BELOW their integer, so the floor
    keeps nine of eleven cells and the top-up has to reach every guide in the
    well, not just the first. A top-up that stopped short would leave cells
    with no slot to be assigned to.
    """
    got = assign_well(list(np.linspace(-2.0, 2.0, 11)),
                      {"g0": 1e-6, "g1": 1e-7},
                      {"g0": 1.0, "g1": -1.0})

    assert got.counts == {"g0": 10, "g1": 1}
    assert len(got.guides) == 11
    assert AMBIGUOUS not in got.guides


@pytest.mark.parametrize("fractions,n", [
    ({"a": 0.5, "b": 0.3, "c": 0.2}, 7),
    ({"g0": 1e-6, "g1": 1e-7}, 11),
    ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 5),
    ({"a": 1.0, "b": 2.0, "c": 3.0, "d": 5.0}, 13),
    ({"x": 0.9999, "y": 0.0001}, 3),
    ({"p": 1e300, "q": 1e-300}, 4),
    ({"a": 1 / 7, "b": 2 / 7, "c": 4 / 7}, 100),
    ({"only": 0.42}, 6),
])
def test_the_slots_always_use_every_cell_and_no_more(fractions, n):
    """The invariant the assignment rests on, over awkward wells.

    Read fractions that do not divide, that differ by 600 orders of
    magnitude, and that repeat in binary. The counts must equal the
    largest-remainder apportionment worked out independently above and sum to
    the cell count exactly -- an over-count would leave a guide with slots no
    cell can fill, an under-count would leave a cell with no guide.
    """
    scores = list(np.random.default_rng(0).normal(0.0, 1.0, n))

    got = assign_well(scores, fractions, {g: 0.0 for g in fractions})

    assert got.counts == _largest_remainder(fractions, n)
    assert sum(got.counts.values()) == n
    assert len(got.guides) == n
    assert AMBIGUOUS not in got.guides
