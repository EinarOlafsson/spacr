"""Annotation-power edges: a non-finite share, no wells, and an unreachable ask.

Every number these helpers return is used to argue for or against re-running a
screen, so the degenerate inputs must produce an honest answer rather than a
plausible one. A guide fraction that came back NaN has to be skipped instead of
poisoning the tally, an empty plate map has to say so, and a decision threshold
no specificity can satisfy has to return NaN rather than a specificity above 1.
"""
from __future__ import annotations

import math

import numpy as np

from spacr.annotation_power import (_specificity_for, annotatable,
                                    screen_size_for)


def test_a_non_finite_guide_fraction_is_left_out_of_the_tally():
    """NaN and inf shares are skipped, so the pair count reflects real guides.

    A missing read count arrives as NaN through the fraction table. Counting it
    as a pair would inflate the denominator of every reachability figure and
    make a sparse screen look better annotated than it is.
    """
    fractions = {
        'A1': {'g1': 0.9, 'g2': float('nan'), 'g3': float('inf')},
        'A2': {'g1': 0.01},
    }
    out = annotatable(fractions, sensitivity=0.95, specificity=0.99)
    assert out['pairs'] == 2
    assert out['guides'] == 1
    assert out['guides_reachable'] == 1
    assert out['guides_unreachable'] == 0


def test_a_screen_with_no_wells_says_so_instead_of_dividing_by_zero():
    """An empty fraction table returns an error entry, not a size estimate.

    The caller reaches here when the upstream filter removed every well; a
    made-up multiplier would be quoted in a grant renewal.
    """
    assert screen_size_for({}, sensitivity=0.9, specificity=0.99) == {
        'error': 'no wells'}


def test_a_decision_threshold_of_one_admits_no_specificity():
    """Demanding certainty makes the required specificity NaN, not a number.

    Solving the posterior for a threshold of 1.0 has no solution below perfect
    classification; returning a clipped 1.0 would read as "just tighten the
    classifier a little" when nothing would do.
    """
    fractions = {'A1': {'g1': 0.5, 'g2': 0.5}, 'A2': {'g1': 0.5, 'g3': 0.5}}
    out = screen_size_for(fractions, sensitivity=0.95, specificity=0.99,
                          decision=1.0)
    assert math.isnan(out['specificity_needed_at_current_shape'])
    assert out['wells_now'] == 2

    assert math.isnan(_specificity_for(4.0, 0.95, 1.0))


def test_wells_that_hold_no_guides_leave_the_specificity_undefined():
    """Zero guides per well gives NaN, because there is no typical share to lift.

    A plate map that lost its guide assignments still reaches this helper; the
    honest answer is "undefined", not a specificity computed from 1/0.
    """
    fractions = {'A1': {}, 'A2': {}}
    out = screen_size_for(fractions, sensitivity=0.95, specificity=0.99)
    assert out['guides_per_well_now'] == 0.0
    assert math.isnan(out['specificity_needed_at_current_shape'])
