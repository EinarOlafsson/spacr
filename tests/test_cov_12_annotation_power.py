"""Annotation-power edge: a non-finite guide share is left out of the tally.

A guide fraction that came back NaN or inf has to be skipped instead of
poisoning the reachability counts every re-run decision is argued from.
"""
from __future__ import annotations

from spacr.annotation_power import annotatable


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
