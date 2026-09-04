"""The weight ``attributable`` filters on is the weight it stores.

Instruction 310, entry A51. ``attributable`` built its competitor list as

    rest = [(float(e), float(w)) for e, w in (others or ()) if float(w) > 0.0]

which converts each weight TWICE -- once in the ``if`` and again in the tuple
it keeps. A weight object whose conversion is not pure (a lazily fetched
count, a mutable proxy, a value re-read from a stream) could therefore pass
the positivity filter and be stored non-positive. The well would then be
reported as "this guide can never be called" -- ``(False, 0.0)`` -- on data
whose weights all looked positive at the moment they were checked.

THE IMPURE WEIGHT HERE IS NOT A FICTION FOR THE TEST'S SAKE. It is the
smallest object that distinguishes reading once from reading twice, which is
the only thing the fix changes. A test that passed a plain float could not
tell the two versions apart, and would agree with the mutant.

A52 IS THE SAME LINE AND MUST NOT BE "FIXED" ALONGSIDE IT: that entry
describes the NaN handling as intended behaviour. Nothing here asserts
anything about NaN weights.
"""
from __future__ import annotations

import pytest

from spacr.guide_attribution import attributable


class _WeightReadTwiceGoesToZero:
    """Positive the first time it is converted, zero afterwards."""

    def __init__(self) -> None:
        self.reads = 0

    def __float__(self) -> float:
        self.reads += 1
        return 1.0 if self.reads == 1 else 0.0


def test_each_competitor_weight_is_converted_exactly_once():
    weight = _WeightReadTwiceGoesToZero()
    attributable(0.5, 1.0, 0.5, others=[(0.0, weight)])
    assert weight.reads == 1, (
        f"the weight was converted {weight.reads} times; the value used must "
        "be the value the positivity filter approved"
    )


def test_an_impure_weight_does_not_become_never_callable():
    """The defect's user-visible cost, stated as an assertion.

    Read twice, the stored weight is 0.0, ``total`` is 0.0 and the guard at
    ``if total <= 0`` returns the "can never be called" answer.
    """
    can_it, ceiling = attributable(
        0.5, 1.0, 0.5, others=[(0.0, _WeightReadTwiceGoesToZero())]
    )
    assert (can_it, ceiling) != (False, 0.0), (
        "a competitor whose weight was positive when filtered made the guide "
        "un-callable, which is the A51 defect"
    )
    assert ceiling > 0.0


def test_a_plain_positive_weight_is_unaffected():
    """The ordinary path must be unchanged by the fix."""
    plain = attributable(0.5, 1.0, 0.5, others=[(0.0, 1.0)])
    impure = attributable(
        0.5, 1.0, 0.5, others=[(0.0, _WeightReadTwiceGoesToZero())]
    )
    assert plain == impure


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_a_non_positive_weight_is_still_filtered_out(weight):
    """Reading once must not smuggle non-positive competitors back in.

    With the only competitor dropped, the list falls back to the implicit
    ``(0.0, 1 - prior)`` competitor rather than emptying, so the answer is the
    same as passing no competition at all.
    """
    assert (attributable(0.5, 1.0, 0.5, others=[(0.0, weight)])
            == attributable(0.5, 1.0, 0.5))
