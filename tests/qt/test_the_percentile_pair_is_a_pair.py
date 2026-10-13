"""The percentile pair can express the whole normalisation ladder.

Instruction 334: the control stopped at three decimals, so the top of spaCR's
own ladder -- the part that actually rescues a black-looking field -- could not
be typed at all.
"""
from __future__ import annotations

import pytest


def test_the_top_of_the_normalisation_ladder_is_expressible(qapp):
    """334: 99.9999 must be enterable, and survive the round trip.

    The control stopped at three decimals, so the ladder spaCR's own
    normalisation walks -- 98, 99, 99.9, 99.99, 99.999 -- ended exactly where
    it stops being useful. On a 2048x2048 field 99.999 still keeps forty
    pixels, and it is the last four or five (a cosmic ray, a saturated bead,
    one hot sensor pixel) that pin the display range and make every real
    object look black. 99.9999 clips those four.

    Asserted on the VALUE rather than on the widget's decimals, because a spin
    box that accepts the digits is no use if `value()` rounds them away.
    """
    from spacr.qt.widgets.percentile_pair import PercentilePair

    pair = PercentilePair([2.0, 98.0])
    pair.high().setValue(99.9999)

    assert pair.high().value() == pytest.approx(99.9999)
    low, high = pair.value()
    assert high == pytest.approx(99.9999), (
        f"the pair rounded the high end away: {high}")


def test_the_two_percentile_controls_agree_on_precision(qapp):
    """One quantity, one answer.

    make_masks' display boxes and this pair both express a percentile. Two
    different caps means the same number is enterable on one screen and not on
    the other, which reads as a bug wherever the user happens to be.
    """
    from spacr.qt.screens.make_masks import PERCENTILE_DECIMALS
    from spacr.qt.widgets.percentile_pair import DECIMALS

    assert DECIMALS == PERCENTILE_DECIMALS
