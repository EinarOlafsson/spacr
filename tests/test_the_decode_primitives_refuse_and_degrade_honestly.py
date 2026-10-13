"""The decode primitives on the inputs a real field sometimes hands them.

tests/test_in_situ_barcodes_are_decoded.py plants barcodes and demands them
back; this file holds `spacr.ops_sbs` to what it promises at the edges it
names in its docstrings: a stack of the wrong rank, a one-cycle experiment, a
threshold, a cross-talk fit with too few spots or no inverse, a base that
never wins, and a vote whose owners do not line up with its reads.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import ops_sbs


def test_a_stack_that_is_not_four_dimensional_is_refused():
    with pytest.raises(ValueError, match=r"\(cycles, channels, Y, X\)"):
        ops_sbs.estimate_read_locations(np.zeros((3, 4, 4)))


def test_one_cycle_scores_colour_across_channels():
    """With no cycle-to-cycle variance, a spot of one colour is what scores.

    A grey spot, equally bright in every channel, has no spread across
    channels and scores nothing, however bright it is.
    """
    stack = np.zeros((1, 4, 8, 8), np.float32)
    stack[0, :, 5, 5] = 7.0
    stack[0, 2, 3, 3] = 9.0

    score = ops_sbs.estimate_read_locations(stack)

    assert score.shape == (8, 8) and score.dtype == np.float32
    assert np.unravel_index(score.argmax(), score.shape) == (3, 3)
    assert score[5, 5] == 0.0


def test_a_threshold_drops_the_weaker_maxima():
    score = np.zeros((16, 16), np.float32)
    score[3, 3], score[10, 10] = 5.0, 1.0

    assert ops_sbs.find_peaks(score, gpu=False).tolist() == [[3, 3], [10, 10]]
    assert ops_sbs.find_peaks(score, threshold=2.0,
                              gpu=False).tolist() == [[3, 3]]


def test_an_unknown_compensation_method_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown method 'mean'"):
        ops_sbs.compensate_crosstalk(np.ones((10, 1, 4)), method="mean",
                                     gpu=False)


def test_fewer_spots_than_channels_are_returned_as_measured():
    """Four axes cannot be fitted from one spot, so none is invented."""
    values = np.array([[[1.0, 2.0, 3.0, 4.0]]], np.float32)

    assert np.array_equal(ops_sbs.compensate_crosstalk(values, gpu=False),
                          values)


def test_a_bleed_that_cannot_be_inverted_leaves_the_intensities_alone():
    """Two channels that are copies give two identical axes and no inverse."""
    values = np.abs(np.random.default_rng(0).normal(size=(50, 1, 4)))
    values = values.astype(np.float32)
    values[..., 1] = values[..., 0]

    assert np.array_equal(ops_sbs.compensate_crosstalk(values, gpu=False),
                          values)


def test_bases_that_never_win_under_the_median_rule_keep_their_own_axes():
    """Channels 2 and 3 are never the brightest, so no spot defines their
    axes; they stay the identity and the two real bases still call."""
    values = np.zeros((40, 1, 4), np.float32)
    values[:, 0, 0] = 10.0
    values[20:, 0, 1] = 20.0
    values[:, 0, 2] = 1.0
    values[:, 0, 3] = 0.5

    corrected = ops_sbs.compensate_crosstalk(values, method="median",
                                             gpu=False)

    assert corrected.shape == values.shape
    assert np.isfinite(corrected).all()
    assert corrected[0, 0].argmax() == 0
    assert corrected[25, 0].argmax() == 1


def test_owners_that_do_not_line_up_with_the_reads_are_refused():
    """Votes would land on the wrong objects, so nothing is counted."""
    with pytest.raises(ValueError, match="3 owners for 2 barcodes"):
        ops_sbs.assign_reads_to_objects(np.array([1, 1, 2]), ["ACGT", "ACGT"])
