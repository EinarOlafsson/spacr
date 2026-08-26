"""Invasion helpers that answer "cannot be computed" rather than guessing.

Every one of these returns NaN or an empty answer where a number would be a
claim: a bimodality coefficient below the sample size that can support it, a
threshold method that could not find a valley, a control-well list that names
no wells. A confident wrong number here becomes an invasion rate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spacr import submodules as S


# ---------------------------------------------------------------------------
# Figure chrome
# ---------------------------------------------------------------------------

def test_a_figure_with_no_colour_bar_is_left_alone():
    """``sns.heatmap`` adds the bar's axes after the fact; it may not be there."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111)

    S._style_colour_bar(figure)     # a single-axes figure has no bar

    assert len(figure.axes) == 1


def test_a_colour_bar_loses_its_frame_and_follows_the_ink():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    figure = Figure()
    main = figure.add_subplot(121)
    bar = figure.add_subplot(122)

    S._style_colour_bar(figure)

    assert all(not spine.get_visible() for spine in bar.spines.values())
    assert any(spine.get_visible() for spine in main.spines.values())


# ---------------------------------------------------------------------------
# "Are there two populations here at all?"
# ---------------------------------------------------------------------------

def test_too_few_parasites_to_answer_is_answered_as_unknown():
    """The uncorrected coefficient false-passes badly on a small field."""
    assert math.isnan(S._bimodality_coefficient(np.arange(10.0)))
    assert math.isnan(S._bimodality_coefficient([], min_objects=4))


def test_one_value_repeated_is_one_population_not_a_division_by_zero():
    assert S._bimodality_coefficient(np.full(40, 7.0)) == 0.0


def test_a_two_point_mixture_scores_one_and_a_normal_scores_a_third():
    mixture = np.concatenate([np.zeros(20), np.ones(20)])
    assert S._bimodality_coefficient(mixture) == pytest.approx(1.0)

    normal = np.random.default_rng(3).normal(size=400)
    assert S._bimodality_coefficient(normal) < 5 / 9


def test_a_sample_whose_kurtosis_cannot_support_the_ratio_is_unknown(
        monkeypatch):
    """``kurtosis + 3`` at or below zero has no coefficient to report."""
    import scipy.stats

    monkeypatch.setattr(S, "_bimodality_coefficient",
                        S._bimodality_coefficient)  # keep the real one
    monkeypatch.setattr(scipy.stats, "kurtosis",
                        lambda *_a, **_k: -3.0)

    assert math.isnan(S._bimodality_coefficient(
        np.concatenate([np.zeros(20), np.ones(20)])))


# ---------------------------------------------------------------------------
# The outside-channel threshold
# ---------------------------------------------------------------------------

def test_a_threshold_method_nobody_offers_is_refused():
    with pytest.raises(ValueError, match="outside_threshold_method"):
        S._invasion_threshold(np.arange(10.0), method="kittler")


def test_a_field_with_one_distinct_value_has_no_threshold():
    assert math.isnan(S._invasion_threshold(np.full(20, 3.0)))
    assert math.isnan(S._invasion_threshold(np.array([1.0])))
    assert math.isnan(S._invasion_threshold(np.array([]), method="mean"))


def test_a_threshold_method_that_refuses_the_data_yields_no_threshold(
        monkeypatch):
    """skimage raises on inputs it cannot histogram; that is not a crash."""
    from skimage import filters

    def refuse(_values):
        raise ValueError("cannot compute a threshold from this")

    monkeypatch.setattr(filters, "threshold_otsu", refuse)

    values = np.concatenate([np.zeros(20), np.ones(20)])
    assert math.isnan(S._invasion_threshold(values, method="otsu"))


def test_a_real_gap_gets_a_threshold_inside_it():
    values = np.concatenate([np.full(20, 1.0), np.full(20, 9.0)])

    cut = S._invasion_threshold(values, method="otsu")

    assert 1.0 < cut < 9.0
    assert 1.0 < S._invasion_threshold(values, method="mean") < 9.0


# ---------------------------------------------------------------------------
# Which wells are controls
# ---------------------------------------------------------------------------

def _wells():
    return pd.DataFrame({
        "prc": ["p1_A_c1", "p1_B_c2", "p1_C_c3"],
        "rowID": ["A", "B", "C"],
        "columnID": ["c1", "c2", "c3"],
    })


def test_one_control_well_named_as_a_string_is_still_a_list():
    """A single well is the common case and must not be read character by character."""
    mask = S._invasion_control_mask(_wells(), "p1_B_c2")

    assert list(mask) == [False, True, False]


def test_no_control_wells_named_selects_none():
    assert not S._invasion_control_mask(_wells(), None).any()
    assert not S._invasion_control_mask(_wells(), []).any()
    assert not S._invasion_control_mask(_wells(), ()).any()


def test_a_control_can_be_named_by_row_by_column_or_by_well():
    frame = _wells()

    assert list(S._invasion_control_mask(frame, ["A"])) == [True, False, False]
    assert list(S._invasion_control_mask(frame, ["c3"])) == [False, False, True]
    assert list(S._invasion_control_mask(frame, ["B_c2"])) == [False, True, False]
