"""The distribution panels, on the frames they are actually handed.

A regression report is written from whatever table the fit used, and that
table sometimes has no well column, no guide shares, or three rows. Each
panel answers with a Panel that is not drawn and says why, because a
missing figure with no explanation is read as a broken run.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr.figures import distributions as dist  # noqa: E402


@pytest.fixture()
def axes():
    figure = plt.figure()
    ax = figure.add_subplot(111)
    yield ax
    plt.close(figure)


def _library(wells=40, guides=4, seed=0):
    """One row per guide-in-well, the shape the pipeline hands over."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(wells):
        shares = rng.dirichlet(np.ones(guides))
        response = float(rng.normal(0.0, 1.0))
        for guide, share in enumerate(shares):
            rows.append({"prc": f"p1_r1_c{well}", "grna": f"g{guide}",
                         "fraction": float(share),
                         "log_pred": response,
                         "pred": float(np.exp(response))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# finding the columns
# --------------------------------------------------------------------------

def test_a_frame_with_no_share_column_has_no_share_column():
    assert dist.fraction_column(pd.DataFrame({"prc": ["a"]})) is None
    assert dist.well_column(pd.DataFrame({"fraction": [0.5]})) is None


def test_one_numeric_column_is_the_response_without_guessing():
    frame = pd.DataFrame({"prc": ["a", "b"], "anything_at_all": [1.0, 2.0]})

    assert dist.response_column(frame) == "anything_at_all"


def test_several_numeric_columns_fall_back_to_the_known_names():
    frame = pd.DataFrame({"score": [1.0], "log_pred": [2.0], "other": [3.0]})

    assert dist.response_column(frame) == "log_pred", (
        "the pipeline's own dependent variable comes first in the order")


def test_a_named_column_that_is_not_there_is_not_invented():
    assert dist.response_column(pd.DataFrame({"a": [1.0]}), "b") is None


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------

def test_an_empty_sample_has_no_evenness_rather_than_a_perfect_one():
    assert np.isnan(dist.gini([]))
    assert np.isnan(dist.gini([-1.0, 2.0])), (
        "a negative share is not a share")


def test_shares_that_sum_to_nothing_have_no_evenness():
    assert np.isnan(dist.gini([0.0, 0.0, 0.0]))


def test_two_values_are_not_a_shape():
    shape = dist.shape_of([1.0, 2.0])

    assert shape["n"] == 2
    assert shape["verdict"] == "not measurable"
    assert np.isnan(shape["skew"])


def test_a_long_right_tail_is_called_strongly_skewed():
    values = np.concatenate([np.zeros(50), np.array([50.0, 80.0, 200.0])])

    assert dist.shape_of(values)["verdict"] == "strongly skewed right"


# --------------------------------------------------------------------------
# guide representation
# --------------------------------------------------------------------------

def test_without_a_share_column_the_representation_panel_says_so(axes):
    panel = dist.guide_fraction(axes, pd.DataFrame({"prc": ["a"] * 10}))

    assert panel.drawn is False
    assert panel.reason == "no fraction column"


def test_without_a_well_column_the_relative_view_cannot_be_computed(axes):
    frame = _library().drop(columns=["prc"])

    panel = dist.guide_fraction(axes, frame, relative=True)

    assert panel.drawn is False
    assert "no well column" in panel.reason


def test_the_raw_view_needs_no_well_and_says_what_it_mixes(axes):
    frame = _library()

    panel = dist.guide_fraction(axes, frame, relative=False)

    assert panel.drawn is True
    assert "raw share of its well" in panel.caption
    assert "no well column was available to separate them" in panel.caption
    assert panel.needs == ("fraction",), (
        "the raw view does not depend on the well column")


def test_shares_that_are_all_identical_still_get_a_histogram(axes):
    frame = pd.DataFrame({"prc": [f"w{i // 2}" for i in range(20)],
                          "fraction": [0.25] * 20})

    panel = dist.guide_fraction(axes, frame, relative=False)

    assert panel.drawn is True, (
        "a zero-width range must be widened, not binned into nothing")
    assert axes.patches, "something was actually drawn"


def test_too_few_usable_shares_is_a_reason_not_a_rug(axes):
    frame = pd.DataFrame({"prc": ["w1"] * 3, "fraction": [0.3, 0.3, 0.4]})

    panel = dist.guide_fraction(axes, frame, relative=False)

    assert panel.drawn is False
    assert f"at least {dist.MIN_VALUES}" in panel.reason


# --------------------------------------------------------------------------
# the response
# --------------------------------------------------------------------------

def test_a_frame_with_no_response_column_says_which_one_it_wanted(axes):
    frame = pd.DataFrame({"prc": ["a"] * 10, "grna": ["g"] * 10})

    panel = dist.response(axes, frame)

    assert panel.drawn is False
    assert panel.reason == "no response column could be identified"
    assert panel.needs == ("log_pred",)


def test_three_wells_are_not_a_response_distribution(axes):
    frame = _library(wells=3, guides=1)

    panel = dist.response(axes, frame, column="log_pred")

    assert panel.drawn is False
    assert "histogram needs at least" in panel.reason


def test_a_logged_response_reports_the_skew_the_log_removed(axes):
    frame = _library(wells=40, guides=2)

    panel = dist.response(axes, frame, column="log_pred")

    assert panel.drawn is True
    texts = [child.get_text() for child in axes.texts]
    assert any("skew before log" in text for text in texts), (
        "the raw column is here, so what the transform bought is stated")


# --------------------------------------------------------------------------
# writing them out
# --------------------------------------------------------------------------

def test_both_panels_are_written_under_the_names_a_run_expects(tmp_path,
                                                               capsys):
    open_before = set(plt.get_fignums())

    written = dist.save_distributions(_library(), str(tmp_path),
                                      response_variable="log_pred")

    assert set(written) == {"guide_fraction", "response"}
    assert os.path.basename(written["guide_fraction"]) \
        == "fraction_histogram.pdf"
    assert os.path.basename(written["response"]) == "log_pred_histogram.pdf"
    assert all(os.path.exists(path) for path in written.values())
    assert set(plt.get_fignums()) == open_before, (
        "every figure this opened is closed after it is saved")


def test_a_panel_that_cannot_be_drawn_is_omitted_and_named(tmp_path, capsys):
    frame = _library().drop(columns=["fraction"])
    open_before = set(plt.get_fignums())

    written = dist.save_distributions(frame, str(tmp_path),
                                      response_variable="log_pred")

    assert set(written) == {"response"}
    assert "Skipped guide_fraction: no fraction column" \
        in capsys.readouterr().out
    assert set(plt.get_fignums()) == open_before, (
        "the skipped figure is closed too")
