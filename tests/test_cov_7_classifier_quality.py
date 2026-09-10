"""What the classifier-quality helpers do when the evidence runs out.

Every function here reports on a classifier a user is about to trust with a
screen. The interesting cases are the degenerate ones: a split with no finite
scores, a prevalence band that no cell falls into, a mixture component with no
spread at all. Each of those has to come back with a defined answer, because
the alternative is a ZeroDivisionError or a silent empty table in the middle
of a calibration report.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr.classifier_quality import (
    Confusion,
    best_threshold,
    confusion,
    deconvolve,
    operating_points,
    sensitivity_by_prevalence,
)


# ---------------------------------------------------------------------------
# Confusion.summary
# ---------------------------------------------------------------------------

def test_a_confusion_summarises_all_four_of_its_numbers():
    """The one-line summary is what gets printed beside a calibration."""
    matrix = Confusion(true_positive=6, false_positive=2,
                       true_negative=8, false_negative=4, threshold=0.5)
    text = matrix.summary()

    assert "se 0.600" in text
    assert "sp 0.800" in text
    assert "accuracy 0.700" in text
    assert "prevalence 0.500" in text


def test_an_empty_confusion_summarises_as_nan_rather_than_crashing():
    """A split with no cells still prints; it must not divide by zero."""
    text = Confusion(0, 0, 0, 0, 0.5).summary()

    assert text.count("nan") == 4


# ---------------------------------------------------------------------------
# operating_points / best_threshold
# ---------------------------------------------------------------------------

def test_scores_with_nothing_finite_yield_no_operating_points():
    """No finite score means no threshold can be quoted at all."""
    assert operating_points([float("nan"), float("inf"), float("-inf")],
                            [True, False, True]) == []


def test_an_unusable_split_still_returns_a_confusion_at_the_default_cut():
    """``best_threshold`` must hand back a Confusion, never None.

    Callers read ``.sensitivity`` off the result to decide whether a
    Rogan-Gladen correction is possible; returning nothing would turn an
    uninformative split into an AttributeError.
    """
    point = best_threshold([], [])

    assert isinstance(point, Confusion)
    assert point.threshold == 0.5
    assert point.usable is False


def test_an_unknown_criterion_is_refused_by_name():
    """A misspelt criterion must not silently fall back to Youden's J."""
    with pytest.raises(ValueError, match="unknown criterion"):
        best_threshold([0.1, 0.9], [False, True], criterion="f1")


# ---------------------------------------------------------------------------
# sensitivity_by_prevalence
# ---------------------------------------------------------------------------

def test_no_usable_cell_gives_no_prevalence_bands():
    """Non-finite scores leave nothing to band, so the table is empty."""
    rows = sensitivity_by_prevalence([float("nan")] * 4,
                                     [True, False, True, False],
                                     ["A1", "A1", "A2", "A2"])
    assert rows == []


def test_an_empty_prevalence_band_is_dropped_not_reported_as_zero():
    """Asking for more bands than there are wells must not invent rows.

    Two wells at prevalence 0 and 1 with ten requested bands leaves eight of
    the ten bands holding no cell at all. A row for such a band would read as
    a measured sensitivity of nan at a prevalence nothing was measured at.
    """
    rows = sensitivity_by_prevalence([0.2, 0.8], [False, True], ["A1", "A2"],
                                     bins=10)

    assert len(rows) == 2
    assert [row["n"] for row in rows] == [1.0, 1.0]
    assert rows[0]["prevalence_low"] == 0.0
    assert rows[-1]["prevalence_high"] == 1.0


# ---------------------------------------------------------------------------
# deconvolve
# ---------------------------------------------------------------------------

class _ZeroSpreadMixture:
    """A mixture whose components collapse onto their means.

    Stands in for the degenerate fit that sklearn produces when a class holds
    one repeated score: the spread is zero and the normal CDF the estimator
    would otherwise evaluate is undefined.
    """

    def __init__(self, n_components, random_state=0):
        self.n_components = int(n_components)
        self.random_state = random_state

    def fit(self, values):
        self.means_ = np.array([[0.1], [0.9]])
        self.covariances_ = np.zeros((2, 1, 1))
        self.weights_ = np.array([0.75, 0.25])
        return self


def test_a_mixture_with_no_spread_still_reports_a_definite_call(monkeypatch):
    """Zero-variance components give hard 0/1 rates, not a division by zero.

    ``deconvolve`` turns each component into the share of it lying above the
    midpoint threshold. With no spread there is no share to integrate: the
    component is entirely on one side, and that is what has to be reported.
    """
    import sklearn.mixture

    monkeypatch.setattr(sklearn.mixture, "GaussianMixture", _ZeroSpreadMixture)
    result = deconvolve([0.1] * 15 + [0.9] * 15)

    assert result["threshold"] == pytest.approx(0.5)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["separation"] == 0.0
    assert result["trustworthy"] == 0.0


def test_too_few_scores_are_refused_before_a_mixture_is_fitted():
    """Under twenty scores cannot support a two-component fit."""
    result = deconvolve([0.5] * 19)

    assert result["error"] == 1.0
    assert result["n"] == 19.0


def test_a_confusion_counts_only_the_finite_scores():
    """A NaN score is not a negative call; it is not a call at all."""
    matrix = confusion([0.9, float("nan"), 0.1], [True, True, False])

    assert (matrix.true_positive, matrix.true_negative) == (1, 1)
    assert matrix.false_negative == 0
