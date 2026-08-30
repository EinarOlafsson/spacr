"""Choosing a classifier threshold, and refusing a criterion that is not one.

The threshold decides which cells count as positive, so it decides every
prevalence the screen reports. Youden's J -- sensitivity + specificity - 1 --
is the criterion, and the docstring gives the reason: it favours a threshold
whose sensitivity and specificity are both far from the diagonal, which is what
makes the Rogan-Gladen denominator stable.
"""
from __future__ import annotations

import numpy as np
import pytest


def _separable(n=200, seed=0):
    """Scores that separate cleanly, so the best threshold is unambiguous."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2 == 0
    scores = np.where(labels, rng.normal(0.8, 0.05, n),
                      rng.normal(0.2, 0.05, n))
    return scores, labels


def test_a_separable_score_picks_a_threshold_between_the_two_clouds():
    """The maximisation itself, on data where the answer is obvious."""
    from spacr.classifier_quality import best_threshold

    scores, labels = _separable()

    best = best_threshold(scores, labels)

    assert 0.2 < best.threshold < 0.8
    assert best.sensitivity > 0.9 and best.specificity > 0.9


def test_the_chosen_point_maximises_youden_over_every_candidate():
    """Line 151 and the key it maximises with, checked against the full set.

    Asserting the RESULT is best rather than merely plausible is what makes
    this a test of the criterion and not of the fixture.
    """
    from spacr.classifier_quality import best_threshold, operating_points

    scores, labels = _separable()

    best = best_threshold(scores, labels)
    every = operating_points(scores, labels)

    def youden(point):
        return point.sensitivity + point.specificity - 1.0

    assert youden(best) == pytest.approx(max(youden(p) for p in every))


def test_a_non_finite_candidate_is_ranked_last_not_chosen():
    """Line 150's ``-np.inf`` for a non-finite J.

    A threshold beyond every score gives a sensitivity or specificity of NaN,
    and NaN compares False against everything -- so without the guard max()
    could return it depending on iteration order, and the screen would be
    scored at a threshold nothing passes.
    """
    from spacr.classifier_quality import best_threshold

    # All one class: specificity is undefined at every point.
    scores = np.linspace(0.0, 1.0, 20)
    labels = np.ones(20, dtype=bool)

    best = best_threshold(scores, labels)

    assert best is not None
    assert np.isfinite(best.threshold)


def test_an_empty_family_returns_a_neutral_confusion():
    """The early return: nothing to threshold, so the default 0.5 stands.

    Raising would take down a report over a fold that happened to be empty,
    and 0.5 is the value a caller would have used anyway.
    """
    from spacr.classifier_quality import best_threshold

    best = best_threshold([], [])

    assert best.threshold == 0.5


def test_a_criterion_that_is_not_youden_is_refused_by_name():
    """The raise, which names what was asked for.

    Silently falling back to Youden would report a threshold chosen by a rule
    the user did not ask for, in a figure that says which rule it used.
    """
    from spacr.classifier_quality import best_threshold

    scores, labels = _separable(n=40)

    with pytest.raises(ValueError) as excinfo:
        best_threshold(scores, labels, criterion="f1")

    assert "f1" in str(excinfo.value)
