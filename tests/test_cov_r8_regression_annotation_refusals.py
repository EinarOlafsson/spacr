"""The annotation strategies refusing a design they cannot make honest.

Each of these is a refusal rather than a degraded answer, and that is the
module's whole posture: a hold-out drawn for the wrong reason, or a
propagation with no seed inside the pool, produces numbers that look
finished and are not. Saying so is the only honest option.

None of the four had been driven. They are the states a small or badly
shaped screen actually reaches -- too few labelled cells, a self-training
round that never fits, seeds outside the selectable pool, and a
neighbour search that observes no distance at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import regression_annotation as ra

# The plate builder and request helper are borrowed rather than
# re-invented: they produce a screen whose score is a known function of
# two columns, which is what makes these refusals meaningful.
from tests.test_the_cell_annotation_strategies import _plate, _request


class TestTheLabelledCountRefusalTheGuardAboveItRetired:
    """`if labelled.size < 4: raise ...` is not reached.

    The split runs over the LABELLED rows only -- stratifying over rows
    that carry no annotation would balance the hold-out on a label
    nobody wrote. But an earlier guard already refuses a screen with
    fewer than four SCORED cells in the chosen wells, and the labelling
    that follows draws its positives and negatives from that pool, so by
    the time the count is taken there are always at least four.

    Searched over 300 screen shapes -- two to six wells, two to six
    cells per well, two to five positives, three hold-out fractions --
    and none reaches it. The refusals that fire instead are asserted
    below, because those are the ones a user actually meets.
    """

    def test_a_screen_with_too_few_scored_cells_is_refused_first(self):
        frame = _plate(wells=1, per_well=3)
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra.prepare(_request(frame, n_positive=2))
        message = str(caught.value)
        assert "scored cell(s) are in the chosen wells" in message
        assert "top-scoring set" in message, (
            "the earlier refusal changed; the labelled-count guard below "
            "it may now be reachable")

    @pytest.mark.parametrize("wells, n_positive, refusal", [
        # Every labelled cell coming from one well means the hold-out would
        # be that well, and the fit would be measured on the only group it
        # never saw -- which is not a hold-out, it is a different experiment.
        pytest.param(1, 2, "every labelled cell comes from one well",
                     id="a-single-well-cannot-make-a-grouped-split"),
        pytest.param(2, 1, "one cell cannot make a class",
                     id="one-positive-cannot-make-a-class"),
    ])
    def test_a_design_that_cannot_be_split_is_refused_by_name(
            self, wells, n_positive, refusal):
        """The refusals that fire before the labelled-count guard."""
        frame = _plate(wells=wells, per_well=4)
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra.prepare(_request(frame, n_positive=n_positive))
        assert refusal in str(caught.value)

    def test_an_ordinary_screen_is_not_refused(self):
        """So the refusals above are visibly refusals."""
        prepared = ra.prepare(_request(_plate()))
        assert prepared is not None
        assert int(np.asarray(prepared.known).sum()) >= 4


class TestTheSelfTrainingFallbackThatCannotFire:
    """`if best is None: raise NotEnoughLabels(...)` is not reached.

    The round loop is `for index in range(max(1, int(request.rounds)))`,
    so it always runs at least once however `rounds` is set -- zero and
    negative both become one. And the first iteration assigns `best`
    unconditionally, because the test is `if best is None or ...`.

    So by the time the check is reached `best` is always a report. It is
    pinned to the two things that guarantee that.
    """

    @pytest.mark.parametrize("rounds", [0, -1, 1, 3])
    def test_at_least_one_round_always_runs(self, rounds):
        frame = _plate()
        request = _request(frame, rounds=rounds)
        prepared = ra.prepare(request)
        result = ra._run_self_training(prepared, request)
        assert result is not None, (
            f"rounds={rounds} produced no result; the `best is None` "
            "fallback may now be reachable")

    def test_a_run_reports_the_round_it_settled_on(self):
        """The live behaviour the fallback would have replaced."""
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)
        result = ra._run_self_training(prepared, request)
        assert result.notes, "a self-training run said nothing about itself"


class TestNeighbourPropagationWithNothingToPropagateFrom:

    @pytest.mark.parametrize("target, replacement, refusal", [
        # A seed that is not in the pool cannot carry its label anywhere.
        # `Prepared` is frozen, so the seeds are moved by replacing
        # `_seed_training` -- which is where they come from.
        pytest.param(
            "_seed_training",
            lambda frame: (lambda *_a, **_k: (
                np.asarray([10 ** 9, 10 ** 9 + 1], dtype=int),
                np.zeros(len(frame), dtype=int), [])),
            "seed cell", id="seeds-outside-the-selectable-pool"),
        # A neighbour search needs a space to measure a distance in.
        pytest.param(
            "_standardised",
            lambda frame: (lambda *_a, **_k: np.empty((5, 0))),
            "no feature column", id="no-feature-column-to-measure-in"),
    ])
    def test_propagating_from_nothing_is_refused_by_name(
            self, monkeypatch, target, replacement, refusal):
        """Propagating from nothing would return the input unchanged and
        call it a result."""
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)

        monkeypatch.setattr(ra, target, replacement(frame))

        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra._run_neighbour_propagation(prepared, request)
        assert refusal in str(caught.value)

class TestTheRadiusTheRunReports:
    """The radius is normally a quantile of the seed-to-neighbour
    distances the screen actually produced -- and the run SAYS which,
    because a propagation radius chosen silently is a parameter nobody
    can check.
    """

    def test_a_given_cut_is_used_and_named(self):
        frame = _plate()
        request = _request(frame, distance_cut=1.0)
        prepared = ra.prepare(request)
        result = ra._run_neighbour_propagation(prepared, request)
        joined = " ".join(result.notes).lower()
        assert "radius" in joined

    def test_without_one_the_quantile_actually_used_is_named(self):
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)
        result = ra._run_neighbour_propagation(prepared, request)
        joined = " ".join(result.notes).lower()
        assert "quantile" in joined or "distance" in joined

    def test_with_no_observed_distance_the_radius_is_zero_and_said_so(
            self, monkeypatch):
        """THE UNCOVERED ARM.

        With no seed-to-neighbour distance observed there is no quantile
        to take. A radius of zero propagates to nothing, which is the
        honest answer -- and the note says "no observed distance at all"
        rather than quoting a quantile of an empty set.
        """
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)

        real = ra.np.quantile

        class _NoNeighbours:
            def __init__(self, *a, **k):
                pass

            def fit(self, matrix):
                self._n = len(matrix)
                return self

            def kneighbors(self, *_a, **_k):
                empty = np.empty((self._n, 0))
                return empty, empty.astype(int)

        import sklearn.neighbors as skn

        monkeypatch.setattr(skn, "NearestNeighbors", _NoNeighbours)
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra._run_neighbour_propagation(prepared, request)

        message = str(caught.value)
        # The radius the arm produced, and where it came from, both said.
        assert "radius 0" in message
        assert "no observed distance at all" in message
        # And what to do about it, which is the half a bare refusal omits.
        assert "distance_cut" in message and "distance_quantile" in message
        assert "annotate more seeds" in message
