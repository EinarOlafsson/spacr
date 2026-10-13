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

    def test_a_single_well_cannot_make_a_grouped_split(self):
        """The other refusal that fires before the count.

        Every labelled cell coming from one well means the hold-out
        would be that well, and the fit would be measured on the only
        group it never saw -- which is not a hold-out, it is a different
        experiment.
        """
        frame = _plate(wells=1, per_well=4)
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra.prepare(_request(frame, n_positive=2))
        assert "every labelled cell comes from one well" in str(caught.value)

    def test_one_positive_cannot_make_a_class(self):
        frame = _plate(wells=2, per_well=4)
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra.prepare(_request(frame, n_positive=1))
        assert "one cell cannot make a class" in str(caught.value)

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

    def test_the_loop_bound_is_still_clamped_to_one(self):
        import inspect

        source = inspect.getsource(ra._run_self_training)
        assert "range(max(1, int(request.rounds)))" in source, (
            "the round count is no longer clamped to at least one, so the "
            "`best is None` fallback may now be reachable")
        assert "if best is None or report.balanced_accuracy" in source, (
            "the first round no longer assigns best unconditionally")

    def test_a_run_reports_the_round_it_settled_on(self):
        """The live behaviour the fallback would have replaced."""
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)
        result = ra._run_self_training(prepared, request)
        assert result.notes, "a self-training run said nothing about itself"


class TestNeighbourPropagationWithNothingToPropagateFrom:

    def test_seeds_outside_the_selectable_pool_are_refused(self,
                                                           monkeypatch):
        """A seed that is not in the pool cannot carry its label anywhere.

        Propagating from nothing would return the input unchanged and
        call it a result. `Prepared` is frozen, so the seeds are moved by
        replacing `_seed_training` -- which is where they come from.
        """
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)

        monkeypatch.setattr(
            ra, "_seed_training",
            lambda *_a, **_k: (np.asarray([10 ** 9, 10 ** 9 + 1], dtype=int),
                               np.zeros(len(frame), dtype=int), []))

        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra._run_neighbour_propagation(prepared, request)
        assert "seed cell" in str(caught.value)

    def test_no_feature_column_to_measure_a_distance_in_is_refused(
            self, monkeypatch):
        """The refusal above it: a neighbour search needs a space."""
        frame = _plate()
        request = _request(frame)
        prepared = ra.prepare(request)

        monkeypatch.setattr(ra, "_standardised",
                            lambda *_a, **_k: np.empty((5, 0)))
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra._run_neighbour_propagation(prepared, request)
        assert "no feature column" in str(caught.value)


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
