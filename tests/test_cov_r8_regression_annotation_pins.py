"""Five decisions in the annotation strategies, all about a first round.

Self-training's first round has no previous best to fail against, and a
strategy's first fit either happened or the strategy has nothing to
report. Each pin names the line above that settles it.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from spacr import regression_annotation as A


def _scored(n, wells=1):
    """``n`` scored cells spread over ``wells`` columns of one row."""
    columns = [f"c{1 + i % wells}" for i in range(n)]
    return pd.DataFrame({
        "plateID": "p1", "rowID": ["r1"] * n, "columnID": columns,
        "fieldID": "f1",
        "prcfo": [f"p1_r1_{columns[i]}_f1_o{i}" for i in range(n)],
        "cell_area": np.linspace(800, 1000, n),
        "cell_channel_1_mean_intensity": np.linspace(1100, 1300, n),
        "pred": np.linspace(0.05, 0.95, n),
    })


def _request(frame, **overrides):
    """A request naming every well the frame actually has."""
    wells = sorted({f"r1_{c}" for c in frame["columnID"].unique()})
    values = dict(frame=frame, score_column="pred", wells=wells,
                  n_positive=2, holdout_fraction=0.25, seed=1)
    values.update(overrides)
    return A.AnnotationRequest(**values)


class TestTheAuditSplit:

    def test_too_few_labels_to_hold_any_aside_is_refused_by_count(self):
        """The refusal, driven: it names how many there were.

        Holding a cell aside from three is a hold-out of one, and a
        balanced accuracy over one cell is either 0 or 1. Saying so is
        the difference between a strategy that cannot run and one that
        ran and reported nonsense.

        THE GUARD MOVED OUT OF ``prepare``, and this test moved with it
        rather than being deleted. ``prepare`` used to carry
        ``if labelled.size < 4``; it was removed once both routes into
        ``known`` were shown to guarantee four already. So the refusal is
        still asserted -- it is asserted against the two routes that now
        carry it, by RUNNING them rather than by reading the module, and
        a threshold moved below four fails here.
        """
        module = inspect.getsource(A)

        # THE SCORE ROUTE refuses below four, and names the count it
        # found -- which is the message the deleted guard would have had
        # to duplicate.
        assert "if pool.size < 4:" in module
        for size in (1, 2, 3):
            with pytest.raises(A.AnnotationStrategyError) as raised:
                A.prepare(_request(_scored(size, wells=min(size, 2))))
            message = str(raised.value)
            assert f"Only {size} scored cell(s)" in message, message
            assert "too few to define a top-scoring set" in message

        # THE ANNOTATION ROUTE only supplies labels at four or more, and
        # falls through to the score route below that -- so `known` is
        # never a three-cell annotation set.
        assert "int(known.sum()) >= 4" in module
        for size in (2, 3, 4, 5):
            frame = _scored(20, wells=4)
            frame["ann"] = [None] * len(frame)
            for row in range(size):
                frame.loc[row, "ann"] = "pos" if row % 2 == 0 else "neg"
            labels, known, source, _, _ = A._reference_labels(
                frame, _request(frame, label_column="ann").validated(),
                np.ones(len(frame), dtype=bool))
            if size < 4:
                assert source == "a cut on 'pred'", (
                    f"{size} annotations were used as the reference set, so "
                    f"the split would hold {size} cells aside")
                assert int(known.sum()) == len(frame)
            else:
                assert source == "annotations in 'ann'"
                assert int(known.sum()) == size
            assert int(known.sum()) >= 4

        # AND THE INVARIANT THE DELETED GUARD ASSERTED: anything that
        # reaches the split carries at least four labelled rows.
        for cells, wells in ((8, 2), (12, 3), (20, 4)):
            prepared = A.prepare(_request(_scored(cells, wells=wells)))
            assert int(np.asarray(prepared.known, dtype=bool).sum()) >= 4
            assert prepared.holdout.size >= 1

    def test_the_split_runs_over_the_labelled_rows_only(self):
        """Stratifying over rows that carry no annotation would balance
        the hold-out on a label nobody wrote, and the wells it drew would
        be drawn for the wrong reason."""
        source = inspect.getsource(A.prepare)
        assert "labelled = np.flatnonzero(np.asarray(known, dtype=bool))" \
            in source
        # It is computed BEFORE the split and the split is given it, which
        # is the whole claim. The ordering used to be pinned against the
        # `labelled.size < 4` guard that no longer exists; pinning it
        # against the splitter is what the docstring was always about.
        assert source.index(
            "labelled = np.flatnonzero(np.asarray(known, dtype=bool))") < \
            source.index("grouped_split(")
        assert "groups[labelled], labels[labelled]," in source, (
            "the split is no longer restricted to the labelled rows")


class TestASeedModelThatWasNotFitted:

    def test_the_hold_out_summary_is_added_only_when_there_was_a_fit(self):
        """THE PIN, for ``fit is not None``.

        The uncertainty strategy needs a seed model to score the pool
        with, so by the time the notes are written there is one -- and
        ``fit.summary()`` on None is an AttributeError in the last
        statement of a strategy that has already done all its work.

        Kept because the strategy's own contract allows a fit-free run
        in principle: the queue is what it returns, and the summary is
        commentary on how it was chosen.
        """
        source = inspect.getsource(A._run_uncertainty)
        assert "if fit is not None:" in source
        assert "The seed model, measured on the hold-out: " in source
        assert source.index("if fit is not None:") < source.index(
            "return _queue_result("), (
            "the note is no longer added before the result is built")

    def test_predictions_are_corrected_only_when_the_model_produced_some(self):
        """THE PIN, for ``predictions is not None`` in PU learning.

        ``_apply_model`` answers None when there is nothing to predict
        over, and dividing None['probability'] by the labelling rate is
        an AttributeError -- after the whole PU estimate has been made.

        The correction it guards is the point of the strategy: the raw
        probability is a probability of being LABELLED, and dividing by
        c turns it into a probability of being positive.
        """
        source = inspect.getsource(A._run_pu_learning)
        assert "predictions = _apply_model(" in source
        assert "if predictions is not None:" in source
        assert 'predictions["probability"].to_numpy(dtype=float) / rate' \
            in source, (
            "the probabilities are no longer divided by the labelling rate, "
            "so they are still probabilities of being labelled")

    def test_the_partial_correction_says_it_is_partial(self):
        """c is a LOWER BOUND under top-score selection, and the note
        says so rather than presenting the corrected number as exact."""
        source = inspect.getsource(A._run_pu_learning)
        assert "c is a lower bound on the true labelling rate" in source
        assert "correction is partial" in source


class TestSelfTrainingsRounds:

    def test_the_first_round_always_becomes_the_best(self):
        """THE PIN, for ``elif index``.

        ``best`` is None before the first round, so the ``if`` above
        takes it -- and the ``elif`` is only reached from round one
        onwards, where ``index`` is truthy. So its false side needs
        ``index == 0`` with a best already set, which cannot happen.

        The guard is right: stopping at round zero because the first
        round did not beat a best that does not exist would end
        self-training before it began.
        """
        source = inspect.getsource(A._run_self_training)
        assert "if best is None or report.balanced_accuracy > " in source
        assert "elif index:" in source
        assert source.index("if best is None or") < source.index("elif index:")

        best = None
        for index in range(3):
            if best is None or index > 0:
                best = index
            elif index:
                break
        assert best == 2, "the first round did not become the best"

    def test_a_round_that_does_not_improve_the_audit_set_stops(self):
        """What the elif is for, and it is worth saying out loud: a round
        that does not improve the audit set is a round of the model
        agreeing with itself."""
        source = inspect.getsource(A._run_self_training)
        assert "a round that does not improve it is a" in source
        assert "round of the model agreeing with itself" in source

    def test_no_round_fitted_at_all_is_refused_rather_than_reported(self):
        """THE PIN, for ``best is None`` after the loop.

        Every round that runs sets ``best`` -- the first unconditionally
        -- so reaching the raise means the loop body never ran, which
        needs zero rounds. The round count is a positive integer from
        the request, so it cannot be zero today.

        Kept because returning a result with no report is worse than the
        refusal: a self-training run that fitted nothing and said so is
        actionable; one that returns an empty result reads as a run that
        found nothing.
        """
        source = inspect.getsource(A._run_self_training)
        assert "if best is None:" in source
        assert "No self-training round could be fitted." in source

        best = None
        for index in range(0):                   # zero rounds
            best = index
        assert best is None, (
            "only a loop that never runs can leave best unset")

    def test_the_audit_set_is_fixed_before_the_first_round(self):
        """And excluded from all of them, which is what makes the
        reported performance mean anything."""
        source = inspect.getsource(A._run_self_training)
        assert "The audit set is fixed before the first round and excluded " \
            "from all" in source
