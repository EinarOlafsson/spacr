"""Both routes to a reference-label set produce at least four labels.

``prepare`` holds part of the labelled set aside to measure against. Fewer
than four labels makes that meaningless -- there is nothing to hold back
that still leaves anything to fit on -- so a guard raised for it.

The guard could not fire. Both routes into ``known`` already guarantee
four or more, and each refuses EARLIER and with a better message:

* the SCORE route stops at ``pool.size < 4`` -- "Only N scored cell(s)
  are in the chosen wells" -- which names the wells, the thing the user
  chose;
* the ANNOTATION route only supplies labels when at least four cells
  carry one, and otherwise falls back to the score route.

Instruction 288 counted the unreachable raise. It is gone, and this file
pins the two guarantees that make its absence safe -- which is the part
worth defending: if either threshold moved below four, removing the guard
would change behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.regression_annotation as ra


def _frame(n, wells=1):
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
    """A request naming every well the frame actually has.

    A SINGLE well is refused by a third guard -- "every labelled cell
    comes from one well... a random cell split would only measure how
    well the model memorised this well". The wells are derived from the
    frame rather than hardcoded so that changing the frame cannot
    silently start testing that refusal instead.
    """
    wells = sorted({f"r1_{c}" for c in frame["columnID"].unique()})
    values = dict(frame=frame, score_column="pred", wells=wells,
                  n_positive=2, holdout_fraction=0.25, seed=1)
    values.update(overrides)
    return ra.AnnotationRequest(**values)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_too_few_scored_cells_is_refused_by_the_earlier_guard(n):
    """AND IT NAMES THE WELLS, which is what the user chose.

    This is the guard that makes the deleted one unreachable, so it is
    asserted by MESSAGE rather than by exception type -- the type alone
    would pass for any of the four refusals in this function.
    """
    with pytest.raises(ra.AnnotationStrategyError) as raised:
        ra.prepare(_request(_frame(n, wells=1)))
    assert "scored cell" in str(raised.value)


def test_a_label_column_with_too_few_labels_falls_back_rather_than_raising():
    """The annotation route's threshold, driven.

    Two labelled cells is below its four, so it does not supply labels --
    and the fallback is the score route, not a refusal. That is the
    behaviour the deleted guard would have interrupted.
    """
    frame = _frame(40, wells=4)
    frame["truth"] = [1, 0] + [None] * 38
    prepared = ra.prepare(_request(frame, label_column="truth",
                                   n_positive=5))
    assert int(np.asarray(prepared.known, dtype=bool).sum()) >= 4


def test_a_usable_label_column_is_used_and_is_large_enough():
    """The other side: enough labels, so they are the reference set."""
    frame = _frame(40, wells=4)
    truth = [1, 1, 0, 0, 1, 0] + [None] * 34
    frame["truth"] = truth
    prepared = ra.prepare(_request(frame, label_column="truth",
                                   n_positive=5))
    assert int(np.asarray(prepared.known, dtype=bool).sum()) == 6


@pytest.mark.parametrize("n,n_positive", [(8, 2), (12, 3), (20, 4)])
def test_whatever_survives_the_earlier_guards_has_four_labels(n, n_positive):
    """THE PREMISE THE DELETION RESTS ON.

    Any request that gets past the earlier refusals carries at least four
    labels. If this ever failed, the removed raise would have been
    reachable and removing it would have changed behaviour.
    """
    prepared = ra.prepare(_request(_frame(n, wells=4),
                                   n_positive=n_positive))
    assert int(np.asarray(prepared.known, dtype=bool).sum()) >= 4
