"""The narrow refusals of the cell-annotation menu, and the shapes behind them.

:mod:`spacr.regression_annotation` runs on whatever object table a screen
produced, and that table's failure modes are quiet ones: an empty score
column, a score that never varies, a named well the hold-out swallowed, a
hold-out that came out one-class, a cluster nobody landed in. Each has one
honest answer -- say what is wrong, in the words of the setting the user can
change -- and one dishonest one, which is to carry on and report a number.
Each test drives one such shape; some reach for the module's own helpers,
because a guard only the strategies can trip is a guard nobody can test.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacr import regression_annotation as ra


def _plate(*, wells: int = 8, per_well: int = 10, seed: int = 0) -> pd.DataFrame:
    """A small screen: eight wells whose score is a function of ``cell_area``."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(wells):
        for index in range(per_well):
            hit = rng.random() < 0.35
            rows.append({
                "plateID": "plate1",
                "rowID": f"r{1 + well // 4}",
                "columnID": f"c{1 + well % 4}",
                "fieldID": "f1",
                "prcfo": (f"plate1_r{1 + well // 4}_c{1 + well % 4}"
                          f"_f1_o{well}_{index}"),
                "cell_area": rng.normal(900 + 250 * hit, 80),
                "nucleus_area": rng.normal(300 + 60 * hit, 25),
            })
    frame = pd.DataFrame(rows)
    noise = rng.normal(0, 0.2, len(frame))
    frame["pred"] = 1.0 / (1.0 + np.exp(
        -(0.01 * (frame["cell_area"] - 900) + noise)))
    return frame


def _request(frame: pd.DataFrame, **overrides) -> ra.AnnotationRequest:
    values = dict(frame=frame, score_column="pred", n_positive=5,
                  holdout_fraction=0.25, seed=3)
    values.update(overrides)
    return ra.AnnotationRequest(**values)


@pytest.fixture(scope="module")
def plate():
    return _plate()


@pytest.fixture(scope="module")
def prepared(plate):
    return ra.prepare(_request(plate))


def _both_classes(prepared, each: int = 6):
    """``each`` selectable rows of each reference class, as positions."""
    by_class = [[int(p) for p in prepared.selectable
                 if prepared.labels[p] == value][:each] for value in (0, 1)]
    return by_class[0], by_class[1]


def test_the_menu_describes_every_strategy_it_lists():
    """The chooser renders :func:`menu` verbatim, one row per strategy. A
    menu shorter than the strategy list hides a strategy from the user, and
    a description missing its own title leaves the row it drew unlabelled.
    """
    lines = ra.menu()
    assert len(lines) == len(ra.STRATEGIES)
    for entry, line in zip(ra.STRATEGIES, lines):
        assert entry.title in line


def test_a_barely_scored_table_is_refused_with_the_count_it_has():
    """"There is no 'pred' column" and "only 2 of your 5 cells carry one"
    are different problems with different fixes, and this pre-flight is the
    only place a user is told which one they have before a run starts.
    """
    frame = pd.DataFrame({"pred": [0.9, 0.8, np.nan, np.nan, np.nan],
                          "cell_area": [1.0, 2.0, 3.0, 4.0, 5.0]})
    why = ra.missing_requirement("score_strata", frame, "pred")
    assert "only 2 of 5 cell(s) carry a finite 'pred'" in why
    assert "no annotation column is named" in why

    frame["pred"] = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert ra.missing_requirement("score_strata", frame, "pred") == ""


def test_a_request_that_cannot_make_a_run_is_refused_before_it_starts(plate):
    """Settings that cannot produce a run are caught at the request. Each
    would otherwise surface as an opaque scikit-learn error about an array
    shape, long after the user waited for a fit; here it names the setting.
    """
    with pytest.raises(ra.AnnotationStrategyError,
                       match="no cells to annotate"):
        _request(plate.iloc[:0]).validated()
    with pytest.raises(ra.AnnotationStrategyError,
                       match="n_positive must be at least 2"):
        _request(plate, n_positive=1).validated()
    with pytest.raises(ra.AnnotationStrategyError,
                       match="holdout_fraction must be a fraction"):
        _request(plate, holdout_fraction=1.0).validated()
    assert _request(plate).validated().n_positive == 5


def test_three_annotations_are_not_yet_an_annotation_column(plate):
    """The count is what the whole module branches on: with a usable column
    the hold-out is scored against what a person wrote, without one against
    a cut on the score. Three would make a hold-out of one cell.
    """
    frame = plate.copy()
    labels = np.array([""] * len(frame), dtype=object)
    labels[:3] = ["hit", "miss", "hit"]
    frame["ann"] = labels
    assert ra.usable_annotations(frame, "ann") == 0
    labels[3] = "miss"
    frame["ann"] = labels
    assert ra.usable_annotations(frame, "ann") == 4


def test_a_well_name_carrying_no_identity_token_selects_nothing(prepared):
    """``wells`` narrows the screen to the guide wells a user typed. An
    all-True fallback on an unparseable name would silently make the whole
    plate the chosen set and report its positives as that guide's.
    """
    assert ra.wells_selected(prepared.groups, ["r1_c1"]).any()
    assert not ra.wells_selected(prepared.groups, ["_"]).any()


def test_a_table_with_no_feature_columns_yields_an_empty_matrix(plate):
    """Strategies build the matrix before they know whether the table
    carries measurements, then refuse on its width. Raising here instead
    would replace their worded refusal with a numpy error about axis 0.
    """
    assert ra._matrix(plate, []).shape == (len(plate), 0)
    assert ra._standardised(plate, []).shape == (len(plate), 0)
    filled = ra._standardised(plate, ["cell_area", "nucleus_area"])
    assert filled.shape == (len(plate), 2)
    assert np.isfinite(filled).all()


def test_an_undefined_auc_is_reported_as_absent_not_as_a_number(prepared):
    """"ROC AUC nan" reads as a score; ``None`` reads as "not defined". Both
    ways scikit-learn signals an undefined AUC -- raising on an input it
    cannot score, returning NaN for a single-class hold-out -- must collapse
    to ``None``, because the lift the leakage control compares falls back to
    balanced accuracy on ``None`` and cannot on NaN.
    """
    size = prepared.holdout.size
    values = np.linspace(0.05, 0.95, size)
    assert ra._score_holdout(prepared, values, 10, ("cell_area",),
                             "hist").roc_auc is not None
    unscorable = values.copy()
    unscorable[0] = np.nan
    assert ra._score_holdout(prepared, unscorable, 10, ("cell_area",),
                             "hist").roc_auc is None
    one_class = dataclasses.replace(
        prepared, labels=np.ones(len(prepared.frame), dtype=int))
    assert ra._score_holdout(one_class, values, 10, ("cell_area",),
                             "hist").roc_auc is None


def test_a_fit_refuses_what_it_cannot_separate_and_predicts_what_it_did_not_see(
        prepared, plate):
    """One cell, or many cells of one class, is not a training set: both
    reach scikit-learn as an error naming an array, and are raised here as
    :class:`NotEnoughLabels` naming the selection instead. What the fit is
    then APPLIED to is the rest of the screen, which can be nothing at all:
    ``None`` is how the result type says "nothing was predicted", where a
    zero-row frame would be written out as an empty CSV.
    """
    request = _request(plate)
    zeros, ones = _both_classes(prepared)
    with pytest.raises(ra.NotEnoughLabels, match="Fewer than two cells"):
        ra._fit_report(prepared, zeros[:1], prepared.features, request)
    with pytest.raises(ra.NotEnoughLabels, match="carries the same label"):
        ra._fit_report(prepared, zeros, prepared.features, request)
    train = zeros + ones
    report, model, probabilities = ra._fit_report(
        prepared, train, prepared.features, request)
    assert report.n_train == len(train)
    assert probabilities.size == prepared.holdout.size
    rest = ra._apply_model(prepared, model, prepared.features, train)
    assert len(rest) == len(plate) - len(train)
    assert ra._apply_model(prepared, model, prepared.features,
                           np.arange(len(plate))) is None
    assert ra._apply_model(prepared, None, prepared.features, train) is None


def test_a_contrast_set_larger_than_the_pool_it_draws_from_is_refused(
        prepared, plate):
    """The contrast draw takes as many cells as there are positives, without
    replacement, from every selectable cell. When fewer are left than are
    wanted, numpy says "Cannot take a larger sample than population", naming
    neither the strategy nor the setting behind it.
    """
    narrow = dataclasses.replace(prepared, selectable=prepared.chosen[:2])
    with pytest.raises(ra.AnnotationStrategyError,
                       match="left to draw a contrast set"):
        ra._top_and_contrast(narrow, _request(plate))


