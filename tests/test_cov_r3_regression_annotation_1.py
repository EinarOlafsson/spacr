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


def _annotated(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with every cell annotated, split at the median score."""
    out = frame.copy()
    out["ann"] = np.where(out["pred"] >= out["pred"].median(), "hit", "miss")
    return out


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


def test_which_columns_count_as_measurements_a_model_may_fit_on():
    """Booleans reach the object table from every mask-derived flag spaCR
    writes, so dropping them throws away real signal while keeping an
    invariant one hands the estimator no information. A NAMED list is the
    caller's decision instead: it keeps the constant column, because naming
    the classifier's own inputs means wanting the matrix it saw, and an
    absent name is a typo rather than a filter.
    """
    frame = _plate(wells=4, per_well=5)
    frame["cell_is_border"] = np.arange(len(frame)) % 2 == 0
    frame["cell_is_alive"] = True
    frame["cell_flat"] = 3.0
    inferred = ra.feature_columns(frame, "pred")
    assert "cell_is_border" in inferred
    assert "cell_is_alive" not in inferred and "cell_flat" not in inferred
    assert ra.feature_columns(frame, "pred", ["cell_flat", "cell_area"]) == (
        "cell_flat", "cell_area")
    with pytest.raises(ra.AnnotationStrategyError,
                       match="not in the object table"):
        ra.feature_columns(frame, "pred", ["cell_no_such_column"])


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


def test_the_score_is_named_once_and_a_barely_measured_column_is_not_judged(
        plate):
    """The leakage control drops the score's inputs and re-fits, so a column
    swept in on a two-point rank correlation leaves the honest fit for no
    reason -- and the score itself must be listed once, not twice.
    """
    frame = plate.copy()
    frame["cell_sparse"] = np.nan
    frame.loc[frame.index[:2], "cell_sparse"] = frame["pred"].to_numpy()[:2]
    inputs = ra.score_input_columns(
        frame, "pred", ["pred", "cell_area", "cell_sparse"],
        correlation_cut=0.5)
    assert list(inputs).count("pred") == 1
    assert "cell_area" in inputs
    assert "cell_sparse" not in inputs


def test_a_well_name_carrying_no_identity_token_selects_nothing(prepared):
    """``wells`` narrows the screen to the guide wells a user typed. An
    all-True fallback on an unparseable name would silently make the whole
    plate the chosen set and report its positives as that guide's.
    """
    assert ra.wells_selected(prepared.groups, ["r1_c1"]).any()
    assert not ra.wells_selected(prepared.groups, ["_"]).any()


def test_the_holdout_carries_its_own_labels_and_an_empty_set_has_no_share(
        prepared):
    """:meth:`Prepared.holdout_labels` is what every fit is scored against,
    so it must be the reference labels at exactly the hold-out rows. And an
    empty selection's positive share is unknown, not the 0.0% that would
    tell a user their strategy enriched nothing.
    """
    held = prepared.holdout_labels()
    assert np.array_equal(held, prepared.labels[prepared.holdout])
    assert 0.0 < prepared.positive_share(prepared.holdout) < 1.0
    assert np.isnan(prepared.positive_share([]))


def test_a_score_that_cannot_define_a_positive_set_is_refused(plate):
    """An empty score column -- the commonest broken join -- would have the
    quantile computed over an empty pool and every cell labelled against a
    NaN threshold. A constant one puts every cell on one side of any cut,
    training a classifier that predicts one class and scoring it as perfect
    on a hold-out that also holds one. Both are refused in words.
    """
    frame = plate.copy()
    frame["pred"] = np.nan
    with pytest.raises(ra.AnnotationStrategyError,
                       match="Every value of 'pred' is missing"):
        ra.prepare(_request(frame))
    frame["pred"] = 0.5
    with pytest.raises(ra.AnnotationStrategyError,
                       match="puts every cell on one side"):
        ra.prepare(_request(frame))


def test_a_table_with_no_plate_identity_cannot_be_split_by_well():
    """The splitter refuses in words; this module must re-raise that refusal
    as its own type, because a caller of :func:`prepare` catches
    :class:`AnnotationStrategyError` and would otherwise see a bare
    ``ValueError`` escape from an import it never made.
    """
    frame = pd.DataFrame({"pred": np.linspace(0.05, 0.95, 12),
                          "cell_area": np.linspace(500, 1500, 12)})
    with pytest.raises(ra.AnnotationStrategyError,
                       match="missing identity columns"):
        ra.prepare(_request(frame))


def test_the_splitters_own_typed_refusal_is_passed_through_unchanged(
        plate, monkeypatch):
    """:class:`AnnotationStrategyError` is a :class:`ValueError`, so the
    ``ValueError`` handler underneath would otherwise catch it, re-wrap it
    in a fresh base-class error and flatten a :class:`NotEnoughLabels` into
    an ordinary refusal -- losing both the subclass a caller branches on and
    the original wording.
    """
    from spacr import classifier_evaluation as evaluation

    def refuse(*args, **kwargs):
        raise ra.AnnotationStrategyError("the hold-out plate is empty")

    monkeypatch.setattr(evaluation, "grouped_split", refuse)
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(plate))
    assert str(caught.value) == "the hold-out plate is empty"


def test_a_named_well_too_small_to_survive_the_holdout_is_refused(plate):
    """The hold-out is drawn over whole wells before any selection, so a
    thinly-populated guide well can end up with no selectable cell at all.
    Selecting from it anyway chooses inside the group the run is scored on.
    """
    frame = _annotated(plate)
    lone = frame.iloc[[0]].copy()
    lone["rowID"], lone["columnID"] = "r9", "c9"
    lone["prcfo"] = "plate1_r9_c9_f1_o999"
    table = pd.concat([frame, lone], ignore_index=True)
    with pytest.raises(ra.AnnotationStrategyError,
                       match="fewer than two cells left"):
        ra.prepare(_request(table, label_column="ann", wells=["r9_c9"]))


def test_a_one_class_holdout_is_refused_rather_than_scored():
    """A cell-level split of a rare phenotype can hand back a test side with
    no positive in it at all; accuracy there is the prevalence of the
    majority class and says nothing about the model, so it must be refused.
    """
    frame = _plate(wells=8, per_well=10, seed=1).iloc[:20].copy()
    annotations = np.array(["miss"] * len(frame), dtype=object)
    annotations[:2] = "zhit"
    frame["ann"] = annotations
    refusals = []
    for seed in range(8):
        try:
            ra.prepare(_request(frame, label_column="ann", group_by="cell",
                                seed=seed, n_positive=3))
        except ra.AnnotationStrategyError as refusal:
            refusals.append(str(refusal))
    assert any("all of one class" in message for message in refusals), refusals


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


def test_a_result_that_selected_nothing_still_reports_its_holdout(
        prepared, tmp_path):
    """The selection table is empty whenever a strategy queued nothing, and
    the hold-out is still what every number was measured on. Counting or
    writing that frame must be skipped: it has no column to count.
    """
    empty = ra._selection_frame(prepared, {})
    assert list(empty.columns) == ["annotation_role", "annotation_group",
                                   "annotation_reference"]
    assert len(empty) == 0
    result = ra.AnnotationResult(
        strategy="score_strata", title="Score strata",
        selection=empty,
        holdout=ra._selection_frame(prepared, {"holdout": prepared.holdout}),
        predictions=None, notes=("nothing was selected",))
    assert result.role_counts() == {"holdout": int(prepared.holdout.size)}
    written = result.write(str(tmp_path))
    assert set(written) == {"holdout", "report"}
    assert "Score strata" in open(written["report"], encoding="utf-8").read()
    assert len(pd.read_csv(written["holdout"])) == int(prepared.holdout.size)


def test_a_chosen_well_of_three_cells_cannot_supply_a_matched_pair(plate):
    """Three selectable cells cannot give two positives and two contrasts,
    and a set of one positive is not a class. The refusal names both the
    cells available and the ``n_positive`` that cannot be met.
    """
    frame = _annotated(plate)
    few = frame.iloc[:3].copy()
    few["rowID"], few["columnID"] = "r9", "c9"
    few["prcfo"] = [f"plate1_r9_c9_f1_o{i}" for i in range(3)]
    table = pd.concat([frame, few], ignore_index=True)
    with pytest.raises(ra.AnnotationStrategyError, match="cannot supply"):
        ra.run("top_score_random",
               _request(table, label_column="ann", wells=["r9_c9"]))


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


def test_a_fully_annotated_screen_seeds_from_labels_and_queues_nothing(plate):
    """The seed is the difference between a model trained on what a person
    wrote and one trained on the score's own top cells. And when every
    selectable cell is annotated there is nothing left to queue -- a
    refusal, not an empty queue.
    """
    frame = _annotated(plate)
    request = _request(frame, label_column="ann")
    setup = ra.prepare(request)
    assert setup.annotated
    train, _labels, notes = ra._seed_training(setup, request)
    assert train.size == setup.labelled().size
    assert any("annotated cell" in note for note in notes)
    with pytest.raises(ra.AnnotationStrategyError,
                       match="nothing left to queue"):
        ra.run("uncertainty", request, setup)


def test_dropping_every_feature_as_a_score_input_leaves_no_model(plate):
    """Uncertainty sampling ranks cells by a decision boundary. Under
    ``leakage='drop'`` with every measurement column counted as a score
    input, no model is produced at all -- and queueing cells by an absent
    boundary is queueing them at random while calling it uncertainty.
    """
    request = _request(plate, leakage="drop", correlation_cut=0.0)
    setup = ra.prepare(request)
    assert setup.features and not setup.honest_features
    with pytest.raises(ra.NotEnoughLabels, match="no decision boundary"):
        ra.run("uncertainty", request, setup)


def test_a_setup_prepared_without_measurements_refuses_to_cluster(plate):
    """:func:`prepare` relaxes the feature requirement for a strategy that
    fits nothing, so a table with no measurements yields a usable setup
    carrying no feature columns. One setup is shared across strategies, and
    clustering on that one must refuse rather than cluster an empty matrix.
    """
    bare = plate[["plateID", "rowID", "columnID", "fieldID", "prcfo",
                  "pred"]].copy()
    request = _request(bare)
    setup = ra.prepare(request, ra.SCORE_STRATA)
    assert setup.features == ()
    with pytest.raises(ra.AnnotationStrategyError,
                       match="no feature column to cluster on"):
        ra.run("diversity", request, setup)


def test_a_cluster_nobody_landed_in_does_not_break_the_budget(plate):
    """Features that do not separate the cells collapse every object onto
    one point, so k-means returns fewer occupied clusters than asked for and
    the empty ones have no member to queue. The run must still produce a
    queue rather than index into an empty member list.
    """
    frame = plate.copy()
    frame["cell_area"] = 100.0
    result = ra.run("diversity", _request(frame, feature_columns=["cell_area"],
                                          n_clusters=5))
    assert result.counts["clusters"] == 5
    assert result.counts["queued"] == 1
    assert result.counts["smallest_cluster"] == result.counts["largest_cluster"]


def test_the_pu_inner_split_passes_a_typed_refusal_through(plate, monkeypatch):
    """Positive-unlabelled learning splits again inside its training set to
    estimate the labelling rate. A refusal already carrying this module's
    type -- :class:`NotEnoughLabels` in particular -- must not be caught by
    the ``ValueError`` handler below it and re-raised as a plain strategy
    error with a sentence glued on the front.
    """
    from spacr import classifier_evaluation as evaluation

    request = _request(plate)
    setup = ra.prepare(request)

    def refuse(*args, **kwargs):
        raise ra.NotEnoughLabels("the inner split has one class")

    monkeypatch.setattr(evaluation, "grouped_split", refuse)
    with pytest.raises(ra.NotEnoughLabels) as caught:
        ra.run("pu_learning", request, setup)
    assert str(caught.value) == "the inner split has one class"
