"""What the annotation strategies refuse, and what they say while refusing.

:mod:`spacr.regression_annotation` is the code path between "annotate some
cells" and a number a biologist will believe. Every test here pins a place
where the module has to stop and explain itself rather than carry on with a
degenerate answer: a score column that is present but almost empty, a
hold-out that swallowed the wells the user named, a labelling rate of zero
that the positive-unlabelled correction would otherwise divide by, a
clustering asked for more clusters than the cells have distinct positions.

The counterpart matters as much: each refusal is checked next to an input
that does NOT trigger it, so a guard that started refusing everything --
which would look exactly like a passing test if only the refusal were
asserted -- is caught here too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


from spacr import regression_annotation as ra


def _screen(*, wells: int = 8, per_well: int = 8, seed: int = 0,
            partial: bool = False, flat: bool = False,
            tiny: int = 0) -> pd.DataFrame:
    """A small plate: two annotated classes per well, a score, measurements.

    ``partial`` leaves the later cells of every well unannotated, ``flat``
    adds two columns that never vary, and ``tiny`` adds a well holding only
    a handful of cells.
    """
    rng = np.random.default_rng(seed)

    def cell(row_name: str, column_name: str, index: int) -> dict:
        hit = index % 2 == 0
        row = {
            "plateID": "plate1", "rowID": row_name,
            "columnID": column_name, "fieldID": "f1",
            "cell_area": float(rng.normal(900 + 300 * hit, 40)),
            "cell_perimeter": float(rng.normal(120 + 25 * hit, 6)),
            "pred": float(rng.uniform(0.4, 0.6) + 0.3 * hit),
            "label": ("hit" if hit else "control")
                     if (not partial or index < 4) else "",
        }
        if flat:
            row["flat_a"], row["flat_b"] = 7.0, 3.0
        return row

    rows = [cell(f"r{1 + well // 4}", f"c{1 + well % 4}", index)
            for well in range(wells) for index in range(per_well)]
    rows.extend(cell("r3", "c1", index) for index in range(tiny))
    return pd.DataFrame(rows)


def _request(frame: pd.DataFrame, **overrides) -> ra.AnnotationRequest:
    """A request over ``frame`` with every well eligible."""
    values = dict(frame=frame, score_column="pred", label_column="label",
                  wells=[], n_positive=4, holdout_fraction=0.25, seed=1,
                  score_inputs=[])
    values.update(overrides)
    return ra.AnnotationRequest(**values)


@pytest.fixture(scope="module")
def annotated():
    """A prepared run whose reference labels are the human annotations."""
    request = _request(_screen())
    return ra.prepare(request, ra.UNCERTAINTY), request


class _Constant:
    """An estimator stand-in that answers the same probability every time."""

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def fit(self, matrix, labels):
        return self

    def predict_proba(self, matrix):
        column = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1.0 - column, column])


def test_a_present_but_empty_score_column_is_counted_not_denied():
    """A user who loaded a score column and still cannot run needs to know
    it is there and nearly all missing; being told the column does not exist
    would send them to rebuild a table they already have."""
    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_o{i}" for i in range(10)],
        "cell_area": np.linspace(800, 1000, 10),
        "pred": [0.1, 0.9, 0.5] + [np.nan] * 7,
    })
    said = ra.missing_requirement(ra.TOP_SCORE_RANDOM.key, frame,
                                  score_column="pred", label_column="")
    assert "only 3 of 10 cell(s) carry a finite 'pred'" in said
    assert "no annotation column is named" in said

    absent = ra.missing_requirement(ra.TOP_SCORE_RANDOM.key, frame,
                                    score_column="nope", label_column="")
    assert "there is no column 'nope'" in absent


def test_a_request_refuses_an_empty_table_a_lone_positive_and_a_whole_plate():
    """These three settings each produce a run that cannot mean anything --
    nothing to select from, a positive class of one cell, a hold-out that is
    the entire screen -- and each has to be named before a fit is attempted,
    not after a stack trace out of scikit-learn."""
    frame = _screen(wells=4, per_well=4)
    with pytest.raises(ra.AnnotationStrategyError, match="no cells to annotate"):
        _request(pd.DataFrame(columns=frame.columns)).validated()
    with pytest.raises(ra.AnnotationStrategyError, match="n_positive"):
        _request(frame, n_positive=1).validated()
    with pytest.raises(ra.AnnotationStrategyError, match="holdout_fraction"):
        _request(frame, holdout_fraction=1.0).validated()
    with pytest.raises(ra.AnnotationStrategyError, match="not one of"):
        _request(frame, leakage="ignore").validated()
    assert _request(frame).validated().n_positive == 4


def test_three_annotated_cells_are_not_enough_to_be_annotations():
    """The chooser and the run must agree about whether a table has labels;
    if the chooser counted three annotations as usable, the strategy would
    be offered and then refuse when it was clicked."""
    frame = pd.DataFrame({"label": ["hit", "control", "hit", None, ""],
                          "cell_area": np.linspace(800, 900, 5)})
    assert ra.usable_annotations(frame, "label") == 0
    wider = pd.concat([frame, frame], ignore_index=True)
    assert ra.usable_annotations(wider, "label") == 6


def test_a_well_name_made_only_of_separators_matches_no_row():
    """Well names come from a plate map a user typed. A name that carries no
    identity token at all must select nothing rather than everything: the
    all-True fallback is reserved for naming no wells, which means "the whole
    screen"."""
    groups = np.array(["plate1\x1fr1\x1fc1", "plate1\x1fr1\x1fc2"],
                      dtype=object)
    assert not ra.wells_selected(groups, ["__"]).any()
    assert ra.wells_selected(groups, ["r1_c1"]).tolist() == [True, False]
    assert ra.wells_selected(groups, []).all()


def test_a_score_that_is_missing_or_flat_defines_no_phenotype():
    """Without annotations the reference label is a cut on the score. A
    score that is entirely missing, or that gives every cell the same value,
    produces a "positive" class holding every cell or none -- a fit against
    it would report perfect accuracy on a distinction nobody made."""
    frame = _screen(wells=2, per_well=4).drop(columns=["label"])
    mask = np.ones(len(frame), dtype=bool)

    blank = frame.copy()
    blank["pred"] = np.nan
    with pytest.raises(ra.AnnotationStrategyError, match="Every value of"):
        ra._reference_labels(blank, _request(blank, label_column=""), mask)

    flat = frame.copy()
    flat["pred"] = 1.0
    with pytest.raises(ra.AnnotationStrategyError,
                       match="puts every cell on one side"):
        ra._reference_labels(flat, _request(flat, label_column=""), mask)

    labels, known, source, threshold, _ = ra._reference_labels(
        frame, _request(frame, label_column=""), mask)
    assert set(np.unique(labels)) == {0, 1} and known.all()
    assert "cut on" in source and np.isfinite(threshold)


def test_no_feature_column_makes_an_empty_matrix_not_a_crash():
    """Strategies that fit nothing run on tables with no measurements at
    all. The matrix builders are called with the empty column list on that
    path, and both must return an empty matrix of the right height rather
    than raising out of pandas."""
    frame = _screen(wells=2, per_well=4)
    assert ra._matrix(frame, []).shape == (len(frame), 0)
    assert ra._standardised(frame, []).shape == (len(frame), 0)
    filled = ra._standardised(frame, ["cell_area", "cell_perimeter"])
    assert filled.shape == (len(frame), 2)
    assert np.isfinite(filled).all()


def test_a_fit_refuses_one_cell_and_refuses_one_class(annotated):
    """A classifier fitted on a single cell, or on cells that all carry the
    same label, has nothing to separate; scikit-learn's own message for that
    names neither the selection nor what to change, and the hold-out numbers
    that followed would be meaningless."""
    prepared, request = annotated
    with pytest.raises(ra.NotEnoughLabels, match="Fewer than two cells"):
        ra._fit_report(prepared, [int(prepared.selectable[0])],
                       prepared.features, request)
    one_class = np.flatnonzero(prepared.labels == 0)[:4]
    with pytest.raises(ra.NotEnoughLabels, match="same"):
        ra._fit_report(prepared, one_class, prepared.features, request)
    both = np.concatenate([np.flatnonzero(prepared.labels == 0)[:6],
                           np.flatnonzero(prepared.labels == 1)[:6]])
    report, estimator, _ = ra._fit_report(prepared, both,
                                          prepared.features, request)
    assert estimator is not None and report.n_train == both.size


def test_a_model_applied_to_every_cell_it_was_fitted_on_predicts_nothing(
        annotated):
    """The prediction table is the model's answer for the cells it was NOT
    fitted on. When there are none, an empty table would read as "the model
    predicted nothing anywhere"; the run reports no prediction table at
    all."""
    prepared, _ = annotated
    everything = np.arange(len(prepared.frame))
    assert ra._apply_model(prepared, _Constant(0.7), prepared.features,
                           everything) is None
    rest = ra._apply_model(prepared, _Constant(0.7), prepared.features,
                           everything[:5])
    assert len(rest) == len(prepared.frame) - 5
    assert set(rest["predicted"].unique()) == {1}


def test_the_share_of_positives_among_no_cells_is_not_zero(annotated):
    """The contrast set's positive share is quoted in the report. An empty
    set has no share, and returning 0.0 would read as "none of them are
    positive" -- a claim about cells that do not exist."""
    prepared, _ = annotated
    assert np.isnan(prepared.positive_share([]))
    assert 0.0 <= prepared.positive_share(prepared.holdout) <= 1.0


def test_uncertainty_needs_a_model_and_an_unqueued_cell(annotated):
    """Uncertainty sampling ranks cells by how close the seed model puts
    them to its boundary. With every feature discarded as a score input
    there is no model to be uncertain with, and with every selectable cell
    already annotated there is nothing left to queue -- both are the user's
    setup, so both are named rather than returning an empty queue."""
    prepared, request = annotated
    with pytest.raises(ra.AnnotationStrategyError,
                       match="already in the seed set"):
        ra._run_uncertainty(prepared, request)

    frame = _screen()
    blind = _request(frame, leakage="drop",
                     score_inputs=["cell_area", "cell_perimeter"])
    with pytest.raises(ra.NotEnoughLabels, match="No model could be fitted"):
        ra._run_uncertainty(ra.prepare(blind, ra.UNCERTAINTY), blind)


def test_a_labelling_rate_of_zero_is_refused_not_divided_by(monkeypatch):
    """Positive-unlabelled learning divides the model's output by the
    estimated labelling rate. A model that gives every held-out positive a
    probability of zero yields a rate of zero; dividing by it would produce
    infinities that clip to a hold-out called entirely positive, which is
    the opposite of what the data said."""
    frame = _screen()
    request = _request(frame, label_column="", n_positive=8)
    prepared = ra.prepare(request, ra.PU_LEARNING)

    monkeypatch.setattr(ra, "_estimator",
                        lambda model, seed: (_Constant(0.0), "stub"))
    with pytest.raises(ra.AnnotationStrategyError,
                       match="estimated labelling rate is 0"):
        ra._run_pu_learning(prepared, request)

    monkeypatch.undo()
    result = ra._run_pu_learning(prepared, request)
    assert result.counts["labelling_rate"] > 0.0
    assert result.predictions is not None


