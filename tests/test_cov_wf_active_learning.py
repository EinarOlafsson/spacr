"""spacr.active_learning — the four branches the loop suites never take.

Every path here is the *quiet* half of a pair that the existing suites
already drive from the loud side:

* ``annotation_coverage`` warns when its ``image_type`` filter threw crops
  away; nothing pinned what it must **not** say when the filter kept every
  crop, and a warning that fires on an empty exclusion teaches annotators
  to ignore it;
* ``should_stop`` appends "the held-out set is too small to resolve this"
  to a flat verdict; the suites only ever fed it a held-out set that was
  in fact too small, so nothing held it to *withholding* that sentence
  when the plateau is measurable;
* ``retrain_round(write_scores=False)`` is the documented way to fit a
  round without touching the database, and nothing checked that it
  actually leaves ``png_list`` alone;
* ``_predict_proba`` pads classes a fold never saw, and drops class ids
  that fall outside the round's class list — only the padding half was
  pinned.

Each test drives both sides of its pair in one function, so an assertion
about something being absent is always paired with the input that
produces it.

No network, no GPU, no torch: sklearn on two dozen synthetic rows.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al


# ---------------------------------------------------------------------------
# A minimal screen: png_list plus a feature matrix handed in directly, so the
# tests below exercise the round and not the measurement join (which
# tests/test_cov_active_learning_rounds.py already pins object by object).
# ---------------------------------------------------------------------------

WELLS = (("r1", "c1"), ("r1", "c2"), ("r2", "c1"), ("r2", "c2"))


def _screen(tmp_path, per_well=6, plate="plate1"):
    """``<tmp>/measurements.db`` with 24 labelled crops over four wells.

    Classes alternate within each well so neither the grouped split nor the
    class balance is degenerate, and ``area`` carries the class signal so a
    fitted model has something real to separate.
    """
    db = os.path.join(str(tmp_path), "measurements.db")
    rng = np.random.default_rng(7)
    crops, features = [], {}
    for well_index, (row_id, column_id) in enumerate(WELLS):
        for obj in range(1, per_well + 1):
            prcf = f"{plate}_{row_id}_{column_id}_f1"
            cls = obj % 2
            path = f"/crops/cell_png/{prcf}_o{obj}.png"
            crops.append({
                "png_path": path, "prcfo": f"{prcf}_o{obj}",
                "plateID": plate, "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "annotate": cls,
            })
            features[path] = {
                "cell_area": 500.0 + 400.0 * cls + float(rng.normal(0, 5)),
                "cell_noise": float(rng.normal(0, 1)),
            }
    columns = ["png_path", "prcfo", "plateID", "rowID", "columnID",
               "fieldID", "annotate"]
    con = sqlite3.connect(db)
    try:
        con.execute(
            "CREATE TABLE png_list ("
            + ", ".join(f'"{c}" ' + ("INTEGER" if c == "annotate" else "TEXT")
                        for c in columns) + ")")
        con.executemany(
            f'INSERT INTO png_list ({", ".join(columns)}) '
            f'VALUES ({", ".join("?" * len(columns))})',
            [tuple(row[c] for c in columns) for row in crops])
        con.commit()
    finally:
        con.close()
    matrix = pd.DataFrame.from_dict(features, orient="index")
    matrix.index.name = "png_path"
    return {"db": db, "crops": crops, "features": matrix}


def _png_list_columns(db):
    con = sqlite3.connect(db)
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(png_list)")]
    finally:
        con.close()


def _curve(accuracies, new_labels=30, n_holdout=200):
    """A learning-curve frame with a chosen held-out accuracy per round."""
    n = len(accuracies)
    return pd.DataFrame({
        "round": list(range(n)),
        "finished_utc": [""] * n,
        "n_labels": list(np.cumsum([new_labels] * n)),
        "n_new_labels": [new_labels] * n,
        "n_holdout": [n_holdout] * n,
        "holdout_accuracy": list(accuracies),
        "holdout_f1_macro": list(accuracies),
        "per_class": [{}] * n,
        "split_rule": [""] * n,
        "model_type": [""] * n,
        "model_path": [""] * n,
        "card_path": [""] * n,
        "measure": [""] * n,
        "diversity": [""] * n,
        "notes": [[]] * n,
        "gain": pd.Series(list(accuracies)).diff(),
    })


# ---------------------------------------------------------------------------
# annotation_coverage — the filter note fires only when the filter cut
# ---------------------------------------------------------------------------

def test_a_filter_that_kept_every_crop_does_not_claim_it_excluded_any(
        tmp_path):
    """A filter note that appears when nothing was filtered is noise.

    ``image_type`` is the Annotate screen's own view filter, and coverage
    prints a note saying how many crops it removed so the denominator can
    be trusted. The screen passes the filter on every call, including the
    common one where the database holds a single crop mode and the filter
    removes nothing. If the note appeared there too, an annotator would
    read "excluded 0 of 24" on every report and stop reading the notes —
    including on the run where the filter really did halve the population.
    """
    screen = _screen(tmp_path)

    kept_all = al.annotation_coverage(screen["db"], "annotate",
                                      image_type="cell_png")
    meta = kept_all.attrs["spacr_annotation_coverage"]
    assert meta["n_rows"] == 24
    assert meta["n_rows_unfiltered"] == 24
    assert meta["n_annotated"] == 24
    assert meta["by_class"] == {"0": 12, "1": 12}
    assert [note for note in meta["notes"] if "excluded" in note] == []

    # The same call on a filter that DOES cut, so the absence above is a
    # property of the empty exclusion and not of a note nobody ever writes.
    half = al.annotation_coverage(screen["db"], "annotate", image_type="_r1_")
    half_meta = half.attrs["spacr_annotation_coverage"]
    assert half_meta["n_rows"] == 12
    assert half_meta["n_rows_unfiltered"] == 24
    assert half_meta["by_class"] == {"0": 6, "1": 6}
    excluded = [note for note in half_meta["notes"] if "excluded" in note]
    assert len(excluded) == 1
    assert "excluded 12 of 24 crops" in excluded[0]
    assert "describes the 12 that matched" in excluded[0]


# ---------------------------------------------------------------------------
# should_stop — "flat" and "unmeasurable" are different sentences
# ---------------------------------------------------------------------------

def test_a_measurable_plateau_is_not_excused_as_held_out_noise(tmp_path):
    """A real plateau must not be softened with a standard-error caveat.

    The stopping rule says "stop" for both a plateau it can measure and one
    that is merely below the noise floor, but the two call for different
    work: the first means more labels of this kind are not buying accuracy,
    the second means the held-out set is too small to tell. If the caveat
    about a small held-out set were attached to every flat verdict, an
    annotator with 20 000 held-out objects would be told to go and collect
    more held-out data they already have — and would keep annotating.
    """
    accuracies = [0.90, 0.90, 0.9025]

    measurable = al.should_stop(_curve(accuracies, n_holdout=20000),
                                label_window=50, min_gain=0.003)
    assert measurable.stop is True
    assert measurable.trend == "flat"
    assert measurable.gain == pytest.approx(0.0025)
    assert measurable.confident is True
    assert measurable.noise == pytest.approx(
        float(np.sqrt(0.9025 * 0.0975 / 20000)))
    assert "not buying measurable accuracy" in measurable.reason
    assert "standard error" not in measurable.reason
    assert "unmeasurable" not in measurable.reason

    # Identical curve, smaller held-out set: the same 0.25 % is now inside
    # the noise, and the caveat the verdict above withheld appears.
    unmeasurable = al.should_stop(_curve(accuracies, n_holdout=200),
                                  label_window=50, min_gain=0.003)
    assert unmeasurable.stop is True
    assert unmeasurable.trend == "flat"
    assert unmeasurable.gain == pytest.approx(0.0025)
    assert unmeasurable.confident is False
    assert "held-out set is only 200 objects" in unmeasurable.reason
    assert "unmeasurable, not that it is provably zero" in unmeasurable.reason


# ---------------------------------------------------------------------------
# retrain_round — write_scores=False keeps its hands off the database
# ---------------------------------------------------------------------------

def test_a_round_told_not_to_write_scores_leaves_the_crop_table_alone(
        tmp_path):
    """``write_scores=False`` is how a round is fitted without side effects.

    It is what a script uses to evaluate a model type, a seed or a split on
    somebody else's database, and what the screen uses when it wants a
    number without re-ranking the queue underneath the annotator. If it
    wrote the probability columns anyway, a trial fit would silently
    re-order the crops the annotator is working through, and
    ``build_queue`` — which prefers the al_prob_ columns over ``pred`` —
    would rank on a model nobody chose to adopt.
    """
    screen = _screen(tmp_path)

    dry = al.retrain_round(screen["db"], "annotate",
                           features=screen["features"], write_scores=False,
                           save_model=False, write_card=False, round_index=0)
    assert dry.score_columns == []
    assert dry.scored == 0
    assert dry.n_labels == 24
    assert "Re-scored 0 crops into nothing" in dry.summary()
    columns_after_dry = _png_list_columns(screen["db"])
    assert [c for c in columns_after_dry
            if c.startswith(al.ROUND_PRED_PREFIX)] == []

    # Same call with the writing turned back on, so the emptiness above is
    # the flag doing its job rather than a round that scores nothing.
    wet = al.retrain_round(screen["db"], "annotate",
                           features=screen["features"], write_scores=True,
                           save_model=False, write_card=False, round_index=1)
    assert wet.score_columns == [f"{al.ROUND_PRED_PREFIX}0",
                                 f"{al.ROUND_PRED_PREFIX}1"]
    assert wet.scored == 24
    columns_after = _png_list_columns(screen["db"])
    assert [c for c in columns_after if c.startswith(al.ROUND_PRED_PREFIX)] == \
        [f"{al.ROUND_PRED_PREFIX}0", f"{al.ROUND_PRED_PREFIX}1"]

    con = sqlite3.connect(screen["db"])
    try:
        rows = con.execute(
            f'SELECT "{al.ROUND_PRED_PREFIX}0", "{al.ROUND_PRED_PREFIX}1" '
            f"FROM png_list").fetchall()
    finally:
        con.close()
    assert len(rows) == 24
    assert all(p0 is not None and p1 is not None for p0, p1 in rows)
    assert all(p0 + p1 == pytest.approx(1.0) for p0, p1 in rows)


# ---------------------------------------------------------------------------
# _predict_proba — a class id outside the round's class list is dropped
# ---------------------------------------------------------------------------

def test_a_class_the_round_does_not_know_is_dropped_not_written_elsewhere():
    """An out-of-range class id must never land in another class's column.

    ``_predict_proba`` re-indexes an estimator's ``predict_proba`` columns
    by the estimator's own ``classes_``, and the estimator is not always the
    one this round fitted: a model reloaded from an earlier round can carry
    a class the current label set no longer has. Writing that column
    positionally would put the retired class's probability into the column
    ``build_queue`` reads as class 1, so the queue would rank crops by a
    class that is not on the annotator's screen. Dropping it is the safe
    answer, and the columns that remain must still be the right ones.
    """
    class _ThreeClassModel:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, x):
            return np.tile([0.5, 0.2, 0.3], (len(x), 1))

    out = al._predict_proba(_ThreeClassModel(), np.zeros((4, 2)), 2)
    assert out.shape == (4, 2)
    assert out[:, 0] == pytest.approx(0.5)
    assert out[:, 1] == pytest.approx(0.2), "class 2 did not overwrite class 1"
    # The retired class's 0.3 is gone rather than folded into a live class.
    assert out.sum(axis=1) == pytest.approx(0.7)

    # The in-range counterpart: every class the model reports is kept, and
    # the class the fold never saw is the one that comes back zero.
    class _PartialModel:
        classes_ = np.array([0, 2])

        def predict_proba(self, x):
            return np.tile([0.4, 0.6], (len(x), 1))

    padded = al._predict_proba(_PartialModel(), np.zeros((3, 2)), 3)
    assert padded.shape == (3, 3)
    assert padded[:, 0] == pytest.approx(0.4)
    assert padded[:, 1] == pytest.approx(0.0)
    assert padded[:, 2] == pytest.approx(0.6)
    assert padded.sum(axis=1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# should_stop — a curve that never recorded a held-out size still gets a
# verdict, and gets it without a standard error it cannot compute
# ---------------------------------------------------------------------------

def test_a_curve_with_no_recorded_holdout_size_still_returns_a_flat_verdict():
    """A missing ``n_holdout`` must not cost the annotator the stopping rule.

    ``noise`` is a binomial standard error, and it needs the size of the
    held-out set to exist. Rounds recorded by an older spaCR — or by a
    caller that assembled the curve by hand from ``holdout_accuracy`` alone
    — carry no ``n_holdout``, so ``noise`` is ``None``. The flat branch then
    has to decide whether to attach the "the held-out set is too small"
    caveat while holding no number to justify it, and the only honest answer
    is to withhold it: quoting a standard error of zero would tell the
    annotator the plateau is proven when nothing measured it.

    The ordering of that guard is the part worth pinning. ``noise is not
    None`` has to be tested *before* ``abs(gain) <= noise``, because
    comparing a float to ``None`` raises ``TypeError`` in Python 3. A
    refactor that swaps the two halves — or replaces them with
    ``abs(gain) <= (noise or 0)`` — turns "no held-out size recorded" from a
    usable verdict into a crash, or into a confident claim of convergence.
    """
    accuracies = [0.90, 0.90, 0.9025]

    sizeless = al.should_stop(_curve(accuracies, n_holdout=0),
                              label_window=50, min_gain=0.003)
    assert sizeless.stop is True
    assert sizeless.trend == "flat"
    assert sizeless.gain == pytest.approx(0.0025)
    assert sizeless.noise is None
    # No held-out size means no basis for calling the plateau confident.
    assert sizeless.confident is False
    assert "not buying measurable accuracy" in sizeless.reason
    assert "standard error" not in sizeless.reason
    assert "unmeasurable" not in sizeless.reason

    # The same curve WITH a held-out size, so the withheld caveat above is
    # the missing number doing its job and not a sentence nobody writes.
    sized = al.should_stop(_curve(accuracies, n_holdout=200),
                           label_window=50, min_gain=0.003)
    assert sized.stop is True
    assert sized.trend == "flat"
    assert sized.noise == pytest.approx(float(np.sqrt(0.9025 * 0.0975 / 200)))
    assert "held-out set is only 200 objects" in sized.reason
    assert "standard error" in sized.reason


# ---------------------------------------------------------------------------
# _predict_proba — a NEGATIVE class id is dropped by the same guard that
# drops an over-large one
# ---------------------------------------------------------------------------

def test_a_negative_class_id_is_dropped_and_does_not_index_from_the_end():
    """``classes_ = [-1, ...]`` must not write into the LAST class column.

    ``-1`` is the standard scikit-learn marker for "unlabelled" or "outlier"
    — ``LabelSpreading``, ``LabelPropagation`` and the novelty detectors all
    emit it — and a model handed to a round is not always the one the round
    fitted: ``retrain_round`` can be pointed at an estimator reloaded from an
    earlier round, or at a semi-supervised fit over the annotator's partial
    labels. The re-indexing loop writes ``out[:, int(class_id)]``, and
    ``out[:, -1]`` is perfectly legal NumPy: it silently writes the
    unlabelled column's probability into the round's HIGHEST class. The
    queue would then rank crops as most-likely-class-N on the strength of a
    score that means "no label", which is the opposite of informative.

    The lower half of the range guard is what prevents that, and it is a
    separate half from the upper bound: a test that only feeds an oversized
    class id leaves ``0 <= class_id`` unpinned.
    """
    class _SemiSupervisedModel:
        classes_ = np.array([-1, 0, 1])

        def predict_proba(self, x):
            return np.tile([0.7, 0.1, 0.2], (len(x), 1))

    out = al._predict_proba(_SemiSupervisedModel(), np.zeros((5, 2)), 2)
    assert out.shape == (5, 2)
    assert out[:, 0] == pytest.approx(0.1)
    # The 0.7 belonging to class -1 is gone, NOT wrapped onto the last
    # column: had it wrapped, class 1 would read 0.7 instead of 0.2.
    assert out[:, 1] == pytest.approx(0.2), "class -1 wrapped onto class 1"
    assert out.sum(axis=1) == pytest.approx(0.3)

    # The counterpart runs the SAME re-indexing loop with every id in range
    # — ``classes_`` out of order also defeats the fast path — so the missing
    # 0.7 above is the negative id being rejected and not the loop dropping
    # columns generally.
    class _ReorderedModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, x):
            return np.tile([0.3, 0.7], (len(x), 1))

    kept = al._predict_proba(_ReorderedModel(), np.zeros((5, 2)), 2)
    assert kept[:, 0] == pytest.approx(0.7), "classes_ order was ignored"
    assert kept[:, 1] == pytest.approx(0.3)
    assert kept.sum(axis=1) == pytest.approx(1.0)
