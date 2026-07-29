"""Active-learning queue — the maths, the ordering, and the failure modes.

An uncertainty queue that is quietly wrong is indistinguishable from one
that is right: it still returns crops, in some order, and the annotator
labels them. So every number in here is hand-computed in the test that
asserts it, and the cases that would produce a plausible-looking wrong
order are built explicitly:

* a **single-logit binary head** pushed through a softmax collapses to a
  column of 1.0 and destroys the ordering; a **C-logit head** pushed
  through a sigmoid can *invert* it. Both directions are constructed and
  pinned.
* **NULL is unlabelled, 0 is a class** — a queue that re-serves labelled
  crops wastes exactly the resource it exists to save.
* **pure uncertainty collapses onto one region of feature space** — the
  test that proves diversity is worth having builds the case where the
  pure queue is 100 crops from two wells, and asserts the diversified
  one is not.
* **determinism**, including on the ties that dominate a real screen.
* the ranking maths must not need torch.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spacr.active_learning import (
    CALIBRATION_NOTE,
    DEFAULT_MEASURE,
    UNCERTAINTY_MEASURES,
    as_probabilities,
    build_queue,
    disagreement,
    entropy,
    format_queue_summary,
    least_confidence,
    margin,
    predict_probabilities,
    probabilities_from_logits,
    queue_rows,
    rank_by_uncertainty,
    resolve_measure,
    uncertainty_scores,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Database fixtures — png_list as spacr.utils.filepaths_to_database writes
# it, plus the REAL `pred` column that
# spacr.deep_spacr.merge_predictions_into_db adds, plus the INTEGER
# annotation column spacr.qt.annotate_engine.ensure_annotation_column adds.
# ---------------------------------------------------------------------------

def make_db(path, rows, pred_columns=("pred",), annotation="annotate",
            extra_columns=("plateID", "rowID", "columnID", "fieldID")):
    """Write a png_list table.

    ``rows`` is a list of dicts with at least ``png_path``; missing keys
    become NULL, which is what a real half-annotated database looks like.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["png_path"] + list(extra_columns) + list(pred_columns)
    decls = ['"png_path" TEXT'] + [f'"{c}" TEXT' for c in extra_columns] \
        + [f'"{c}" REAL' for c in pred_columns]
    if annotation:
        cols.append(annotation)
        decls.append(f'"{annotation}" INTEGER')
    con = sqlite3.connect(path)
    con.execute(f'CREATE TABLE png_list ({", ".join(decls)})')
    con.executemany(
        f'INSERT INTO png_list ({", ".join(cols)}) '
        f'VALUES ({", ".join("?" * len(cols))})',
        [tuple(r.get(c) for c in cols) for r in rows])
    con.commit()
    con.close()
    return path


def plate_rows(wells, per_well, pred, prefix="p1"):
    """One crop per (well, field), all in plate ``prefix``."""
    out = []
    for w in wells:
        for f in range(per_well):
            out.append({
                "png_path": f"/crops/{prefix}_{w}_1_f{f}_cell.png",
                "plateID": prefix, "rowID": w, "columnID": "1",
                "fieldID": f"f{f}",
                "pred": pred(w, f),
            })
    return out


@pytest.fixture
def simple_db(tmp_path):
    """Four crops, scores 0.5 / 0.6 / 0.9 / 0.99, none annotated."""
    rows = [
        {"png_path": "/c/a.png", "plateID": "p1", "rowID": "r1",
         "columnID": "1", "fieldID": "f1", "pred": 0.50},
        {"png_path": "/c/b.png", "plateID": "p1", "rowID": "r1",
         "columnID": "1", "fieldID": "f2", "pred": 0.60},
        {"png_path": "/c/c.png", "plateID": "p1", "rowID": "r2",
         "columnID": "1", "fieldID": "f1", "pred": 0.90},
        {"png_path": "/c/d.png", "plateID": "p1", "rowID": "r2",
         "columnID": "1", "fieldID": "f2", "pred": 0.99},
    ]
    return make_db(tmp_path / "run" / "measurements" / "measurements.db", rows)


# ===========================================================================
# The measures, against arithmetic
# ===========================================================================

def test_entropy_of_a_fair_coin_is_ln_two():
    assert entropy(np.array([0.5]))[0] == pytest.approx(math.log(2))


def test_entropy_of_a_uniform_three_class_row_is_ln_three():
    row = np.full((1, 3), 1 / 3)
    assert entropy(row)[0] == pytest.approx(math.log(3))


def test_entropy_of_a_one_hot_row_is_exactly_zero():
    """0·log 0 is 0, not NaN — a certain crop is the minimum, not missing."""
    assert entropy(np.array([[1.0, 0.0, 0.0]]))[0] == 0.0
    assert entropy(np.array([0.0]))[0] == 0.0
    assert entropy(np.array([1.0]))[0] == 0.0


def test_entropy_exact_value_for_ninety_ten():
    expected = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert expected == pytest.approx(0.32508297339144836)
    assert entropy(np.array([0.1]))[0] == pytest.approx(expected)


def test_entropy_uniform_is_the_maximum_over_random_rows():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=200)
    probs = np.vstack([probs, np.full((1, 4), 0.25)])
    assert int(np.argmax(entropy(probs))) == len(probs) - 1


def test_entropy_in_bits_and_normalised():
    assert entropy(np.array([0.5]), base=2)[0] == pytest.approx(1.0)
    assert entropy(np.full((1, 4), 0.25), normalize=True)[0] == pytest.approx(1.0)
    assert entropy(np.full((1, 4), 0.25), base=2, normalize=True)[0] \
        == pytest.approx(1.0)


def test_entropy_rejects_a_nonsense_log_base():
    with pytest.raises(ValueError, match="base"):
        entropy(np.array([0.5]), base=1)
    with pytest.raises(ValueError, match="base"):
        entropy(np.array([0.5]), base=0)


def test_least_confidence_bounds():
    assert least_confidence(np.array([[1.0, 0.0, 0.0]]))[0] == 0.0
    assert least_confidence(np.full((1, 3), 1 / 3))[0] == pytest.approx(2 / 3)
    assert least_confidence(np.full((1, 3), 1 / 3), normalize=True)[0] \
        == pytest.approx(1.0)


def test_margin_bounds_and_orientation():
    """margin() returns UNCERTAINTY: 1 at a tie, 0 at a one-hot row."""
    assert margin(np.array([[0.5, 0.5]]))[0] == pytest.approx(1.0)
    assert margin(np.array([[1.0, 0.0]]))[0] == pytest.approx(0.0)
    assert margin(np.array([[0.6, 0.3, 0.1]]))[0] == pytest.approx(1 - 0.3)


def test_margin_is_twice_least_confidence_on_two_classes():
    """So they are ONE choice on a binary screen, not two.

    ``1 − (p₁ − p₂) = 2·(1 − max p)`` exactly; the ranking they produce
    is therefore identical, ties and all.
    """
    p = np.linspace(0.0, 1.0, 41)
    assert margin(p) == pytest.approx(2.0 * least_confidence(p))
    assert rank_by_uncertainty(p, "margin").tolist() == \
        rank_by_uncertainty(p, "least_confidence").tolist()


def test_margin_and_least_confidence_diverge_from_three_classes():
    """The real distinction only appears at C >= 3."""
    probs = np.array([[0.5, 0.5, 0.0],
                      [0.5, 0.25, 0.25]])
    lc = least_confidence(probs)
    mg = margin(probs)
    assert lc[0] == pytest.approx(lc[1])          # same top class mass
    assert mg[0] > mg[1]                          # but the runner-up differs
    assert rank_by_uncertainty(probs, "margin").tolist() == [0, 1]
    assert entropy(probs)[1] > entropy(probs)[0]  # entropy sees all three


def test_every_measure_is_maximal_at_uniform_and_minimal_at_one_hot():
    uniform = np.full((1, 5), 0.2)
    one_hot = np.eye(5)[:1]
    for name, fn in UNCERTAINTY_MEASURES.items():
        assert fn(one_hot)[0] == pytest.approx(0.0), name
        assert fn(uniform)[0] > fn(one_hot)[0], name


def test_measures_accept_one_d_two_d_and_single_column():
    """(N,), (N,1) and (N,2) are the same binary problem."""
    flat = np.array([0.3, 0.7])
    column = flat.reshape(-1, 1)
    pair = np.column_stack([1 - flat, flat])
    for fn in UNCERTAINTY_MEASURES.values():
        assert fn(flat) == pytest.approx(fn(column))
        assert fn(flat) == pytest.approx(fn(pair))


def test_measures_return_nan_for_an_unusable_row():
    scores = np.array([np.nan, 0.5])
    with np.errstate(all="raise"):
        for fn in UNCERTAINTY_MEASURES.values():
            out = fn(scores)
            assert math.isnan(out[0])
            assert np.isfinite(out[1])


def test_measures_on_empty_input():
    for fn in UNCERTAINTY_MEASURES.values():
        assert fn(np.zeros(0)).shape == (0,)
    assert as_probabilities(np.zeros(0)).shape == (0, 2)
    assert rank_by_uncertainty(np.zeros(0)).tolist() == []


def test_a_scalar_is_a_caller_error_not_a_one_crop_queue():
    with pytest.raises(ValueError, match="scalar"):
        entropy(0.5)


def test_three_dimensional_input_is_rejected_and_points_at_disagreement():
    with pytest.raises(ValueError, match="disagreement"):
        entropy(np.zeros((2, 3, 4)))


def test_zero_column_input_is_rejected():
    with pytest.raises(ValueError, match="no columns"):
        entropy(np.zeros((3, 0)))


# ===========================================================================
# Head shape: single logit vs C logits
# ===========================================================================

def test_single_logit_head_uses_a_sigmoid_and_keeps_its_order():
    """A softmax over a (N,1) array returns 1.0 for every crop.

    Row 0 is a confident logit, row 1 is dead on the boundary, so the
    correct queue is [1, 0]. Under the wrong link every score collapses
    to the same value and the order degenerates to row order, [0, 1].
    """
    logits = np.array([[3.0], [0.0]])
    probs = probabilities_from_logits(logits)
    assert probs[:, 1] == pytest.approx([1 / (1 + math.exp(-3.0)), 0.5])
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert rank_by_uncertainty(probs).tolist() == [1, 0]

    wrong = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert wrong.ravel() == pytest.approx([1.0, 1.0])   # information gone
    assert rank_by_uncertainty(wrong).tolist() == [0, 1]  # != the truth


def test_multiclass_head_uses_a_softmax_and_a_sigmoid_would_invert_it():
    """[5, 5] is a perfect tie; a sigmoid of column 1 calls it 0.993.

    Correct order is [tie, near-tie] = [0, 1]. Reading the same logits as
    a single-logit head inverts it to [1, 0].
    """
    logits = np.array([[5.0, 5.0], [0.2, 0.0]])
    probs = probabilities_from_logits(logits)
    assert probs[0] == pytest.approx([0.5, 0.5])
    assert probs[1][1] == pytest.approx(1 / (1 + math.exp(0.2)))
    assert rank_by_uncertainty(probs).tolist() == [0, 1]

    wrong = probabilities_from_logits(logits[:, 1])      # pretend single-logit
    assert rank_by_uncertainty(wrong).tolist() == [1, 0]
    assert rank_by_uncertainty(probs).tolist() != rank_by_uncertainty(wrong).tolist()


def test_three_class_logits_softmax_to_a_distribution():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = probabilities_from_logits(logits)
    assert probs.shape == (2, 3)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probs[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert entropy(probs)[1] > entropy(probs)[0]


def test_softmax_does_not_overflow_on_large_logits():
    probs = probabilities_from_logits(np.array([[1000.0, 999.0]]))
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0)


def test_logits_from_a_duck_typed_tensor_need_no_torch_import():
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr
            self.detached = False

        def detach(self):
            self.detached = True
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    t = FakeTensor(np.array([[0.0], [4.0]]))
    probs = probabilities_from_logits(t)
    assert t.detached
    assert probs[0] == pytest.approx([0.5, 0.5])


# ===========================================================================
# as_probabilities — stored scores, not logits
# ===========================================================================

def test_stored_single_column_is_read_as_the_positive_class():
    probs = as_probabilities(np.array([0.25]))
    assert probs[0] == pytest.approx([0.75, 0.25])


def test_unnormalised_rows_are_renormalised():
    probs = as_probabilities(np.array([[2.0, 2.0], [1.0, 3.0]]))
    assert probs[0] == pytest.approx([0.5, 0.5])
    assert probs[1] == pytest.approx([0.25, 0.75])


def test_impossible_rows_become_nan_rather_than_a_wrong_order():
    probs = as_probabilities(np.array([[-1.0, 2.0], [0.0, 0.0], [0.4, 0.6]]))
    assert np.all(np.isnan(probs[0]))       # negatives are not probabilities
    assert np.all(np.isnan(probs[1]))       # no mass at all
    assert probs[2] == pytest.approx([0.4, 0.6])


def test_a_single_column_outside_zero_one_is_treated_as_unusable():
    probs = as_probabilities(np.array([1.7, 0.3, -2.0]))
    assert np.all(np.isnan(probs[0]))
    assert np.all(np.isnan(probs[2]))
    assert probs[1] == pytest.approx([0.7, 0.3])


# ===========================================================================
# disagreement
# ===========================================================================

def test_identical_members_disagree_about_nothing():
    p = np.array([0.5, 0.9])
    assert disagreement([p, p, p]) == pytest.approx([0.0, 0.0])


def test_disagreement_grows_with_spread():
    tight = disagreement([np.array([0.50]), np.array([0.55])])
    wide = disagreement([np.array([0.10]), np.array([0.90])])
    assert wide[0] > tight[0] > 0.0


def test_disagreement_accepts_a_stacked_three_d_array():
    stack = np.array([[[0.5, 0.5]], [[0.1, 0.9]]])
    assert disagreement(stack)[0] > 0.0


def test_disagreement_accepts_a_stacked_tensor_without_importing_torch():
    """MC-dropout passes usually arrive as one stacked tensor."""
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    stack = FakeTensor(np.array([[[0.5, 0.5]], [[0.1, 0.9]]]))
    assert disagreement(stack)[0] > 0.0


def test_a_single_score_set_cannot_disagree_with_itself():
    out = disagreement([np.array([0.5, 0.9])])
    assert out == pytest.approx([0.0, 0.0])
    assert disagreement(np.array([0.5, 0.9])) == pytest.approx([0.0, 0.0])


def test_bald_ignores_crops_the_members_agree_are_ambiguous():
    """The distinction ensembles exist to make.

    Crop 0: every member says 50/50 — genuinely ambiguous data, a label
    will not help the model. Crop 1: the members split 0.05 / 0.95 —
    the model class is undecided, and a label settles it. Entropy of the
    mean ranks them equally; BALD does not.
    """
    a = np.array([0.5, 0.05])
    b = np.array([0.5, 0.95])
    mean_probs = as_probabilities((a + b) / 2)
    assert entropy(mean_probs)[0] == pytest.approx(entropy(mean_probs)[1])
    bald = disagreement([a, b], method="bald")
    assert bald[0] == pytest.approx(0.0, abs=1e-12)
    # ln 2 − H(0.05): the mean is a coin flip, each member is nearly sure.
    h_member = -(0.05 * math.log(0.05) + 0.95 * math.log(0.95))
    assert bald[1] == pytest.approx(math.log(2) - h_member)
    assert bald[1] > 0.49


def test_disagreement_is_nan_where_any_member_is_unusable():
    out = disagreement([np.array([np.nan, 0.5]), np.array([0.2, 0.5])])
    assert math.isnan(out[0])
    assert np.isfinite(out[1])


def test_misaligned_members_are_refused_not_averaged():
    with pytest.raises(ValueError, match="same crops"):
        disagreement([np.zeros(3), np.zeros(4)])


def test_disagreement_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="variance"):
        disagreement([np.zeros(3), np.ones(3)], method="wibble")


def test_disagreement_needs_at_least_one_member():
    with pytest.raises(ValueError, match="at least one"):
        disagreement([])


def test_disagreement_is_not_in_the_measure_registry():
    """Its signature takes a LIST of score sets, so it cannot be swapped
    in behind ``measure=`` without silently mis-reading its input."""
    assert "disagreement" not in UNCERTAINTY_MEASURES
    assert set(UNCERTAINTY_MEASURES) == {
        "least_confidence", "margin", "entropy"}
    assert DEFAULT_MEASURE in UNCERTAINTY_MEASURES


# ===========================================================================
# Ranking
# ===========================================================================

def test_rank_puts_the_most_uncertain_first():
    assert rank_by_uncertainty(np.array([0.99, 0.5, 0.8])).tolist() == [1, 2, 0]


def test_rank_honours_limit():
    assert rank_by_uncertainty(np.array([0.99, 0.5, 0.8]), limit=2).tolist() \
        == [1, 2]
    assert rank_by_uncertainty(np.array([0.99, 0.5]), limit=0).tolist() == []


def test_ties_break_on_row_index_and_never_move_between_runs():
    probs = np.full(50, 0.5)
    first = rank_by_uncertainty(probs).tolist()
    assert first == list(range(50))
    for _ in range(5):
        assert rank_by_uncertainty(probs).tolist() == first


def test_a_seed_makes_tie_breaking_reproducible_not_random():
    probs = np.full(50, 0.5)
    a = rank_by_uncertainty(probs, seed=7).tolist()
    b = rank_by_uncertainty(probs, seed=7).tolist()
    assert a == b
    assert sorted(a) == list(range(50))
    assert a != rank_by_uncertainty(probs, seed=8).tolist()


def test_a_seed_never_overrides_the_scores_themselves():
    """Only ties are shuffled: a more uncertain crop always sorts first."""
    probs = np.array([0.99, 0.5, 0.5, 0.5, 0.01])
    order = rank_by_uncertainty(probs, seed=3)
    assert set(order[:3].tolist()) == {1, 2, 3}
    assert set(order[3:].tolist()) == {0, 4}


def test_unusable_scores_sort_last_and_stay_in_the_permutation():
    probs = np.array([np.nan, 0.9, 0.5, np.nan])
    order = rank_by_uncertainty(probs).tolist()
    assert order[:2] == [2, 1]
    assert sorted(order[2:]) == [0, 3]


def test_rank_accepts_precomputed_scores():
    order = rank_by_uncertainty(None, scores=np.array([0.1, 0.9, 0.5]))
    assert order.tolist() == [1, 2, 0]


def test_unknown_measure_name_lists_the_real_ones():
    with pytest.raises(ValueError) as exc:
        uncertainty_scores(np.array([0.5]), "confidence")
    assert "entropy" in str(exc.value)
    assert "least_confidence" in str(exc.value)


def test_a_callable_measure_is_accepted():
    name, fn = resolve_measure(lambda p: np.zeros(len(np.atleast_1d(p))))
    assert callable(fn)
    order = rank_by_uncertainty(np.array([0.9, 0.1]),
                                measure=lambda p: np.array([0.0, 1.0]))
    assert order.tolist() == [1, 0]
    assert resolve_measure("entropy")[0] == "entropy"
    assert resolve_measure(entropy)[0] == "entropy"


# ===========================================================================
# build_queue — exclusion of annotated crops
# ===========================================================================

def test_queue_is_ordered_most_uncertain_first(simple_db):
    q = build_queue(simple_db, "annotate", diversity="none")
    assert q["png_path"].tolist() == ["/c/a.png", "/c/b.png", "/c/c.png",
                                      "/c/d.png"]
    assert q["rank"].tolist() == [1, 2, 3, 4]
    assert q["uncertainty"].is_monotonic_decreasing


def test_annotated_crops_never_appear_and_zero_counts_as_annotated(tmp_path):
    """NULL is unlabelled; 0 is a real class, not a falsy blank.

    Reading 0 as "not yet labelled" would re-serve every negative on the
    plate — the majority of crops on a real screen.
    """
    rows = [
        {"png_path": "/c/unlabelled.png", "pred": 0.5},                # NULL
        {"png_path": "/c/labelled_zero.png", "pred": 0.5, "annotate": 0},
        {"png_path": "/c/labelled_one.png", "pred": 0.5, "annotate": 1},
        {"png_path": "/c/labelled_two.png", "pred": 0.5, "annotate": 2},
    ]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate", diversity="none")
    assert q["png_path"].tolist() == ["/c/unlabelled.png"]
    meta = q.attrs["spacr_active_learning"]
    assert meta["n_annotated"] == 3
    assert meta["n_unlabelled"] == 1
    assert meta["labelled_class_balance"] == {0: 1, 1: 1, 2: 1}


def test_a_second_annotation_column_has_its_own_queue(tmp_path):
    rows = [{"png_path": f"/c/{i}.png", "pred": 0.5} for i in range(4)]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    con = sqlite3.connect(db)
    con.execute('ALTER TABLE png_list ADD COLUMN "bob" INTEGER')
    con.execute('UPDATE png_list SET annotate = 1 WHERE png_path = "/c/0.png"')
    con.execute('UPDATE png_list SET bob = 1 WHERE png_path = "/c/1.png"')
    con.commit()
    con.close()
    assert "/c/0.png" not in build_queue(db, "annotate")["png_path"].tolist()
    assert "/c/0.png" in build_queue(db, "bob")["png_path"].tolist()
    assert "/c/1.png" not in build_queue(db, "bob")["png_path"].tolist()


def test_a_database_with_no_annotation_column_queues_everything(simple_db):
    q = build_queue(simple_db, "someone_new")
    assert len(q) == 4
    meta = q.attrs["spacr_active_learning"]
    assert meta["annotation_column_present"] is False
    assert any("has no 'someone_new' column" in n for n in meta["notes"])
    assert "nothing has been annotated" in " ".join(meta["notes"])


def test_everything_annotated_gives_an_empty_queue_that_says_so(tmp_path):
    rows = plate_rows(["A01", "A02"], 3, lambda w, f: 0.5)
    for i, r in enumerate(rows):
        r["annotate"] = i % 2
    db = make_db(tmp_path / "m" / "measurements.db", rows)
    q = build_queue(db, "annotate", diversity="well")
    assert len(q) == 0
    assert list(q.columns)                    # still a usable frame
    text = format_queue_summary(q)
    assert "EMPTY" in text
    assert "already annotated" in text
    assert "6" in text
    assert queue_rows(q) == []


def test_no_unlabelled_crop_can_be_scored_is_its_own_explanation(tmp_path):
    rows = [{"png_path": f"/c/{i}.png"} for i in range(3)]     # NULL pred
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate")
    assert len(q) == 0
    text = format_queue_summary(q)
    assert "no unlabelled crop has a usable score" in text
    assert "3 unscorable" in text


def test_an_empty_png_list_says_run_measure_first(tmp_path):
    db = make_db(tmp_path / "m" / "measurements.db", [], extra_columns=())
    q = build_queue(db, "annotate")
    assert len(q) == 0
    assert "save_png" in format_queue_summary(q)


# ===========================================================================
# build_queue — missing / broken score columns
# ===========================================================================

def test_no_prediction_column_is_explained_not_guessed(tmp_path):
    db = make_db(tmp_path / "m" / "measurements.db",
                 [{"png_path": "/c/a.png"}], pred_columns=(),
                 extra_columns=())
    with pytest.raises(ValueError) as exc:
        build_queue(db, "annotate")
    msg = str(exc.value)
    assert "no prediction column" in msg
    assert "merge_predictions_into_db" in msg


def test_an_explicitly_named_missing_column_is_named_back(simple_db):
    with pytest.raises(ValueError, match="'nope'"):
        build_queue(simple_db, "annotate", pred_column="nope")


def test_a_missing_table_and_a_missing_file_are_distinguished(tmp_path):
    db = str(tmp_path / "empty.db")
    sqlite3.connect(db).close()
    with pytest.raises(ValueError, match="has no 'png_list' table"):
        build_queue(db, "annotate")
    with pytest.raises(FileNotFoundError, match="No such database"):
        build_queue(str(tmp_path / "nope.db"), "annotate")
    with pytest.raises(ValueError, match="No database path"):
        build_queue("   ", "annotate")


def test_a_table_without_png_path_is_refused(tmp_path):
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (other TEXT, pred REAL)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="png_path"):
        build_queue(db, "annotate")


def test_null_and_nan_scores_are_dropped_counted_and_explained(tmp_path):
    rows = [
        {"png_path": "/c/a.png", "pred": 0.5},
        {"png_path": "/c/b.png"},                       # NULL pred
        {"png_path": "/c/c.png", "pred": float("nan")},
        {"png_path": "/c/d.png", "pred": 0.95},
    ]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate", diversity="none")
    assert q["png_path"].tolist() == ["/c/a.png", "/c/d.png"]
    meta = q.attrs["spacr_active_learning"]
    assert meta["n_unscorable"] == 2
    assert any("no usable score" in n for n in meta["notes"])
    assert "no usable score" in format_queue_summary(q)


def test_a_pred_column_holding_logits_is_caught(tmp_path):
    """Values outside [0, 1] mean the merge wrote logits, not probabilities."""
    rows = [{"png_path": "/c/a.png", "pred": 0.5},
            {"png_path": "/c/b.png", "pred": 4.2},
            {"png_path": "/c/c.png", "pred": -3.1}]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate")
    assert q["png_path"].tolist() == ["/c/a.png"]
    notes = " ".join(q.attrs["spacr_active_learning"]["notes"])
    assert "outside [0, 1]" in notes
    assert "probabilities_from_logits" in notes


def test_unnormalised_multiclass_columns_are_renormalised_with_a_note(tmp_path):
    rows = [{"png_path": "/c/a.png", "pred_0": 2.0, "pred_1": 2.0,
             "pred_2": 0.0},
            {"png_path": "/c/b.png", "pred_0": 0.9, "pred_1": 0.05,
             "pred_2": 0.05}]
    db = make_db(tmp_path / "m" / "measurements.db", rows,
                 pred_columns=("pred_0", "pred_1", "pred_2"),
                 extra_columns=())
    q = build_queue(db, "annotate", diversity="none")
    assert q["png_path"].tolist() == ["/c/a.png", "/c/b.png"]
    assert q["uncertainty"].iloc[0] == pytest.approx(math.log(2))
    notes = " ".join(q.attrs["spacr_active_learning"]["notes"])
    assert "did not sum to 1" in notes
    assert "renormalised" in notes


def test_multiclass_columns_are_auto_detected_in_numeric_order(tmp_path):
    rows = [{"png_path": "/c/a.png", "pred_0": 0.2, "pred_1": 0.3,
             "pred_2": 0.5}]
    db = make_db(tmp_path / "m" / "measurements.db", rows,
                 pred_columns=("pred_0", "pred_1", "pred_2"),
                 extra_columns=())
    q = build_queue(db, "annotate")
    meta = q.attrs["spacr_active_learning"]
    assert meta["pred_columns"] == ["pred_0", "pred_1", "pred_2"]
    assert meta["n_classes"] == 3
    assert q["predicted_class"].tolist() == [2]


def test_an_explicit_column_list_beats_auto_detection(tmp_path):
    rows = [{"png_path": "/c/a.png", "pred_0": 0.2, "pred_1": 0.8,
             "other": 0.5}]
    db = make_db(tmp_path / "m" / "measurements.db", rows,
                 pred_columns=("pred_0", "pred_1", "other"),
                 extra_columns=())
    q = build_queue(db, "annotate", pred_column=["other"])
    assert q.attrs["spacr_active_learning"]["pred_columns"] == ["other"]
    assert q["uncertainty"].iloc[0] == pytest.approx(math.log(2))


def test_an_empty_pred_column_list_is_a_caller_error(simple_db):
    with pytest.raises(ValueError, match="empty list"):
        build_queue(simple_db, "annotate", pred_column=[])


def test_a_single_class_output_is_flagged_rather_than_ranked(tmp_path):
    """Every crop scoring the same means the ordering says nothing."""
    rows = [{"png_path": f"/c/{i}.png", "pred": 1.0} for i in range(5)]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate")
    assert len(q) == 5
    assert q["uncertainty"].tolist() == [0.0] * 5
    notes = " ".join(q.attrs["spacr_active_learning"]["notes"])
    assert "scored exactly 0" in notes
    assert "single distinct value" in notes
    assert q["png_path"].tolist() == [f"/c/{i}.png" for i in range(5)]


# ===========================================================================
# Diversity — the reason this feature is worth having
# ===========================================================================

def _two_hot_wells_db(tmp_path):
    """Two wells of dead-boundary crops, eight wells of merely-uncertain.

    This is the shape of a real plate: whatever the model finds confusing
    is confined to a couple of wells, so a pure uncertainty ranking hands
    the annotator a hundred near-copies of the same ambiguity.
    """
    hot = ["A01", "A02"]
    cold = [f"B{i:02d}" for i in range(1, 9)]
    rows = plate_rows(hot, 60, lambda w, f: 0.5)
    rows += plate_rows(cold, 60, lambda w, f: 0.62)
    return make_db(tmp_path / "m" / "measurements.db", rows)


def test_pure_uncertainty_returns_a_hundred_crops_from_two_wells(tmp_path):
    db = _two_hot_wells_db(tmp_path)
    q = build_queue(db, "annotate", limit=100, diversity="none")
    assert len(q) == 100
    assert set(q["rowID"]) == {"A01", "A02"}


def test_the_diversified_queue_spreads_across_every_well(tmp_path):
    """Same data, same measure — the queue stops being two wells deep."""
    db = _two_hot_wells_db(tmp_path)
    q = build_queue(db, "annotate", limit=100, diversity="well")
    assert len(q) == 100
    assert len(set(q["rowID"])) == 10
    counts = q["rowID"].value_counts()
    assert counts.max() == 10                 # 100 crops / 10 wells
    # The single most uncertain crop is still first — diversity reorders
    # what follows, it does not throw away the top of the ranking.
    assert q["rowID"].iloc[0] in {"A01", "A02"}
    assert q["uncertainty"].iloc[0] == pytest.approx(math.log(2))


def test_diversity_costs_average_uncertainty_and_the_summary_shows_it(tmp_path):
    db = _two_hot_wells_db(tmp_path)
    pure = build_queue(db, "annotate", limit=100, diversity="none")
    div = build_queue(db, "annotate", limit=100, diversity="well")
    assert div["uncertainty"].mean() < pure["uncertainty"].mean()
    assert "Spread: 10 distinct" in format_queue_summary(div)
    # Spread is reported whatever the strategy — it is most informative
    # exactly here, where the pure queue has collapsed onto two wells.
    pure_text = format_queue_summary(pure)
    assert "Spread: 2 distinct" in pure_text
    assert "Diversity is OFF" in pure_text


def test_field_and_plate_strata_group_differently(tmp_path):
    rows = plate_rows(["A01"], 5, lambda w, f: 0.5, prefix="p1")
    rows += plate_rows(["A01"], 5, lambda w, f: 0.5, prefix="p2")
    db = make_db(tmp_path / "m" / "measurements.db", rows)
    by_plate = build_queue(db, "annotate", limit=4, diversity="plate")
    assert by_plate["plateID"].tolist() == ["p1", "p2", "p1", "p2"]
    by_field = build_queue(db, "annotate", limit=4, diversity="field")
    assert len(set(by_field["fieldID"])) == 4
    for name in ("row", "column"):
        assert len(build_queue(db, "annotate", limit=4, diversity=name)) == 4


def test_custom_group_columns_can_diversify_over_a_cluster_id(tmp_path):
    """The docstring's alternative: round-robin over a feature cluster.

    ``cluster`` is not one of the crop-metadata columns, so this also
    pins that an arbitrary user column is carried into the queue frame.
    """
    rows = [{"png_path": f"/c/{i}.png", "cluster": f"k{i % 3}", "pred": 0.5}
            for i in range(6)]
    db = make_db(tmp_path / "m" / "measurements.db", rows,
                 extra_columns=("cluster",))
    q = build_queue(db, "annotate", limit=3, group_columns=["cluster"])
    assert sorted(q["cluster"]) == ["k0", "k1", "k2"]
    meta = q.attrs["spacr_active_learning"]
    assert meta["diversity"] == "custom"
    assert meta["diversity_columns"] == ["cluster"]
    assert "Spread: 3 distinct cluster groups" in format_queue_summary(q)


def test_custom_group_columns_that_are_not_there_fall_back(tmp_path):
    rows = [{"png_path": f"/c/{i}.png", "pred": 0.5} for i in range(3)]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate", group_columns=["cluster"])
    assert q.attrs["spacr_active_learning"]["diversity"] == "none"
    assert len(q) == 3


def test_missing_plate_metadata_falls_back_to_pure_uncertainty(tmp_path):
    rows = [{"png_path": f"/c/{i}.png", "pred": 0.5} for i in range(3)]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate", diversity="well")
    meta = q.attrs["spacr_active_learning"]
    assert meta["diversity"] == "none"
    assert any("pure uncertainty" in n for n in meta["notes"])
    assert len(q) == 3


def test_partial_plate_metadata_diversifies_over_what_is_there(tmp_path):
    rows = [{"png_path": f"/c/{i}.png", "plateID": "p1",
             "rowID": f"r{i % 2}", "pred": 0.5} for i in range(4)]
    db = make_db(tmp_path / "m" / "measurements.db", rows,
                 extra_columns=("plateID", "rowID"))
    q = build_queue(db, "annotate", diversity="well")
    meta = q.attrs["spacr_active_learning"]
    assert meta["diversity_columns"] == ["plateID", "rowID"]
    assert any("fell back to" in n for n in meta["notes"])


def test_an_unknown_diversity_strategy_is_refused(simple_db):
    with pytest.raises(ValueError, match="Unknown diversity"):
        build_queue(simple_db, "annotate", diversity="wibble")


def test_diversity_can_be_switched_off_by_several_names(simple_db):
    for off in (None, False, "none", "off"):
        q = build_queue(simple_db, "annotate", diversity=off)
        assert q.attrs["spacr_active_learning"]["diversity"] == "none"
    for on in (True, "auto"):
        q = build_queue(simple_db, "annotate", diversity=on)
        assert q.attrs["spacr_active_learning"]["diversity"] == "well"


# ===========================================================================
# Determinism, limits, filters
# ===========================================================================

def test_the_same_database_and_seed_give_the_same_queue_every_time(tmp_path):
    """Ties dominate: 40 crops all scoring exactly 0.5."""
    rows = plate_rows(["A01", "A02", "A03", "A04"], 10, lambda w, f: 0.5)
    db = make_db(tmp_path / "m" / "measurements.db", rows)
    runs = [build_queue(db, "annotate", seed=11)["png_path"].tolist()
            for _ in range(4)]
    assert all(r == runs[0] for r in runs)
    unseeded = [build_queue(db, "annotate")["png_path"].tolist()
                for _ in range(3)]
    assert all(r == unseeded[0] for r in unseeded)
    assert sorted(runs[0]) == sorted(unseeded[0])


def test_limit_truncates_after_diversification(tmp_path):
    db = _two_hot_wells_db(tmp_path)
    q = build_queue(db, "annotate", limit=5, diversity="well")
    assert len(q) == 5
    assert len(set(q["rowID"])) == 5
    assert build_queue(db, "annotate", limit=0)["png_path"].tolist() == []


def test_image_type_filters_the_pool_like_the_annotate_screen(tmp_path):
    rows = [{"png_path": "/c/x_cell.png", "pred": 0.5},
            {"png_path": "/c/x_nucleus.png", "pred": 0.5}]
    db = make_db(tmp_path / "m" / "measurements.db", rows, extra_columns=())
    q = build_queue(db, "annotate", image_type="nucleus")
    assert q["png_path"].tolist() == ["/c/x_nucleus.png"]
    assert "excluded 1 of 2" in " ".join(
        q.attrs["spacr_active_learning"]["notes"])
    # A filter that excludes nothing says nothing.
    everything = build_queue(db, "annotate", image_type="/c/")
    assert len(everything) == 2
    assert not any("excluded" in n for n in
                   everything.attrs["spacr_active_learning"]["notes"])


def test_queue_rows_matches_the_annotate_screens_page_shape(simple_db):
    q = build_queue(simple_db, "annotate", limit=2, diversity="none")
    rows = queue_rows(q)
    assert rows == [("/c/a.png", None), ("/c/b.png", None)]
    assert all(isinstance(p, str) and a is None for p, a in rows)
    with pytest.raises(ValueError, match="png_path"):
        queue_rows(q.drop(columns=["png_path"]))


def test_measure_choice_changes_the_multiclass_order(tmp_path):
    rows = [{"png_path": "/c/a.png", "pred_0": 0.5, "pred_1": 0.5,
             "pred_2": 0.0},
            {"png_path": "/c/b.png", "pred_0": 0.5, "pred_1": 0.25,
             "pred_2": 0.25}]
    db = make_db(tmp_path / "m" / "measurements.db", rows,
                 pred_columns=("pred_0", "pred_1", "pred_2"),
                 extra_columns=())
    by_margin = build_queue(db, "annotate", measure="margin",
                            diversity="none")["png_path"].tolist()
    by_entropy = build_queue(db, "annotate", measure="entropy",
                             diversity="none")["png_path"].tolist()
    assert by_margin == ["/c/a.png", "/c/b.png"]
    assert by_entropy == ["/c/b.png", "/c/a.png"]


def test_building_a_queue_never_writes_to_the_database(simple_db):
    before = os.path.getmtime(simple_db)
    build_queue(simple_db, "annotate")
    con = sqlite3.connect(simple_db)
    assert con.execute("SELECT COUNT(*) FROM png_list").fetchone()[0] == 4
    con.close()
    assert os.path.getmtime(simple_db) == before


# ===========================================================================
# Class balance + summary
# ===========================================================================

def test_class_balance_of_an_imbalanced_screen_is_reported(tmp_path):
    """98 % negative: the queue skews to the majority class's boundary.

    Reported rather than hidden, so the annotator can see it.
    """
    rows = [{"png_path": f"/c/n{i}.png", "plateID": "p1", "rowID": "A01",
             "columnID": "1", "fieldID": f"f{i}", "pred": 0.40}
            for i in range(98)]
    rows += [{"png_path": f"/c/p{i}.png", "plateID": "p1", "rowID": "A02",
              "columnID": "1", "fieldID": f"f{i}", "pred": 0.95}
             for i in range(2)]
    db = make_db(tmp_path / "m" / "measurements.db", rows)
    con = sqlite3.connect(db)
    con.execute('UPDATE png_list SET annotate = 0 WHERE png_path = "/c/n0.png"')
    con.commit()
    con.close()
    q = build_queue(db, "annotate", limit=10, diversity="none")
    meta = q.attrs["spacr_active_learning"]
    assert meta["queue_class_balance"] == {0: 10}
    assert meta["pool_class_balance"] == {0: 97, 1: 2}
    assert meta["labelled_class_balance"] == {0: 1}
    text = format_queue_summary(q)
    assert "Predicted-class balance" in text
    assert "imbalanced screen" in text
    assert "Already annotated: class 0: 1" in text


def test_a_real_valued_annotation_column_still_reports_whole_classes(tmp_path):
    """Some databases carry the label as REAL; 1.0 is class 1, not '1.0'."""
    rows = [{"png_path": "/c/a.png", "pred": 0.5, "annotate": 1.0},
            {"png_path": "/c/b.png", "pred": 0.5}]
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (png_path TEXT, pred REAL, "
                "annotate REAL)")
    con.executemany("INSERT INTO png_list VALUES (?,?,?)",
                    [(r["png_path"], r["pred"], r.get("annotate"))
                     for r in rows])
    con.commit()
    con.close()
    q = build_queue(db, "annotate")
    assert q["png_path"].tolist() == ["/c/b.png"]
    balance = q.attrs["spacr_active_learning"]["labelled_class_balance"]
    assert balance == {1: 1}
    assert [type(k) for k in balance] == [int]      # "class 1", not "class 1.0"
    assert "Already annotated: class 1: 1" in format_queue_summary(q)


def test_summary_reports_every_count_that_makes_the_queue_readable(tmp_path):
    db = _two_hot_wells_db(tmp_path)
    text = format_queue_summary(build_queue(db, "annotate", limit=20, seed=1))
    for fragment in ("Active-learning queue", "Measure: entropy",
                     "diversity: well", "seed: 1", "scores from: pred",
                     "Crops:", "600 total", "20 queued", "Uncertainty score:",
                     "Spread:", "Predicted-class balance"):
        assert fragment in text, fragment


def test_summary_never_calls_a_score_a_confidence(tmp_path):
    """Calibration honesty: no '87 % sure' anywhere, ever."""
    db = _two_hot_wells_db(tmp_path)
    text = format_queue_summary(build_queue(db, "annotate", limit=10))
    assert CALIBRATION_NOTE in text
    assert "not calibrated probabilities" in text
    # The caveat itself quotes the phrasing it forbids, so check the rest.
    body = text.replace(CALIBRATION_NOTE, "").lower()
    for banned in ("% sure", "confident", "confidence", "certainty of",
                   "probability that"):
        assert banned not in body, banned
    assert "uncertainty" in body


def test_summary_survives_losing_its_attrs(simple_db):
    """A sliced or round-tripped frame must still render something true."""
    q = build_queue(simple_db, "annotate")
    stripped = pd.DataFrame(q.to_dict("list"))
    assert not stripped.attrs
    text = format_queue_summary(stripped)
    assert "Active-learning queue" in text
    assert CALIBRATION_NOTE in text


def test_summary_omits_what_a_mangled_frame_can_no_longer_support():
    """Better a short summary than an invented one."""
    bare = pd.DataFrame({"png_path": ["/c/a.png", "/c/b.png"]})
    text = format_queue_summary(bare)
    assert "Active-learning queue" in text
    assert CALIBRATION_NOTE in text
    body = text.replace(CALIBRATION_NOTE, "")
    assert "Uncertainty score:" not in body
    assert "Predicted-class balance" not in body
    assert "Spread:" not in body

    all_nan = pd.DataFrame({"png_path": ["/c/a.png"],
                            "uncertainty": [float("nan")]})
    nan_text = format_queue_summary(all_nan)
    assert "Uncertainty score:" not in nan_text.replace(CALIBRATION_NOTE, "")


def test_module_docstring_states_the_calibration_caveat():
    import spacr.active_learning as mod
    doc = mod.__doc__.lower()
    assert "not calibrated" in CALIBRATION_NOTE.lower()
    assert "softmax is not a probability" in doc
    assert "monotone" in margin.__doc__ or "linear transform" in margin.__doc__
    assert "cost" in build_queue.__doc__.lower()


# ===========================================================================
# The live-model bridge — mocked, never a real model
# ===========================================================================

class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTorch:
    """Just enough torch for predict_probabilities, without importing it."""

    def __init__(self):
        self.entered = 0

    def no_grad(self):
        self.entered += 1
        return _FakeNoGrad()


def test_predict_probabilities_scores_batches_under_no_grad(monkeypatch):
    fake = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    batches = [np.array([[0.0], [3.0]]), np.array([[-3.0]])]
    probs = predict_probabilities(lambda x: x, batches)
    assert fake.entered == 1
    assert probs.shape == (3, 2)
    assert probs[0] == pytest.approx([0.5, 0.5])
    assert probs[2, 1] == pytest.approx(1 / (1 + math.exp(3.0)))


def test_predict_probabilities_works_when_torch_is_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # `import torch` raises
    probs = predict_probabilities(lambda x: x, [np.array([[0.0], [3.0]])])
    assert probs.shape == (2, 2)


def test_predict_probabilities_takes_the_inputs_out_of_a_loader_batch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    batches = [(np.array([[0.0]]), np.array([1]), "meta")]
    assert predict_probabilities(lambda x: x, batches).shape == (1, 2)


def test_predict_probabilities_puts_a_module_in_eval_mode(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    class Model:
        def __init__(self):
            self.evalled = False
            self.moved = None

        def eval(self):
            self.evalled = True

        def to(self, device):
            self.moved = device
            return self

        def __call__(self, x):
            return np.asarray(x, dtype=float)

    model = Model()
    predict_probabilities(model, [np.array([[0.0]])], device="cpu")
    assert model.evalled and model.moved == "cpu"

    no_device = Model()
    predict_probabilities(no_device, [np.array([[0.0]])])
    assert no_device.evalled and no_device.moved is None


def test_predict_probabilities_moves_tensor_inputs_to_the_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    class Batch:
        def __init__(self, arr):
            self._arr = arr
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    batch = Batch(np.array([[0.0]]))
    probs = predict_probabilities(lambda x: x, [batch], device="cuda:0")
    assert batch.device == "cuda:0"
    assert probs[0] == pytest.approx([0.5, 0.5])


def test_predict_probabilities_on_no_batches_and_on_stored_probabilities(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert predict_probabilities(lambda x: x, []).shape == (0, 2)
    probs = predict_probabilities(lambda x: x, [np.array([[0.25]])],
                                  from_logits=False)
    assert probs[0] == pytest.approx([0.75, 0.25])


def test_predict_probabilities_refuses_inconsistent_head_widths(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    batches = [np.zeros((2, 1)), np.zeros((2, 3))]
    with pytest.raises(ValueError, match="head width"):
        predict_probabilities(lambda x: x, batches)


# ===========================================================================
# Dependency weight
# ===========================================================================

def test_the_ranking_maths_works_with_torch_absent_from_sys_modules():
    """Blocked torch, in a fresh interpreter: the maths must not care.

    ``sys.modules['torch'] = None`` makes any ``import torch`` raise, so
    a stray import anywhere on the ranking path fails loudly here rather
    than costing the Qt screen a multi-second import in production.
    """
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import math, numpy as np\n"
        "from spacr.active_learning import (entropy, least_confidence, margin,\n"
        "    disagreement, probabilities_from_logits, rank_by_uncertainty,\n"
        "    as_probabilities)\n"
        "try:\n"
        "    import torch\n"
        "    raise SystemExit('torch was importable; the block failed')\n"
        "except ImportError:\n"
        "    pass\n"
        "assert abs(entropy(np.array([0.5]))[0] - math.log(2)) < 1e-12\n"
        "assert least_confidence(np.array([0.5]))[0] == 0.5\n"
        "assert margin(np.array([0.5]))[0] == 1.0\n"
        "assert disagreement([np.array([0.1]), np.array([0.9])])[0] > 0\n"
        "p = probabilities_from_logits(np.array([[0.0], [5.0]]))\n"
        "assert rank_by_uncertainty(p).tolist() == [0, 1]\n"
        "assert as_probabilities(np.array([0.25]))[0][1] == 0.25\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_importing_the_module_does_not_drag_in_torch_or_cellpose():
    """Measured as modules ADDED by the import, so a sitecustomize that
    pre-imports torch cannot make this pass or fail by accident."""
    code = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import spacr.active_learning\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    added = json.loads(proc.stdout.strip().splitlines()[-1])
    tops = {name.split(".")[0] for name in added}
    for heavy in ("torch", "cellpose", "torchvision", "cv2", "matplotlib"):
        assert heavy not in tops, (
            f"importing spacr.active_learning dragged in {heavy!r}; the "
            f"ranking maths must stay numpy/pandas/sqlite only")


def test_torch_is_only_ever_imported_inside_a_function():
    import spacr.active_learning as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for line in source.splitlines():
        if line.lstrip().startswith("import torch"):
            assert line.startswith(" "), (
                f"module-level torch import: {line!r}")
