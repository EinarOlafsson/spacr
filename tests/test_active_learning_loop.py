"""Closing the active-learning loop (C4) and annotation coverage (C5).

The queue already existed. What did not was any of the things that make it a
*loop*: retraining on the labels so far, re-ranking with the retrained model,
a curve you can watch flatten, a rule that tells you when to stop, and a
record of which round each label came from.

Every number below is constructed so that a broken implementation cannot
accidentally pass:

* the lopsided coverage case is built with a known 27-of-30-from-one-well
  split, and the counts are asserted per class *and* per well, not just in
  total;
* the re-rank test builds a screen where the round-1 model and the round-2
  model disagree by construction, so "the queue changed" cannot be an
  artefact of the labelled crops dropping out;
* the stopping rule is fed a flat curve and a rising curve with identical
  shapes otherwise, so it must be reading the accuracy and not the row count.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al


# ---------------------------------------------------------------------------
# Fixtures — a png_list as spacr writes it
# ---------------------------------------------------------------------------

def make_db(path, rows, annotation="annotate", extra_pred=True):
    """Write a png_list table with metadata and (optionally) a `pred` column.

    ``rows`` is a list of dicts with ``png_path`` and whatever metadata the
    test cares about.
    """
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    columns = ["png_path", "plateID", "rowID", "columnID", "fieldID", "prcfo",
               "cell_id"]
    decls = [f'"{c}" TEXT' for c in columns]
    if extra_pred:
        columns.append("pred")
        decls.append('"pred" REAL')
    columns.append(annotation)
    decls.append(f'"{annotation}" INTEGER')
    con = sqlite3.connect(str(path))
    con.execute(f'CREATE TABLE png_list ({", ".join(decls)})')
    con.executemany(
        f'INSERT INTO png_list ({", ".join(columns)}) '
        f'VALUES ({", ".join("?" * len(columns))})',
        [tuple(row.get(c) for c in columns) for row in rows])
    con.commit()
    con.close()


@pytest.fixture
def screen(tmp_path):
    """A two-plate screen: 2 plates x 2 rows x 3 columns x 10 crops = 120."""
    db = tmp_path / "measurements" / "measurements.db"
    rng = np.random.default_rng(0)
    rows, features = [], []
    i = 0
    for plate in ("p1", "p2"):
        for row_id in ("r1", "r2"):
            for column in ("c1", "c2", "c3"):
                for _ in range(10):
                    path = f"/crops/cell_{i:04d}.png"
                    # a latent class: even index -> class 1, odd -> class 2
                    latent = 1 if i % 2 == 0 else 2
                    rows.append({
                        "png_path": path, "plateID": plate, "rowID": row_id,
                        "columnID": column, "fieldID": "f1",
                        "prcfo": f"{plate}_{row_id}_{column}_f1_o{i}",
                        "cell_id": f"o{i}",
                        "pred": 0.5, "annotate": None})
                    features.append({
                        "png_path": path,
                        "signal": float(latent) + rng.normal(0, 0.25),
                        "noise": rng.normal(0, 1.0),
                        "latent": latent})
                    i += 1
    make_db(db, rows)
    frame = pd.DataFrame(features).set_index("png_path")
    return {"db": str(db), "rows": rows, "latent": frame.pop("latent"),
            "features": frame, "tmp": tmp_path}


def label(db, paths_and_classes):
    """Write labels straight into png_list, as the save worker would."""
    con = sqlite3.connect(db)
    con.executemany("UPDATE png_list SET annotate=? WHERE png_path=?",
                    [(int(v), p) for p, v in paths_and_classes.items()])
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# C5 — annotation coverage
# ---------------------------------------------------------------------------

def test_coverage_counts_a_deliberately_lopsided_set(screen):
    """27 of 30 labels from one well is the case this exists to catch."""
    paths = [r["png_path"] for r in screen["rows"]]
    # p1/r1/c1 is rows 0..9; p1/r1/c2 is 10..19; p1/r1/c3 is 20..29
    labels = {}
    for i in range(10):                       # the whole first well
        labels[paths[i]] = 1 if i % 2 == 0 else 2
    for i in range(10, 27):                   # most of the second
        labels[paths[i]] = 1 if i % 2 == 0 else 2
    for i in (30, 60, 90):                    # one crop from three other wells
        labels[paths[i]] = 1
    label(screen["db"], labels)

    coverage = al.annotation_coverage(screen["db"], "annotate")
    meta = coverage.attrs["spacr_annotation_coverage"]

    assert meta["n_annotated"] == 30
    assert meta["n_rows"] == 120
    assert meta["wells_total"] == 12
    # three whole/partial wells on plate 1 plus the three singletons
    assert meta["wells_annotated"] == 6
    assert meta["plates_annotated"] == 2

    # per class, exactly
    expected_class_1 = sum(1 for p, v in labels.items() if v == 1)
    expected_class_2 = sum(1 for p, v in labels.items() if v == 2)
    assert meta["by_class"] == {"1": expected_class_1, "2": expected_class_2}

    # per well, exactly — the whole point
    assert meta["by_well"]["p1/r1/c1"] == 10
    assert meta["by_well"]["p1/r1/c2"] == 10
    assert meta["by_well"]["p1/r1/c3"] == 7
    assert meta["by_well"]["p1/r2/c1"] == 1

    # per plate
    assert meta["by_plate"] == {"p1": 28, "p2": 2}

    # the frame itself is one row per (well, class)
    first_well = coverage[(coverage["plateID"] == "p1")
                          & (coverage["rowID"] == "r1")
                          & (coverage["columnID"] == "c1")]
    assert int(first_well["n"].sum()) == 10
    assert set(first_well["class"]) == {"1", "2"}
    assert float(coverage["share"].sum()) == pytest.approx(1.0)


def test_coverage_flags_a_class_that_came_from_one_well(screen):
    """A class labelled only in one well cannot be told apart from the well."""
    paths = [r["png_path"] for r in screen["rows"]]
    labels = {paths[i]: 1 for i in range(12)}       # all of class 1, one well+
    labels.update({paths[i]: 2 for i in range(30, 42)})
    label(screen["db"], labels)

    coverage = al.annotation_coverage(screen["db"], "annotate")
    meta = coverage.attrs["spacr_annotation_coverage"]
    conc = meta["concentration"]["1"]
    assert conc["n"] == 12
    assert conc["top"] == "p1/r1/c1"
    assert conc["top_n"] == 10
    assert conc["top_share"] == pytest.approx(10 / 12)
    assert conc["n_groups"] == 2
    # Herfindahl: (10/12)^2 + (2/12)^2
    assert conc["hhi"] == pytest.approx((10 / 12) ** 2 + (2 / 12) ** 2)
    assert any("one well" in note for note in meta["notes"])

    text = al.format_coverage_summary(coverage)
    assert "Annotation coverage" in text
    assert "p1/r1/c1" in text
    assert "Busiest wells" in text


def test_coverage_is_empty_but_explicit_before_anything_is_annotated(screen):
    coverage = al.annotation_coverage(screen["db"], "annotate")
    meta = coverage.attrs["spacr_annotation_coverage"]
    assert meta["n_annotated"] == 0
    assert meta["by_class"] == {}
    assert any("Nothing is annotated" in n for n in meta["notes"])
    assert "Nothing annotated yet" in al.format_coverage_summary(coverage)


def test_coverage_refuses_a_column_that_does_not_exist(screen):
    with pytest.raises(ValueError, match="has no 'nope' column"):
        al.annotation_coverage(screen["db"], "nope")


def test_coverage_attributes_labels_to_the_round_that_produced_them(screen):
    paths = [r["png_path"] for r in screen["rows"]]
    first = {paths[i]: 1 for i in range(5)}
    second = {paths[i]: 2 for i in range(50, 58)}
    label(screen["db"], {**first, **second})
    al.record_labels(screen["db"], "annotate", first, 0, source="manual")
    al.record_labels(screen["db"], "annotate", second, 1, source="queue")

    meta = al.annotation_coverage(
        screen["db"], "annotate").attrs["spacr_annotation_coverage"]
    assert meta["by_round"] == {0: 5, 1: 8}
    assert meta["by_source"] == {"manual": 5, "queue": 8}


# ---------------------------------------------------------------------------
# Round provenance
# ---------------------------------------------------------------------------

def test_record_labels_keeps_the_first_round_across_a_correction(screen):
    path = screen["rows"][0]["png_path"]
    al.record_labels(screen["db"], "annotate", {path: 1}, 0)
    al.record_labels(screen["db"], "annotate", {path: 2}, 3)
    rounds = al.label_rounds(screen["db"], "annotate")
    assert len(rounds) == 1
    assert int(rounds["round"].iloc[0]) == 3, "the label it has now"
    assert int(rounds["first_round"].iloc[0]) == 0, "when it was first seen"


def test_record_labels_records_a_cleared_label_rather_than_dropping_it(screen):
    path = screen["rows"][0]["png_path"]
    assert al.record_labels(screen["db"], "annotate", {path: None}, 0) == 1
    assert len(al.label_rounds(screen["db"], "annotate")) == 1


def test_next_round_starts_at_zero_and_follows_the_log(screen):
    assert al.next_round(screen["db"], "annotate") == 0
    al.record_round(screen["db"], "annotate", 0, n_labels=10,
                    holdout_accuracy=0.8)
    assert al.next_round(screen["db"], "annotate") == 1
    al.record_round(screen["db"], "annotate", 1, n_labels=40,
                    holdout_accuracy=0.85)
    assert al.next_round(screen["db"], "annotate") == 2
    # a different column has its own counter
    assert al.next_round(screen["db"], "other") == 0


def test_next_round_on_a_missing_database_is_zero_not_an_exception(tmp_path):
    assert al.next_round(str(tmp_path / "nope.db"), "annotate") == 0
    assert al.learning_curve(str(tmp_path / "nope.db"), "annotate").empty


# ---------------------------------------------------------------------------
# Retrain + re-rank
# ---------------------------------------------------------------------------

def test_round_two_queue_differs_from_round_one_after_a_retrain(screen):
    """The whole point: the second round must not serve the first's order.

    Built so a passing result cannot be an accident. Round 1 is fitted on
    labels from one corner of the screen; round 2 adds labels that contradict
    it, so a genuinely retrained model ranks a *different* population as
    uncertain. The two queues are compared after removing the crops that were
    labelled in between, so "the queue changed" cannot just mean "the
    labelled ones dropped out".
    """
    paths = [r["png_path"] for r in screen["rows"]]
    features = screen["features"]

    first = {paths[i]: (1 if i % 2 == 0 else 2) for i in range(20)}
    label(screen["db"], first)
    al.record_labels(screen["db"], "annotate", first, 0)
    round_one = al.retrain_round(screen["db"], "annotate", features=features,
                                 seed=0, save_model=False)
    assert round_one.round_index == 0
    assert round_one.score_columns == ["al_prob_0", "al_prob_1"]
    assert round_one.scored == 120

    queue_one = al.build_queue(screen["db"], "annotate", diversity="none")
    # the queue is now ranked by the round's own scores, not the stale `pred`
    assert queue_one.attrs["spacr_active_learning"]["pred_columns"] == \
        ["al_prob_0", "al_prob_1"]

    # Contradicting labels: for these crops the SAME feature now means the
    # other class, so the refitted model has to move.
    second = {paths[i]: (2 if i % 2 == 0 else 1) for i in range(60, 100)}
    label(screen["db"], second)
    al.record_labels(screen["db"], "annotate", second, 1)
    round_two = al.retrain_round(screen["db"], "annotate", features=features,
                                 seed=0, save_model=False)
    assert round_two.round_index == 1
    assert round_two.n_new_labels == 40

    queue_two = al.build_queue(screen["db"], "annotate", diversity="none")

    still_unlabelled = set(queue_two["png_path"])
    order_one = [p for p in queue_one["png_path"] if p in still_unlabelled]
    order_two = list(queue_two["png_path"])
    assert order_one, "there is something left to compare"
    assert order_two[:20] != order_one[:20], (
        "round 2 re-ranked the SAME crops differently — otherwise the loop "
        "is open and the annotator is being served a stale model's opinion")


def test_a_round_holds_out_by_well_rather_than_at_random(screen):
    paths = [r["png_path"] for r in screen["rows"]]
    labels = {paths[i]: (1 if i % 2 == 0 else 2) for i in range(60)}
    label(screen["db"], labels)
    result = al.retrain_round(screen["db"], "annotate",
                              features=screen["features"], seed=0,
                              save_model=False)
    assert "no group appears on both sides" in result.split_rule
    assert result.report["n"] > 0
    assert sum(result.report["class_support"]) == result.report["n"]
    # per-class held-out accuracy, not just the aggregate
    assert set(result.per_class) == {"1", "2"}


def test_a_round_refuses_when_a_grouped_split_is_impossible(screen):
    """All labels from one well cannot yield an independent accuracy."""
    paths = [r["png_path"] for r in screen["rows"]]
    labels = {paths[i]: (1 if i % 2 == 0 else 2) for i in range(10)}
    label(screen["db"], labels)
    with pytest.raises(ValueError, match="one well.*memorised"):
        al.retrain_round(screen["db"], "annotate",
                         features=screen["features"], seed=0,
                         save_model=False)


def test_a_round_refuses_to_fit_on_too_few_labels(screen):
    paths = [r["png_path"] for r in screen["rows"]]
    label(screen["db"], {paths[0]: 1, paths[1]: 2})
    with pytest.raises(ValueError, match="Only 2 labels"):
        al.retrain_round(screen["db"], "annotate",
                         features=screen["features"], save_model=False)


def test_a_round_refuses_to_fit_on_one_class(screen):
    paths = [r["png_path"] for r in screen["rows"]]
    label(screen["db"], {paths[i]: 1 for i in range(20)})
    with pytest.raises(ValueError, match="at least two classes"):
        al.retrain_round(screen["db"], "annotate",
                         features=screen["features"], save_model=False)


def test_a_round_writes_a_model_and_a_card_beside_it(screen):
    paths = [r["png_path"] for r in screen["rows"]]
    labels = {paths[i]: (1 if i % 2 == 0 else 2) for i in range(60)}
    label(screen["db"], labels)
    result = al.retrain_round(screen["db"], "annotate",
                              features=screen["features"], seed=0)
    assert result.model_path and os.path.isfile(result.model_path)
    assert result.card_path and os.path.isfile(result.card_path)

    import json
    card = json.loads(open(result.card_path).read())
    assert card["module"] == "active_learning"
    assert card["split_rule"] == result.split_rule
    assert card["classes"] == ["1", "2"]
    assert card["extra"]["round"] == result.round_index
    assert card["extra"]["n_labels"] == 60
    # the card carries where the labels came from, not only how many
    assert card["extra"]["annotation_coverage"]["by_class"]

    # and its held-out numbers are its own matrix, recomputable
    matrix = np.asarray(card["held_out"]["confusion_matrix"], dtype=float)
    assert card["held_out"]["accuracy"] == pytest.approx(
        float(np.trace(matrix) / matrix.sum()))


def test_a_round_records_itself_on_the_learning_curve(screen):
    paths = [r["png_path"] for r in screen["rows"]]
    label(screen["db"], {paths[i]: (1 if i % 2 == 0 else 2)
                         for i in range(40)})
    first = al.retrain_round(screen["db"], "annotate",
                             features=screen["features"], seed=0,
                             save_model=False)
    label(screen["db"], {paths[i]: (1 if i % 2 == 0 else 2)
                         for i in range(40, 100)})
    second = al.retrain_round(screen["db"], "annotate",
                              features=screen["features"], seed=0,
                              save_model=False)

    curve = al.learning_curve(screen["db"], "annotate")
    assert list(curve["round"]) == [0, 1]
    assert list(curve["n_labels"]) == [40, 100]
    assert list(curve["n_new_labels"]) == [40, 60]
    # the first round has nothing to be a gain over
    assert np.isnan(curve["gain"].iloc[0])
    assert curve["gain"].iloc[1] == pytest.approx(
        second.accuracy - first.accuracy)
    assert curve["per_class"].iloc[1]
    assert "no group appears on both sides" in curve["split_rule"].iloc[1]

    text = al.format_learning_curve(curve, al.should_stop(curve))
    assert "Active-learning rounds" in text
    assert "worst class" in text


def test_predict_proba_pads_classes_a_fold_never_saw():
    class _Partial:
        classes_ = np.array([0, 2])

        def predict_proba(self, x):
            return np.tile([0.3, 0.7], (len(x), 1))

    out = al._predict_proba(_Partial(), np.zeros((4, 2)), 3)
    assert out.shape == (4, 3)
    assert out[:, 0] == pytest.approx(0.3)
    assert out[:, 1] == pytest.approx(0.0)
    assert out[:, 2] == pytest.approx(0.7), "class 2 stays class 2"


# ---------------------------------------------------------------------------
# The stopping rule
# ---------------------------------------------------------------------------

def _curve(accuracies, new_labels=30, n_holdout=200):
    """A learning curve with a chosen accuracy per round."""
    return pd.DataFrame({
        "round": list(range(len(accuracies))),
        "finished_utc": [""] * len(accuracies),
        "n_labels": np.cumsum([new_labels] * len(accuracies)),
        "n_new_labels": [new_labels] * len(accuracies),
        "n_holdout": [n_holdout] * len(accuracies),
        "holdout_accuracy": list(accuracies),
        "holdout_f1_macro": list(accuracies),
        "per_class": [{}] * len(accuracies),
        "split_rule": [""] * len(accuracies),
        "model_type": [""] * len(accuracies),
        "model_path": [""] * len(accuracies),
        "card_path": [""] * len(accuracies),
        "measure": [""] * len(accuracies),
        "diversity": [""] * len(accuracies),
        "notes": [[]] * len(accuracies),
        "gain": pd.Series(accuracies).diff(),
    })


def test_stopping_rule_fires_on_a_flat_curve():
    """0.3 % over the last 50 labels is the signal to stop."""
    verdict = al.should_stop(_curve([0.80, 0.900, 0.901, 0.9015]),
                             label_window=50, min_gain=0.003)
    assert verdict.stop is True
    assert bool(verdict) is True
    assert verdict.trend == "flat"
    assert verdict.labels_in_window >= 50
    assert verdict.gain == pytest.approx(0.9015 - 0.900)
    assert "not buying measurable accuracy" in verdict.reason


def test_stopping_rule_does_not_fire_on_a_rising_curve():
    """Same shape, same row count, same labels — only the accuracy differs."""
    verdict = al.should_stop(_curve([0.60, 0.70, 0.80, 0.90]),
                             label_window=50, min_gain=0.003)
    assert verdict.stop is False
    assert verdict.trend == "rising"
    # rounds are 30 labels each, so a 50-label window spans the last TWO:
    # the baseline is round 1 (0.70) and the gain is measured to 0.90.
    assert verdict.window_from == 1
    assert verdict.labels_in_window == 60
    assert verdict.gain == pytest.approx(0.20)
    assert "Still learning" in verdict.reason


def test_stopping_rule_waits_until_the_window_is_full():
    verdict = al.should_stop(_curve([0.90, 0.901], new_labels=5),
                             label_window=50)
    assert verdict.stop is False
    assert verdict.labels_in_window == 5
    assert "waits for 50" in verdict.reason


def test_stopping_rule_needs_more_than_one_round():
    verdict = al.should_stop(_curve([0.9]))
    assert verdict.stop is False
    assert "Only 1 round" in verdict.reason
    empty = al.should_stop(pd.DataFrame())
    assert empty.stop is False
    assert "No round has been recorded" in empty.reason


def test_stopping_rule_calls_a_falling_curve_what_it_is():
    # the 50-label window spans the last two rounds, so the comparison is
    # round 1 (0.95) against round 3 (0.85)
    verdict = al.should_stop(_curve([0.80, 0.95, 0.90, 0.85]), label_window=50)
    assert verdict.stop is True
    assert verdict.trend == "falling"
    assert verdict.gain == pytest.approx(-0.10)
    assert "FELL" in verdict.reason
    assert "not convergence" in verdict.reason


def test_stopping_rule_separates_flat_from_unmeasurable():
    """A tiny held-out set cannot resolve 0.3 %, and the verdict says so."""
    small = al.should_stop(_curve([0.90, 0.90, 0.901], n_holdout=20),
                           label_window=50, min_gain=0.003)
    assert small.stop is True
    assert small.confident is False
    assert "standard error" in small.reason
    assert small.noise == pytest.approx(np.sqrt(0.901 * 0.099 / 20))

    big = al.should_stop(_curve([0.60, 0.75, 0.90], n_holdout=20000),
                         label_window=50)
    assert big.stop is False
    assert big.confident is True


def test_stopping_verdict_is_serialisable():
    verdict = al.should_stop(_curve([0.9, 0.9, 0.9]))
    payload = verdict.to_dict()
    assert payload["stop"] is True
    assert set(payload) >= {"stop", "reason", "gain", "trend", "confident"}
    assert "StoppingVerdict(" in repr(verdict)


# ---------------------------------------------------------------------------
# Object routing (the half spacr.qt.linked_selection calls into)
# ---------------------------------------------------------------------------

def test_crops_for_object_keys_returns_exactly_the_requested_keys(screen):
    rows = screen["rows"]
    # object keys are plateID_rowID_columnID_fieldID_object_label
    wanted = [f"{r['plateID']}_{r['rowID']}_{r['columnID']}_{r['fieldID']}_"
              f"{r['cell_id'][1:]}" for r in (rows[7], rows[3], rows[70])]
    resolved = al.crops_for_object_keys(screen["db"], wanted,
                                        annotation_column="annotate")
    assert [p for p, _ in resolved] == [rows[7]["png_path"],
                                        rows[3]["png_path"],
                                        rows[70]["png_path"]], "caller's order"


def test_crops_for_object_keys_drops_keys_that_are_not_there(screen):
    rows = screen["rows"]
    good = (f"{rows[0]['plateID']}_{rows[0]['rowID']}_{rows[0]['columnID']}_"
            f"{rows[0]['fieldID']}_0")
    resolved = al.crops_for_object_keys(screen["db"],
                                        [good, "p9_r9_c9_f9_999"])
    assert [p for p, _ in resolved] == [rows[0]["png_path"]]
    assert al.crops_for_object_keys(screen["db"], []) == []


def test_crops_for_object_keys_reads_the_existing_label(screen):
    rows = screen["rows"]
    label(screen["db"], {rows[0]["png_path"]: 2})
    key = f"p1_r1_c1_f1_0"
    resolved = al.crops_for_object_keys(screen["db"], [key],
                                        annotation_column="annotate")
    assert resolved == [(rows[0]["png_path"], 2)]
    without = al.crops_for_object_keys(screen["db"], [key])
    assert without == [(rows[0]["png_path"], None)]


def test_crops_for_object_keys_also_accepts_paths_and_prcfo(screen):
    rows = screen["rows"]
    resolved = al.crops_for_object_keys(
        screen["db"], [rows[5]["png_path"], rows[6]["prcfo"]])
    assert [p for p, _ in resolved] == [rows[5]["png_path"],
                                        rows[6]["png_path"]]


def _two_children_db(tmp_path):
    """A crop table holding a nucleus 1 and a pathogen 1 in ONE field.

    The exact collision the object type went into the key for: two objects
    that used to share ``p1_r1_c1_f1_1``, so one of them could not be opened
    and which one depended on the row order below.
    """
    db = str(tmp_path / "measurements.db")
    frame = pd.DataFrame([
        {"png_path": "/crops/nucleus_png/p1_r1_c1_f1_o1.png",
         "file_name": "p1_r1_c1_f1_o1.png", "prcfo": "p1_r1_c1_f1_o1",
         "plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "nucleus_id": "o1", "pathogen_id": None},
        {"png_path": "/crops/pathogen_png/p1_r1_c1_f1_o1.png",
         "file_name": "p1_r1_c1_f1_o1.png", "prcfo": "p1_r1_c1_f1_o1",
         "plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "nucleus_id": None, "pathogen_id": "o1"},
    ])
    con = sqlite3.connect(db)
    try:
        frame.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db


def test_a_nucleus_and_a_pathogen_with_one_label_open_as_two_crops(tmp_path):
    """The defect, end to end. Two keys in, two different crops out."""
    db = _two_children_db(tmp_path)
    resolved = al.crops_for_object_keys(
        db, ["p1_r1_c1_f1_nucleus1", "p1_r1_c1_f1_pathogen1"])
    assert [p for p, _ in resolved] == [
        "/crops/nucleus_png/p1_r1_c1_f1_o1.png",
        "/crops/pathogen_png/p1_r1_c1_f1_o1.png"]
    # And in the caller's order, not the table's.
    reversed_order = al.crops_for_object_keys(
        db, ["p1_r1_c1_f1_pathogen1", "p1_r1_c1_f1_nucleus1"])
    assert [p for p, _ in reversed_order] == [
        "/crops/pathogen_png/p1_r1_c1_f1_o1.png",
        "/crops/nucleus_png/p1_r1_c1_f1_o1.png"]


def test_an_untyped_key_still_opens_one_of_them_as_it_always_did(tmp_path):
    """Under-specified, not broken. It named one crop before and still does."""
    db = _two_children_db(tmp_path)
    resolved = al.crops_for_object_keys(db, ["p1_r1_c1_f1_1"])
    assert len(resolved) == 1


def test_a_typed_key_falls_back_when_the_crop_table_cannot_say_what_it_is(
        tmp_path):
    """A row that has said nothing has not contradicted the key.

    Resolving nothing here would take a lasso made in a typed view and open
    an empty grid on a database whose ``png_list`` predates the ``*_id``
    columns — silence where there used to be a crop.
    """
    db = str(tmp_path / "measurements.db")
    con = sqlite3.connect(db)
    try:
        pd.DataFrame([{
            "png_path": "/crops/cell_png/p1_r1_c1_f1_o1.png",
            "file_name": "p1_r1_c1_f1_o1.png", "prcfo": "p1_r1_c1_f1_o1",
            "plateID": "p1", "rowID": "r1", "columnID": "c1",
            "fieldID": "f1",
        }]).to_sql("png_list", con, index=False)
    finally:
        con.close()
    resolved = al.crops_for_object_keys(db, ["p1_r1_c1_f1_nucleus1"])
    assert [p for p, _ in resolved] == ["/crops/cell_png/p1_r1_c1_f1_o1.png"]


def test_the_png_id_column_map_is_the_one_the_writer_uses():
    """Derived from the schema to keep spacr.utils out of this import chain.

    Two copies of the same mapping is how they drift, and a drift here means
    a crop's object type is read off a column the writer never fills.
    """
    from spacr.utils import PNG_CROP_MODE_BY_ID_COLUMN

    assert al.PNG_ID_COLUMN_TYPES == PNG_CROP_MODE_BY_ID_COLUMN


def test_object_label_survives_the_png_list_sentinels():
    assert al._object_label("o5") == "5"
    assert al._object_label(5) == "5"
    assert al._object_label(5.0) == "5"
    assert al._object_label("omulti") == ""
    assert al._object_label("onone") == ""
    assert al._object_label(None) == ""
    assert al._object_label("") == ""


def test_holdout_report_matches_a_hand_computed_matrix():
    y = [0, 0, 0, 1, 1, 2]
    probs = np.array([
        [0.9, 0.1, 0.0],   # 0 -> 0
        [0.9, 0.1, 0.0],   # 0 -> 0
        [0.1, 0.9, 0.0],   # 0 -> 1  (a mistake)
        [0.1, 0.9, 0.0],   # 1 -> 1
        [0.1, 0.9, 0.0],   # 1 -> 1
        [0.1, 0.1, 0.8],   # 2 -> 2
    ])
    report = al.holdout_report(y, probs, ["a", "b", "c"])
    assert report["confusion_matrix"] == [[2, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert report["accuracy"] == pytest.approx(5 / 6)
    assert report["per_class_accuracy"] == pytest.approx([2 / 3, 1.0, 1.0])
    assert report["class_support"] == [3, 2, 1]
    assert report["predicted_support"] == [2, 3, 1]
    # macro-F1 by hand: class a P=1.0 R=2/3 -> 0.8; b P=2/3 R=1 -> 0.8; c 1.0
    assert report["f1_macro"] == pytest.approx((0.8 + 0.8 + 1.0) / 3)


def test_holdout_report_is_the_same_function_deep_spacr_uses():
    import spacr.deep_spacr as D
    y = [0, 1, 1, 0]
    probs = [0.2, 0.8, 0.3, 0.1]
    assert D.held_out_report(y, probs) == al.holdout_report(y, probs)
