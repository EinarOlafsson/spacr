"""Table-driven tests of ``spacr.cell_montage``: which cells a montage draws, and from where.

``select_montage`` turns one coefficient into a plan: the wells that
contribute, how many cells each gives and the exact caption arithmetic quoted
against each, plus plan-level notes (exclusions, the attribution pre-flight).
``_sudoku_calls`` decides which wells the multi-guide attribution learns from;
``load_montage_objects`` refuses a database with no scores in words that say
whether a score table was offered; ``resolve_montage_crop_source`` answers a
missing source with a sentence and says when the channels were assumed.
Every caption is asserted in full, because a wrong picture under a caption
that still reads as if everything worked is the failure these guard.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import cell_montage as cm

PER_WELL = 6
GENE_OF = {"GENE_1": "GENE", "GENE_2": "GENE", "OTHER_1": "OTHER",
           "OTHER_2": "OTHER", "OTHER_3": "OTHER", "GRA14_1": "GRA14"}


def _objects(wells, per_well=PER_WELL):
    """The per-object frame a montage selects from, one plate, evenly scored."""
    rows = []
    for index, well in enumerate(wells):
        row_id, column_id = well.split("_")
        for label in range(1, per_well + 1):
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id, "fieldID": "f1",
                "object_label": label,
                "pred": round(0.05 + 0.9 * (label - 1) / (per_well - 1), 4),
                "area": 100.0 + label + 10 * index,
                "perimeter": 40.0 + 2 * label,
            })
    return pd.DataFrame(rows)


def _counts(fractions, cell_count=PER_WELL):
    """A ``regression_data.csv``-shaped frame: one row per well and guide."""
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id,
                "grna": guide, "gene": GENE_OF[guide], "fraction": fraction,
                "cell_count": cell_count, "pred": 0.5,
            })
    return pd.DataFrame(rows)


def _round(frac, norm, total, n=6, single=False):
    """The fraction-arithmetic caption a well records."""
    text = (f"round({frac} x 6) = {n}, where {frac} is this guide's {frac} "
            f"normalised by {norm} (the well's fractions sum to {total})")
    if single:
        text += ("; (attributed needs more than one guide in a well; this one "
                 "holds a single guide)")
    return text


EXCLUDED_1 = ("excluded 1 guide(s) named by exclude_grnas before any fraction "
              "was formed")
EXCLUDED_2 = ("excluded 2 guide(s) named by exclude_grnas before any fraction "
              "was formed")
EFFECTS = {"GENE_1": 0.5, "OTHER_1": -0.3}
GRID = pd.DataFrame({"area": [0.6, 0.4, -0.5], "perimeter": [0.3, 0.2, -0.2]},
                    index=["GENE_1", "GENE_2", "OTHER_1"])

# id, fractions, coefficient, select_montage kwargs,
# -> plan notes, [(well, n_selected, caption)], n_objects, capped, n_before_cap
SELECT_CASES = [
    ("a well emptied by an exclusion does not stop the pre-flight",
     {"r1_c1": {"GENE_1": 1.0}, "r1_c2": {"GENE_1": 0.4, "OTHER_1": 0.6}},
     "GENE_1",
     dict(picking="attributed", effects=EFFECTS, exclude_grnas=["GENE_1"]),
     (EXCLUDED_1, "GENE_1: no well carries it."),
     [("plate1_r1_c1", 6, _round(1, 1, 0, single=True)),
      ("plate1_r1_c2", 4, "round(0.6667 x 6) = 4, where 0.6667 is this "
       "guide's 0.4 normalised by 1.667 (the well's fractions sum to 0.6); "
       "(attributed needs more than one guide in a well; this one holds a "
       "single guide)")],
     10, False, 10),
    ("a lone guide is ranked, a shared well is attributed",
     {"r1_c1": {"GENE_1": 1.0}, "r1_c2": {"GENE_1": 0.5, "OTHER_1": 0.5}},
     "GENE_1", dict(picking="attributed", effects=EFFECTS),
     ("GENE_1 can be attributed in 2 of its 2 wells (best possible posterior "
      "1.000 against a 0.55 threshold).",),
     [("plate1_r1_c1", 6, _round(1, 1, 1, single=True)),
      ("plate1_r1_c2", 3, "attributed chose 3 of 6 classified cell(s); the "
       "fraction would support 3")],
     9, False, 9),
    ("excluding every guide leaves the pre-flight silent",
     {"r1_c1": {"GENE_1": 1.0}, "r1_c2": {"GENE_1": 0.5, "OTHER_1": 0.5}},
     "GENE_1",
     dict(picking="attributed", effects=EFFECTS,
          exclude_grnas=["GENE_1", "OTHER_1"]),
     (EXCLUDED_2,),
     [("plate1_r1_c1", 6, _round(1, 1, 0, single=True)),
      ("plate1_r1_c2", 3, _round(0.5, 1, 0, n=3, single=True))],
     9, False, 9),
    ("a gene-level multivariate montage falls through to the fractions",
     {"r1_c1": {"GENE_1": 0.4, "OTHER_1": 0.6},
      "r1_c2": {"GENE_2": 0.5, "OTHER_1": 0.5}, "r1_c3": {"OTHER_1": 1.0}},
     "GENE",
     dict(picking="multivariate", effects_grid=GRID,
          effects={"GENE_1": 0.5, "GENE_2": 0.4, "OTHER_1": -0.3}),
     ("GENE_1 can be attributed in 1 of its 1 well (best possible posterior "
      "1.000 against a 0.55 threshold).",
      "GENE_2 can be attributed in 1 of its 1 well (best possible posterior "
      "1.000 against a 0.55 threshold)."),
     [("plate1_r1_c1", 2, _round(0.4, 1, 1, n=2)),
      ("plate1_r1_c2", 3, _round(0.5, 1, 1, n=3))],
     5, False, 5),
    ("the cap annotates only the wells it trimmed",
     {"r1_c1": {"GENE_1": 1.0, "OTHER_1": 0.0},
      "r1_c2": {"GENE_1": 0.02, "OTHER_1": 0.98}},
     "GENE_1", dict(cap=2), (),
     [("plate1_r1_c1", 2, _round(1, 1, 1)
       + "; trimmed by the montage cap from 6 to 2"),
      ("plate1_r1_c2", 0, "round(0.02 x 6) rounds to zero, so the design "
       "expects no object from this well (0.02 is this guide's 0.02 "
       "normalised by 1, the well's fractions summing to 1; 6 of the well's "
       "6 objects carry a score)")],
     2, True, 6),
]


def test_select_montage_plans_exact_wells_counts_and_captions(subtests):
    """Each plan's notes, per-well counts and captions, total and cap bookkeeping are exactly as quoted."""
    for (case_id, fractions, coefficient, kwargs, notes, wells, n_objects,
         capped, n_before) in SELECT_CASES:
        with subtests.test(msg=case_id):
            plan = cm.select_montage(_objects(tuple(fractions)),
                                     _counts(fractions), coefficient, 0.4,
                                     **kwargs)
            assert plan.notes == notes
            assert [(w.well, w.n_selected, w.note)
                    for w in plan.wells] == wells
            assert (plan.n_objects, bool(plan.capped),
                    plan.n_before_cap) == (n_objects, capped, n_before)
            # The plan's total is the sum of what its wells gave.
            assert sum(w.n_selected for w in plan.wells) == n_objects


def test_sudoku_scope_is_the_guides_wells_plus_the_anchoring_wells():
    """Sudoku learns from the guide's own well and the rival-anchoring well only, annotating every cell there."""
    sudoku_per_well = 8
    fractions = {
        "r1_c1": {"GRA14_1": 0.4, "OTHER_1": 0.6},    # the guide's own well
        "r1_c2": {"OTHER_2": 0.7, "OTHER_3": 0.3},    # no rival of GRA14_1
        "r1_c3": {"OTHER_1": 0.55, "OTHER_2": 0.45},  # OTHER_1 anchors here
        "r1_c4": {"OTHER_2": 1.0},                    # no rival of GRA14_1
    }
    objects = _objects(tuple(fractions), per_well=sudoku_per_well)
    notes: list = []

    result = cm._sudoku_calls(objects, _counts(fractions, sudoku_per_well),
                              ["prc"], "grna", "fraction", "pred", "GRA14_1",
                              notes)

    # Rows 0-7 are r1_c1 and rows 16-23 are r1_c3.
    assert result == {index: "OTHER_1"
                      for index in [*range(0, 8), *range(16, 24)]}
    assert {objects.loc[i, "prc"] for i in result} == {"plate1_r1_c1",
                                                       "plate1_r1_c3"}
    assert notes == [
        "sudoku: 1 well(s) hold GRA14_1; 1 more well(s) joined to anchor the "
        "2 guide(s) it is compared against",
        "sudoku: 16 of 16 cell(s) annotated across 2 well(s); 0 abstained"]


def test_load_montage_objects_says_whether_a_score_table_was_offered(
        tmp_path, subtests):
    """No pred column refuses with a sentence naming the offer (none / empty); a readable score CSV loads instead."""
    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    crops = plate / "data" / "w" / "cell_png"
    crops.mkdir(parents=True)
    rows = []
    for i in range(4):
        name = f"plate1_r1_c{i}_f1_{i}.png"
        (crops / name).write_bytes(b"x")
        rows.append({"png_path": str(crops / name), "file_name": name,
                     "plateID": "plate1", "rowID": "r1", "columnID": f"c{i}",
                     "fieldID": "f1", "object_label": i})
    db = plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows).to_sql("png_list", conn, index=False)
    csv = tmp_path / "plate1_dv.csv"
    pd.DataFrame([{"path": r["png_path"], "pred": 0.1 * (i + 1)}
                  for i, r in enumerate(rows)]).to_csv(csv, index=False)

    head = (f"'png_list' in {db} has no 'pred' column, so no object carries a "
            "classification score. ")
    tail = (" Either load the score CSVs this screen was fitted on, or run "
            "Classify and merge its predictions into the database.")
    for case_id, scores, middle in (
            ("no score table offered", None,
             "No score table was offered alongside it."),
            ("an empty offer", [],
             "The loaded score table has no column that joins to it either.")):
        with subtests.test(msg=case_id):
            kwargs = {} if scores is None else {"scores": scores}
            with pytest.raises(cm.MissingScores) as raised:
                cm.load_montage_objects(str(db), **kwargs)
            assert str(raised.value) == head + middle + tail

    with subtests.test(msg="a readable score CSV loads"):
        loaded = cm.load_montage_objects(str(db), scores=[str(csv)])
        assert loaded["pred"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert loaded["prc"].tolist() == [f"plate1_r1_c{i}" for i in range(4)]


NO_SRC = "no 'src' to resolve a crop source from"
ASSUMED = ("no channel list: this run's measurements.db records no png_dims "
           "and none was typed, so the picture is made from the default "
           "planes [2, 1, 0]. Set the channels if that is not what you meant.")
MERGED_ROUTE = "crop route: live crop from merged/ (object-shaped available)"


def test_resolve_montage_crop_source_answers_every_source_shape(tmp_path,
                                                                subtests):
    """Missing roots get the same sentence and no route; a merged-only run is usable and says its channels were assumed unless given."""
    with_meas = tmp_path / "plate1"
    (with_meas / "merged").mkdir(parents=True)
    (with_meas / "measurements").mkdir(parents=True)
    np.save(with_meas / "merged" / "plate1_r1_c1_f1.npy",
            np.zeros((8, 8, 3), np.uint16))
    bare = tmp_path / "plate2"
    (bare / "merged").mkdir(parents=True)
    np.save(bare / "merged" / "plate1_r1_c1_f1.npy",
            np.zeros((8, 8, 5), np.uint16))
    assert not (bare / "measurements" / "measurements.db").exists()

    for empty in ({"src": ""}, {"src": []}, {}, ""):
        with subtests.test(msg=f"no root: {empty!r}"):
            choice = cm.resolve_montage_crop_source(empty)
            assert (choice.available, choice.kind, choice.reason,
                    choice.source) == (False, "", NO_SRC, None)
            assert (choice.requirements.route, choice.requirements.missing,
                    choice.requirements.assumed) == ("none", (NO_SRC,), ())
            assert choice.requirement_notes() == (
                f"crop route: no route to pixels: {NO_SRC}",
                f"MISSING -- {NO_SRC}")

    # id, settings, kwargs, -> assumed notes
    for case_id, settings, kwargs, assumed in (
            ("a run folder with an empty measurements dir",
             {"src": str(with_meas)}, {}, (ASSUMED,)),
            ("declared channels drop the assumption",
             {"src": str(with_meas)}, {"channels": [0, 1, 2]}, ()),
            ("a merged/ path given as a list, objects to cut",
             {"src": [str(bare / "merged")]},
             {"objects": pd.DataFrame({"object_label": [1]})}, (ASSUMED,)),
            ("a run with no measurements database", {"src": str(bare)}, {},
             (ASSUMED,))):
        with subtests.test(msg=case_id):
            choice = cm.resolve_montage_crop_source(settings, **kwargs)
            assert (choice.available, choice.kind) == (True, "merged")
            assert choice.reason == ("stream images (array): no pre-generated "
                                     "crops found, cutting from merged/*.npy")
            assert (choice.requirements.route, choice.requirements.missing,
                    choice.requirements.assumed) == ("merged-mask", (),
                                                     assumed)
            assert choice.requirement_notes() == (
                MERGED_ROUTE, *(f"ASSUMED -- {note}" for note in assumed))
