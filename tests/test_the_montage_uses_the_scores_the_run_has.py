"""A database with no `pred` is not a screen without scores.

Reported 2026-08-19: "when i try to show cells i get: No attached database
yielded per-object rows with a classification score ... Run Classify and merge
its predictions into the database first."

The refusal was true and the advice was wrong. The scores were in the score
CSVs the regression module already had loaded -- one row per cell, and the fit
was run on exactly those numbers. Measured on the real screen: 60,816 of
60,816 objects matched.

THE DATABASE IS NOT WRITTEN TO. A montage is a read, which is the same rule
the crop-path re-rooting follows.
"""
import os
import sqlite3

import pandas as pd
import pytest

from spacr.cell_montage import MissingScores, load_montage_objects
from spacr.predictions import attach_predictions


@pytest.fixture()
def screen(tmp_path):
    """A plate whose png_list has no score column, and its score CSV."""
    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    crops = plate / "data" / "w" / "cell_png"
    crops.mkdir(parents=True)
    rows = []
    for i in range(6):
        name = f"plate1_A0{i}_1_{i}.png"
        (crops / name).write_bytes(b"x")
        rows.append({"png_path": str(crops / name), "file_name": name,
                     "plateID": "plate1", "rowID": "r1", "columnID": f"c{i}",
                     "fieldID": "f1", "object_label": i})
    db = plate / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows).to_sql("png_list", conn, index=False)

    scores = pd.DataFrame([
        {"path": r["png_path"], "pred": 0.1 * (i + 1),
         "cv_predictions": i % 2, "prc": "plate1_r1_c%d" % i}
        for i, r in enumerate(rows)])
    csv = tmp_path / "plate1_dv.csv"
    scores.to_csv(csv, index=False)
    return {"db": str(db), "csv": str(csv), "scores": scores,
            "objects": pd.DataFrame(rows)}


def test_without_scores_it_still_refuses(screen):
    with pytest.raises(MissingScores) as raised:
        load_montage_objects(screen["db"])

    message = str(raised.value)
    assert "no score table was offered" in message.lower()


def test_the_refusal_names_both_places_it_looked(screen, tmp_path):
    """"Run Classify" is wrong advice when the user has the scores loaded."""
    empty = tmp_path / "nothing.csv"
    empty.write_text("a,b\n1,2\n")

    with pytest.raises(MissingScores) as raised:
        load_montage_objects(screen["db"], scores=[str(empty)])

    message = str(raised.value)
    assert "loaded score table" in message
    assert "load the score CSVs" in message


def test_the_scores_are_taken_from_the_loaded_csv(screen):
    frame = load_montage_objects(screen["db"], scores=[screen["csv"]])

    assert "pred" in frame.columns
    assert frame["pred"].notna().sum() == 6


def test_the_database_is_not_written_to(screen):
    import hashlib

    before = hashlib.md5(open(screen["db"], "rb").read()).hexdigest()
    load_montage_objects(screen["db"], scores=[screen["csv"]])
    after = hashlib.md5(open(screen["db"], "rb").read()).hexdigest()

    assert before == after
    with sqlite3.connect(screen["db"]) as conn:
        columns = [r[1] for r in conn.execute('pragma table_info("png_list")')]
    assert "pred" not in columns


def test_a_frame_or_a_path_or_several_are_all_accepted(screen):
    for offered in (screen["scores"], screen["csv"], [screen["csv"]]):
        frame = load_montage_objects(screen["db"], scores=offered)
        assert "pred" in frame.columns


def test_the_join_is_the_one_the_database_merge_uses(screen):
    """Two join implementations would eventually disagree about which object
    got which number, so the montage borrows `merge_prediction_results`'."""
    joined, matched = attach_predictions(screen["objects"], screen["scores"])

    assert matched == 6
    assert list(joined["pred"].round(1)) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_a_database_that_HAS_scores_is_untouched_by_any_of_this(screen):
    """The fallback must not override a screen's own numbers."""
    with sqlite3.connect(screen["db"]) as conn:
        conn.execute('ALTER TABLE "png_list" ADD COLUMN pred REAL')
        conn.execute('UPDATE "png_list" SET pred = 0.99')

    frame = load_montage_objects(screen["db"], scores=[screen["csv"]])

    assert set(frame["pred"].round(2)) == {0.99}
