"""Instruction 167 — the refusal was true and the advice was wrong.

    "No attached database yielded per-object rows with a classification score.
     ... Run Classify and merge its predictions into the database first."

The scores were not missing. They were in the score CSVs the regression module
was ALREADY HOLDING — the run was fitted on exactly those numbers — so the
montage was telling a user to go and produce something they had had loaded the
whole time, and to write it into a database it does not need to touch.

Three properties: the run's scores are used, a refusal that is still right
names BOTH places it looked, and writing to a measurements database is offered
rather than done.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.cell_montage_view import CellMontageView  # noqa: E402


def _digest(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _database(tmp_path, *, with_pred: bool = False):
    """A png_list with no `pred`, which is the maintainer's own shape."""
    path = tmp_path / "measurements.db"
    columns = ["file_name", "plateID", "rowID", "columnID", "fieldID",
               "object_label", "png_path"]
    if with_pred:
        columns.append("pred")
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE png_list ({', '.join(columns)})")
        for i in range(6):
            row = [f"img{i}.png", "plate1", "r1", "c1", "1", i,
                   str(tmp_path / f"img{i}.png")]
            if with_pred:
                row.append(0.5)
            db.execute(
                f"INSERT INTO png_list VALUES ({','.join('?' * len(columns))})",
                row)
    return str(path)


def _scores(tmp_path):
    path = tmp_path / "plate1_dv.csv"
    pd.DataFrame({"path": [f"img{i}.png" for i in range(6)],
                  "pred": [i / 6 for i in range(6)],
                  "cv_predictions": [0, 1] * 3,
                  "prc": ["plate1_r1_c1"] * 6,
                  "object": list(range(6))}).to_csv(path, index=False)
    return str(path)


def _view(qtbot, databases=(), score_files=()):
    view = CellMontageView(
        database_provider=lambda: [{"plate": "plate1", "database": d}
                                   for d in databases],
        threaded=False)
    qtbot.addWidget(view)
    view.score_csvs = lambda: tuple(score_files)
    view.databases = lambda: tuple(databases)
    return view


# -- the refusal names both halves ------------------------------------------

def test_with_no_scores_anywhere_the_refusal_names_both_places():
    from spacr.qt.widgets.cell_montage_view import no_score_refusal

    said = no_score_refusal(["plate1_dv.csv", "plate2_dv.csv"],
                            ["measurements.db has no 'pred' column"])
    # NOT "run Classify": both places are named, and the score files are one.
    assert "database" in said.lower()
    assert "score file" in said.lower()
    assert "plate1_dv.csv" in said
    assert "measurements.db has no 'pred' column" in said


def test_the_refusal_says_when_no_score_file_is_loaded_at_all():
    from spacr.qt.widgets.cell_montage_view import no_score_refusal

    said = no_score_refusal(())
    assert "No score file is loaded" in said
    assert "without modifying a database" in said
    # And it says where they WOULD come from, which is the actionable half.
    assert "join it in memory" in said


def test_many_score_files_are_summarised_rather_than_all_listed():
    from spacr.qt.widgets.cell_montage_view import no_score_refusal

    said = no_score_refusal([f"plate{i}_dv.csv" for i in range(9)])
    assert "9 loaded score files" in said
    assert "+6 more" in said


def test_the_refusal_never_tells_the_user_to_run_classify_first():
    """The advice that was wrong, held so it cannot come back."""
    from spacr.qt.widgets.cell_montage_view import no_score_refusal

    for files in ((), ["plate1_dv.csv"]):
        assert "Run Classify" not in no_score_refusal(files)


# -- writing is offered, never done -----------------------------------------

def test_nothing_is_written_unless_the_user_says_so(qtbot, tmp_path):
    database = _database(tmp_path)
    before = _digest(database)
    view = _view(qtbot, [database], [_scores(tmp_path)])

    assert view.write_scores_into_the_databases(
        confirm=lambda dbs, files: False) == {}
    assert _digest(database) == before, "the database was written to anyway"


def test_the_confirmation_is_shown_what_is_written_and_what_from(qtbot,
                                                                 tmp_path):
    database, scores = _database(tmp_path), _scores(tmp_path)
    view = _view(qtbot, [database], [scores])
    seen = {}

    view.write_scores_into_the_databases(
        confirm=lambda dbs, files: seen.update(dbs=dbs, files=files) or False)
    assert seen["dbs"] == [database]
    assert seen["files"] == [scores]


def test_accepting_writes_the_scores_and_says_how_many_matched(qtbot, tmp_path):
    database = _database(tmp_path)
    before = _digest(database)
    view = _view(qtbot, [database], [_scores(tmp_path)])

    written = view.write_scores_into_the_databases(
        confirm=lambda dbs, files: True)
    assert written, "nothing was merged"
    assert _digest(database) != before
    assert "rows matched" in view.status_text()
    # And it says the montage did not need it, so a user does not conclude
    # the picture depended on this.
    assert "already uses loaded scores" in view.status_text()


def test_with_nothing_to_merge_it_says_so_rather_than_opening_a_dialog(qtbot,
                                                                       tmp_path):
    view = _view(qtbot, [], [_scores(tmp_path)])
    asked = []

    assert view.write_scores_into_the_databases(
        confirm=lambda *_a: asked.append(1) or True) == {}
    assert asked == []
    assert "nothing to merge" in view.status_text().lower()


def test_a_database_that_will_not_take_them_is_named_and_the_rest_go_on(
        qtbot, tmp_path):
    good = _database(tmp_path)
    bad = str(tmp_path / "not-a-database.db")
    open(bad, "w").write("this is not sqlite")
    view = _view(qtbot, [bad, good], [_scores(tmp_path)])

    written = view.write_scores_into_the_databases(confirm=lambda *_a: True)
    assert good in written
    assert bad not in written
