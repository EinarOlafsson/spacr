"""Suggest's empty answers, which are the ones a reviewer actually sees.

`suggest_from_scores` returns an EMPTY frame for six different reasons, and
the reviewer is told which by ``note`` and by nothing else -- the button that
calls it has no other way to explain itself.  A wrong note here is not a
crash: it is a user retraining a model that was already trained, or hunting
for a database problem that is really "you have annotated everything".
`tests/test_suggest_never_overwrites_an_annotation.py` covers the path where
there IS something to suggest, and one of the six notes; these are the rest,
plus the three places where writing back does nothing rather than something.

Each note is asserted by the WORDS a reader needs, not by an exact string, so
rewording the sentence does not fail the test while dropping the meaning
does.
"""
from __future__ import annotations

import sqlite3

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from spacr import suggest
from spacr.active_learning import ROUND_PRED_PREFIX


def _crop_table(path, columns, rows):
    """Write a png_list with exactly these columns and rows."""
    db = sqlite3.connect(path)
    db.execute(f"CREATE TABLE png_list ({', '.join(columns)})")
    if rows:
        placeholders = ",".join("?" * len(rows[0]))
        db.executemany(f"INSERT INTO png_list VALUES ({placeholders})", rows)
    db.commit()
    db.close()
    return str(path)


def _scored(path, rows, *, annotation="test"):
    """A png_list with two score columns and an annotation column."""
    return _crop_table(
        path,
        ["png_path TEXT", f"{annotation} INTEGER",
         f"{ROUND_PRED_PREFIX}0 REAL", f"{ROUND_PRED_PREFIX}1 REAL"],
        rows,
    )


# -- the six reasons the frame comes back empty -----------------------------


def test_a_database_with_no_crop_table_says_so_instead_of_raising(tmp_path):
    """Pointed at the wrong .db, Suggest reports it; it does not traceback."""
    path = tmp_path / "measurements.db"
    sqlite3.connect(path).close()

    result = suggest.suggest_from_scores(str(path), "test")

    assert result.frame.empty
    assert "crop table" in result.note
    assert result.scored == 0


def test_an_empty_crop_table_is_not_reported_as_an_unscored_one(tmp_path):
    """"Nothing measured yet" and "nothing scored yet" want different fixes.

    A measure run that produced no crops and a measure run whose crops were
    never retrained send the user to two different buttons, so the two notes
    must not be the same sentence.
    """
    path = _scored(tmp_path / "measurements.db", [])

    result = suggest.suggest_from_scores(path, "test")

    assert result.frame.empty
    assert "empty" in result.note
    assert "Retrain" not in result.note


def test_an_annotation_column_that_does_not_exist_yet_means_nothing_is_labelled(
    tmp_path,
):
    """The first Suggest of a screen runs before the column is created.

    Every crop is then unannotated, which is the only reading that lets the
    first run do anything at all.
    """
    path = _crop_table(
        tmp_path / "measurements.db",
        ["png_path TEXT", f"{ROUND_PRED_PREFIX}0 REAL",
         f"{ROUND_PRED_PREFIX}1 REAL"],
        [("/crops/a.png", 0.1, 0.9), ("/crops/b.png", 0.8, 0.2)],
    )

    result = suggest.suggest_from_scores(path, "brand_new_column")

    assert result.scored == 2
    assert sorted(result.frame["png_path"]) == ["/crops/a.png", "/crops/b.png"]


def test_the_caller_may_name_the_classes_the_score_columns_stand_for(tmp_path):
    """The classes default to what is in the column; they are not forced to.

    A screen whose annotations are 3 and 7 has score columns 0 and 1, and the
    caller is the only one who knows which is which.
    """
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", None, 0.1, 0.9), ("/crops/b.png", None, 0.8, 0.2)],
    )

    result = suggest.suggest_from_scores(path, "test", classes=[3, 7])

    assert result.classes == [3, 7]
    got = dict(zip(result.frame["png_path"], result.frame["suggested"]))
    assert got == {"/crops/a.png": 7, "/crops/b.png": 3}
    assert dict(zip(result.frame["png_path"], result.frame["stored"])) == {
        "/crops/a.png": 7 + suggest.SUGGESTION_OFFSET,
        "/crops/b.png": 3 + suggest.SUGGESTION_OFFSET,
    }


def test_no_classes_at_all_is_reported_rather_than_guessed(tmp_path):
    """Asked for zero classes, Suggest says so instead of inventing one."""
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", None, 0.1, 0.9)],
    )

    result = suggest.suggest_from_scores(path, "test", classes=[])

    assert result.frame.empty
    assert "annotated" in result.note


def test_a_fully_annotated_screen_says_so_rather_than_saying_unscored(tmp_path):
    """The note that means "you are done", which is not an error."""
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", 1, 0.1, 0.9), ("/crops/b.png", 2, 0.8, 0.2)],
    )

    result = suggest.suggest_from_scores(path, "test")

    assert result.frame.empty
    assert "every crop already carries a value" in result.note
    assert result.classes == [1, 2]


def test_unannotated_crops_whose_scores_are_unusable_are_named_as_such(
    tmp_path,
):
    """A crop added after the retrain has NULL scores, not bad ones.

    It cannot be suggested and it is not an empty database; the note has to
    separate the two or the user re-runs Measure for nothing.
    """
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/labelled.png", 1, 0.5, 0.5),
         ("/crops/after_the_retrain.png", None, None, None)],
    )

    result = suggest.suggest_from_scores(path, "test")

    assert result.frame.empty
    assert "usable score" in result.note
    assert result.classes == [1]


def test_a_score_column_with_no_class_number_sorts_last(tmp_path):
    """Column ORDER is the class mapping, so an odd name may not shift it.

    Anything named like a score column but without an integer suffix goes to
    the end rather than between two classes, where it would silently
    renumber them.
    """
    columns = [
        f"{ROUND_PRED_PREFIX}1",
        f"{ROUND_PRED_PREFIX}notes",
        f"{ROUND_PRED_PREFIX}0",
        "png_path",
    ]

    assert suggest._score_columns(columns) == [
        f"{ROUND_PRED_PREFIX}0",
        f"{ROUND_PRED_PREFIX}1",
        f"{ROUND_PRED_PREFIX}notes",
    ]


# -- writing back, when there is nothing to write ---------------------------


def test_writing_an_empty_frame_writes_nothing_and_says_zero(tmp_path):
    path = _scored(
        tmp_path / "measurements.db", [("/crops/a.png", None, 0.1, 0.9)],
    )

    assert suggest.write_suggestions(path, "test", pd.DataFrame()) == 0
    assert suggest.write_suggestions(
        path, "test", pd.DataFrame({"stored": [11]})) == 0

    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM png_list WHERE test IS NOT NULL"
        ).fetchone()[0] == 0


def test_a_frame_without_the_suggested_column_is_still_written(tmp_path):
    """The collision check reads ``suggested``; ``stored`` is what is written.

    A caller that kept only the two columns the UPDATE needs is not asking
    for anything dangerous, and refusing it would be a surprise.
    """
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", None, 0.1, 0.9), ("/crops/b.png", None, 0.8, 0.2)],
    )
    frame = pd.DataFrame({
        "png_path": ["/crops/a.png", "/crops/b.png"],
        "stored": [12, 11],
    })

    assert suggest.write_suggestions(path, "test", frame) == 2

    with sqlite3.connect(path) as db:
        assert dict(db.execute("SELECT png_path, test FROM png_list")) == {
            "/crops/a.png": 12, "/crops/b.png": 11,
        }


def test_rows_whose_value_is_missing_are_skipped_rather_than_written_as_null(
    tmp_path,
):
    """A NaN in ``stored`` must not become an UPDATE that clears the cell."""
    path = _scored(
        tmp_path / "measurements.db", [("/crops/a.png", None, 0.1, 0.9)],
    )
    frame = pd.DataFrame({
        "png_path": ["/crops/a.png"],
        "suggested": [1.0],
        "stored": [np.nan],
    })

    assert suggest.write_suggestions(path, "test", frame) == 0

    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM png_list WHERE test IS NOT NULL"
        ).fetchone()[0] == 0


def test_resolving_an_empty_selection_touches_nothing(tmp_path):
    """"Accept the rows I selected" with no rows selected accepts none.

    The empty list is not None, and reading it as "every suggestion" would
    accept the whole screen on an empty selection -- the worst possible
    reading of a click.
    """
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", 11, 0.1, 0.9), ("/crops/b.png", 12, 0.8, 0.2)],
    )

    assert suggest.resolve_suggestions(path, "test", keep=True, paths=[]) == 0

    with sqlite3.connect(path) as db:
        assert sorted(
            row[0] for row in db.execute("SELECT test FROM png_list")
        ) == [11, 12]


def test_counting_pending_suggestions_in_a_column_that_does_not_exist_is_zero(
    tmp_path,
):
    """A column that was never made has no suggestions waiting in it.

    THE BUG THIS PINS, found 2026-09-19.  SQLite reads a double-quoted name
    that is not a column as a STRING LITERAL instead of failing, so
    ``WHERE "cell_class" > 10`` became ``WHERE 'cell_class' > 10`` -- and
    TEXT compares greater than every integer in SQLite, so it matched EVERY
    ROW.  The Class counts dialog therefore told a user who had never
    pressed Suggest that all 2,000 of their crops were "suggested, not yet
    kept or thrown away", and the Suggest menu offered to resolve them.  The
    count is 0 whether the column is absent or merely empty, and those are
    the two assertions below.
    """
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", None, 0.1, 0.9), ("/crops/b.png", 1, 0.8, 0.2)],
    )

    assert suggest.pending_suggestions(path, "never_created") == 0
    assert suggest.pending_suggestions(path, "test") == 0


def test_a_real_pending_count_is_still_counted(tmp_path):
    """The negative control for the case above: the guard counts nothing out."""
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", 11, 0.1, 0.9), ("/crops/b.png", 12, 0.8, 0.2),
         ("/crops/c.png", 1, 0.5, 0.5), ("/crops/d.png", None, 0.5, 0.5)],
    )

    assert suggest.pending_suggestions(path, "test") == 2


def test_resolving_a_column_that_does_not_exist_changes_nothing(tmp_path):
    """Same SQLite quirk from the writing side, where it would have thrown.

    ``WHERE "never_created" > 10`` selects every row, and only the failure
    of ``SET "never_created" = ...`` stopped the update -- an
    OperationalError out of a button press.  There is nothing to resolve, so
    the answer is zero.
    """
    path = _scored(
        tmp_path / "measurements.db",
        [("/crops/a.png", 11, 0.1, 0.9), ("/crops/b.png", 1, 0.8, 0.2)],
    )

    assert suggest.resolve_suggestions(path, "never_created", keep=True) == 0
    assert suggest.resolve_suggestions(path, "never_created", keep=False) == 0

    with sqlite3.connect(path) as db:
        assert sorted(
            row[0] for row in db.execute("SELECT test FROM png_list")
        ) == [1, 11]


def test_writing_into_a_column_that_does_not_exist_finds_no_collision(tmp_path):
    """The collision probe read the same string literal back as a value.

    ``SELECT DISTINCT "never_created" ... WHERE "never_created" >= 10``
    returned the column NAME once, and ``int('never_created')`` raised a
    bare ValueError from inside a refusal that had nothing to refuse.
    """
    path = _crop_table(
        tmp_path / "measurements.db",
        ["png_path TEXT", f"{ROUND_PRED_PREFIX}0 REAL",
         f"{ROUND_PRED_PREFIX}1 REAL"],
        [("/crops/a.png", 0.1, 0.9)],
    )
    frame = pd.DataFrame({
        "png_path": ["/crops/a.png"], "suggested": [1], "stored": [11],
    })

    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        suggest.write_suggestions(path, "never_created", frame)
