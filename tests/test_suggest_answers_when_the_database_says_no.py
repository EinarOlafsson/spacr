"""Item 288: what Suggest reports when the database refuses a question.

`spacr.suggest` reads and writes a ``measurements.db`` that another process
(a Measure run, a second window) may be holding, and whose crop table may
have been made by an older spaCR or by pandas. Every query it makes is
therefore allowed to fail, and each failure has a defined answer: zero
suggestions pending, no verdicts, the column not added. Those answers are
what the Suggest menu, the class-counts dialog and the judging screen show,
so they are pinned here with REAL SQLite refusals -- an authorizer that
denies one read, one pragma or one ``ALTER`` -- rather than with a fake
connection.

The ordinary behaviour is pinned in ``test_suggest_never_overwrites_an_
annotation.py`` and ``test_a_suggestion_can_be_judged.py``.
"""
from __future__ import annotations

import sqlite3

import pytest

pd = pytest.importorskip("pandas")

from spacr import suggest  # noqa: E402


def _table(tmp_path, columns, rows):
    """A ``png_list`` with exactly ``columns`` and ``rows``."""
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE png_list ({', '.join(columns)})")
        if rows:
            marks = ",".join("?" * len(rows[0]))
            db.executemany(f"INSERT INTO png_list VALUES ({marks})", rows)
    return str(path)


def _refuse(monkeypatch, opener, *, action, table=None, column=None):
    """Open ``opener``'s connections with one SQLite action denied.

    :param opener: ``"_connect_read_only"`` or ``"_connect_writable"``.
    :param action: the ``sqlite3.SQLITE_*`` action code to deny.
    :param column: for ``SQLITE_READ``, the one column whose read is denied.
    """
    real = getattr(suggest, opener)

    def decide(code, arg1, arg2, _db, _source):
        if code != action:
            return sqlite3.SQLITE_OK
        if column is not None and arg2 != column:
            return sqlite3.SQLITE_OK
        if table is not None and arg1 != table:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def opened(db_path):
        connection = real(db_path)
        connection.set_authorizer(decide)
        return connection

    monkeypatch.setattr(suggest, opener, opened)


def test_a_table_whose_columns_cannot_be_listed_has_no_suggestions(
        tmp_path, monkeypatch):
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/a.png", 11), ("/b.png", 12)])
    assert suggest.pending_suggestions(db_path, "test") == 2
    _refuse(monkeypatch, "_connect_read_only", action=sqlite3.SQLITE_PRAGMA)
    assert suggest.pending_suggestions(db_path, "test") == 0


def test_an_unreadable_annotation_column_has_no_suggestions(
        tmp_path, monkeypatch):
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/a.png", 11)])
    _refuse(monkeypatch, "_connect_read_only", action=sqlite3.SQLITE_READ,
            column="test")
    assert suggest.pending_suggestions(db_path, "test") == 0


def test_a_collision_check_that_cannot_read_still_writes_only_empty_rows(
        tmp_path, monkeypatch):
    """Nothing is known to collide, and the write still keeps its IS NULL."""
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/labelled.png", 2), ("/blank.png", None)])
    _refuse(monkeypatch, "_connect_read_only", action=sqlite3.SQLITE_READ,
            column="test")
    frame = pd.DataFrame({"png_path": ["/labelled.png", "/blank.png"],
                          "suggested": [1, 1], "stored": [11, 11]})
    assert suggest.write_suggestions(db_path, "test", frame) == 1
    with sqlite3.connect(db_path) as db:
        stored = dict(db.execute("SELECT png_path, test FROM png_list"))
    assert stored == {"/labelled.png": 2, "/blank.png": 11}


def test_no_annotation_column_means_no_verdict_column(tmp_path):
    db_path = _table(tmp_path, ["png_path TEXT"], [("/a.png",)])
    assert suggest.ensure_verdict_column(db_path, "") is False


def test_a_missing_crop_table_gets_no_verdict_column(tmp_path):
    db_path = _table(tmp_path, ["png_path TEXT"], [("/a.png",)])
    assert suggest.ensure_verdict_column(db_path, "test",
                                         png_table="no_such_table") is False


def test_a_crop_table_that_cannot_be_listed_gets_no_verdict_column(
        tmp_path, monkeypatch):
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/a.png", None)])
    _refuse(monkeypatch, "_connect_writable", action=sqlite3.SQLITE_PRAGMA)
    assert suggest.ensure_verdict_column(db_path, "test") is False


def test_a_crop_table_that_cannot_be_altered_gets_no_verdict_column(
        tmp_path, monkeypatch):
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/a.png", None)])
    _refuse(monkeypatch, "_connect_writable",
            action=sqlite3.SQLITE_ALTER_TABLE)
    assert suggest.ensure_verdict_column(db_path, "test") is False
    with sqlite3.connect(db_path) as db:
        names = [row[1] for row in db.execute("PRAGMA table_info(png_list)")]
    assert names == ["png_path", "test"]


def test_verdicts_for_no_crops_or_no_column_are_empty(tmp_path):
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/a.png", 11)])
    assert suggest.fetch_verdicts(db_path, "test", []) == {}
    assert suggest.fetch_verdicts(db_path, "test", ["/a.png"]) == {}


def test_unreadable_verdicts_are_reported_as_none(tmp_path, monkeypatch):
    db_path = _table(tmp_path,
                     ["png_path TEXT", "test INTEGER", "test_verdict INTEGER"],
                     [("/a.png", 1, 1), ("/b.png", None, -2)])
    assert suggest.fetch_verdicts(db_path, "test", ["/a.png", "/b.png"]) == \
        {"/a.png": 1, "/b.png": -2}
    _refuse(monkeypatch, "_connect_read_only", action=sqlite3.SQLITE_READ,
            column="test_verdict")
    assert suggest.fetch_verdicts(db_path, "test", ["/a.png", "/b.png"]) == {}


def test_a_verdict_that_is_not_a_number_is_skipped(tmp_path):
    """A TEXT verdict column, as pandas writes one, may hold anything."""
    db_path = _table(tmp_path,
                     ["png_path TEXT", "test INTEGER", "test_verdict TEXT"],
                     [("/a.png", 1, "1"), ("/b.png", None, "maybe")])
    assert suggest.fetch_verdicts(db_path, "test", ["/a.png", "/b.png"]) == \
        {"/a.png": 1}


def test_judgement_counts_without_an_annotation_column(tmp_path):
    """Verdicts alone still count; nothing is left without suggestions."""
    db_path = _table(tmp_path, ["png_path TEXT", "test_verdict INTEGER"],
                     [("/a.png", 1), ("/b.png", -1), ("/c.png", -2)])
    assert suggest.judgement_counts(db_path, "test") == \
        {"left": 0, "confirmed": 1, "rejected": 2}


def test_judgement_counts_without_a_verdict_column(tmp_path):
    db_path = _table(tmp_path, ["png_path TEXT", "test INTEGER"],
                     [("/a.png", 11), ("/b.png", 12), ("/c.png", 1)])
    assert suggest.judgement_counts(db_path, "test") == \
        {"left": 2, "confirmed": 0, "rejected": 0}


@pytest.mark.parametrize("unreadable, expected", [
    ("test", {"left": 0, "confirmed": 1, "rejected": 1}),
    ("test_verdict", {"left": 1, "confirmed": 0, "rejected": 0}),
])
def test_judgement_counts_report_zero_for_what_cannot_be_read(
        tmp_path, monkeypatch, unreadable, expected):
    db_path = _table(tmp_path,
                     ["png_path TEXT", "test INTEGER", "test_verdict INTEGER"],
                     [("/a.png", 11, None), ("/b.png", 1, 1),
                      ("/c.png", None, -1)])
    _refuse(monkeypatch, "_connect_read_only", action=sqlite3.SQLITE_READ,
            column=unreadable)
    assert suggest.judgement_counts(db_path, "test") == expected


def test_rejections_need_the_annotation_column_too(tmp_path):
    db_path = _table(tmp_path, ["png_path TEXT", "test_verdict INTEGER"],
                     [("/a.png", -1)])
    assert suggest.rejected_suggestions(db_path, "test") == {}


def test_unreadable_rejections_are_reported_as_none(tmp_path, monkeypatch):
    db_path = _table(tmp_path,
                     ["png_path TEXT", "test INTEGER", "test_verdict INTEGER"],
                     [("/a.png", None, -2)])
    assert suggest.rejected_suggestions(db_path, "test") == {"/a.png": 2}
    _refuse(monkeypatch, "_connect_read_only", action=sqlite3.SQLITE_READ,
            column="test_verdict")
    assert suggest.rejected_suggestions(db_path, "test") == {}


def test_a_rejection_that_is_not_a_number_is_skipped(tmp_path):
    """In a TEXT column, ``'-x' < 0`` is compared as text and is TRUE."""
    db_path = _table(tmp_path,
                     ["png_path TEXT", "test INTEGER", "test_verdict TEXT"],
                     [("/a.png", None, "-1"), ("/b.png", None, "-x")])
    assert suggest.rejected_suggestions(db_path, "test") == {"/a.png": 1}
