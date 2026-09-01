"""A generated annotation set opens, instead of being reported and abandoned.

`spacr.annotation_dataset` never overwrites an existing set -- it may already
carry hand-made labels -- so a second one lands as `png_list_2`, a third as
`png_list_3`. The annotation screen read `png_list` and only `png_list`, so
everything after the first was written and then unopenable: a user who had just
asked for a set and was shown the old one would reasonably conclude it failed.

Every engine reader now takes the table as a keyword, defaulting to `png_list`,
so nothing changes for a caller that does not pass one.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr.qt.annotate_engine import (DEFAULT_PNG_TABLE, AnnotateSettings,
                                      class_counts, count_rows,
                                      ensure_annotation_column, fetch_page)


@pytest.fixture
def database(tmp_path):
    """A database holding two crop tables with different contents."""
    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    for table, rows in (("png_list", [("a.png", 1), ("b.png", None)]),
                        ("png_list_2", [("c.png", 2), ("d.png", 2),
                                        ("e.png", None)])):
        connection.execute(
            f'create table "{table}" (png_path TEXT, annotate INTEGER)')
        connection.executemany(
            f'insert into "{table}" values (?, ?)', rows)
    connection.commit()
    connection.close()
    return str(path)


def test_the_default_is_still_the_first_table(database):
    """Nothing changes for a caller that does not name one."""
    assert count_rows(database) == 2


def test_a_second_table_can_be_counted(database):
    assert count_rows(database, table="png_list_2") == 3


def test_a_page_comes_from_the_named_table(database):
    rows = fetch_page(database, "annotate", 0, 10, table="png_list_2")
    assert [r[0] for r in rows] == ["c.png", "d.png", "e.png"]


def test_the_first_table_is_untouched_by_reading_the_second(database):
    fetch_page(database, "annotate", 0, 10, table="png_list_2")
    assert [r[0] for r in fetch_page(database, "annotate", 0, 10)] \
        == ["a.png", "b.png"]


def test_counts_come_from_the_named_table(database):
    assert class_counts(database, "annotate", table="png_list_2") == [(2, 2)]
    assert class_counts(database, "annotate") == [(1, 1)]


def test_a_column_is_added_to_the_named_table(database):
    """A writer pointed at the wrong table would put a user's labels on
    somebody else's rows."""
    ensure_annotation_column(database, "infected", table="png_list_2")

    connection = sqlite3.connect(database)
    try:
        second = [r[1] for r in connection.execute(
            'PRAGMA table_info("png_list_2")')]
        first = [r[1] for r in connection.execute(
            'PRAGMA table_info("png_list")')]
    finally:
        connection.close()
    assert "infected" in second
    assert "infected" not in first, "the column landed on the wrong table"


def test_the_settings_carry_the_table():
    assert AnnotateSettings("/tmp/x").png_table == DEFAULT_PNG_TABLE


def test_the_save_worker_writes_to_the_named_table(database):
    from spacr.qt.annotate_engine import SaveWorker

    ensure_annotation_column(database, "annotate", table="png_list_2")
    worker = SaveWorker(database, "annotate", table="png_list_2")
    assert worker.table == "png_list_2"


def test_the_screen_opens_what_it_generated():
    """A source check: the generator's table has to reach the settings, or the
    set is written and then not shown."""
    from pathlib import Path

    import spacr.qt.screens.annotate as annotate

    source = Path(annotate.__file__).read_text(encoding="utf-8")
    assert "self._settings.png_table = table" in source
    assert "This screen currently opens" not in source, (
        "the old refusal is still there")


def test_the_screen_passes_the_table_to_the_engine():
    from pathlib import Path

    import spacr.qt.screens.annotate as annotate

    source = Path(annotate.__file__).read_text(encoding="utf-8")
    assert source.count("table=self._settings.png_table") >= 8, (
        "some engine calls still assume png_list")
