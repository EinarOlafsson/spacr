"""Opening a source in Annotate while another connection reads the database.

Issue #184: picking a source raised ``sqlite3.OperationalError: database is
locked`` out of ``ensure_annotation_column``. That function ran a write
transaction on every open -- ``CREATE INDEX IF NOT EXISTS`` takes the write
lock even when nothing is created -- and under the rollback journal its commit
waits for every reader of the file. A reader that outlasted the 30 s busy
timeout (a previous open's population count, a Measure or Classify run, a
backup client) became an unhandled error in the slot.
"""
from __future__ import annotations

import os
import sqlite3
import time

import pytest

pytest.importorskip("PySide6")

from spacr.database_concurrency import DatabaseBusy
from spacr.qt import annotate_engine


def _database(tmp_path, *, with_column: bool):
    folder = tmp_path / "expt" / "measurements"
    folder.mkdir(parents=True)
    db_path = folder / "measurements.db"
    conn = sqlite3.connect(db_path)
    columns = "png_path TEXT" + (", test INTEGER" if with_column else "")
    conn.execute(f"CREATE TABLE png_list ({columns})")
    conn.executemany("INSERT INTO png_list (png_path) VALUES (?)",
                     [(f"/crops/{i}.png",) for i in range(20)])
    if with_column:
        conn.execute("CREATE INDEX idx_png_path ON png_list (png_path)")
    conn.commit()
    conn.close()
    return str(db_path)


def _hold_a_reader(db_path):
    """A second connection with an open read transaction: a SHARED lock."""
    reader = sqlite3.connect(db_path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT png_path FROM png_list").fetchone()
    return reader


def _columns(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(png_list)")}
    finally:
        conn.close()


def test_an_annotated_source_opens_while_another_connection_reads(tmp_path):
    """The column and index exist, so opening needs no write lock at all."""
    db_path = _database(tmp_path, with_column=True)
    reader = _hold_a_reader(db_path)
    try:
        started = time.monotonic()
        annotate_engine.ensure_annotation_column(db_path, "test")
        elapsed = time.monotonic() - started
    finally:
        reader.close()
    assert elapsed < 2.0, f"waited {elapsed:.1f} s for a lock it did not need"


def test_a_new_column_waits_a_bounded_time_then_raises_database_busy(
        tmp_path, monkeypatch):
    """A column that must be added still needs the lock; the wait is bounded."""
    monkeypatch.setattr(annotate_engine, "_SCHEMA_WRITE_TIMEOUT_S", 0.3)
    db_path = _database(tmp_path, with_column=False)
    reader = _hold_a_reader(db_path)
    try:
        started = time.monotonic()
        with pytest.raises(DatabaseBusy):
            annotate_engine.ensure_annotation_column(db_path, "test")
        elapsed = time.monotonic() - started
    finally:
        reader.close()
    assert elapsed < 5.0
    assert "test" not in _columns(db_path)
    annotate_engine.ensure_annotation_column(db_path, "test")
    assert "test" in _columns(db_path)


def test_the_screen_says_the_database_is_busy_instead_of_raising(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The slot shows a message and leaves the previous source open."""
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt.screens.annotate import AnnotateScreen

    db_path = _database(tmp_path, with_column=False)
    src = os.path.dirname(os.path.dirname(db_path))

    def locked(*_args, **_kwargs):
        raise DatabaseBusy("database is locked")

    warnings = []
    monkeypatch.setattr("spacr.qt.screens.annotate.ensure_annotation_column",
                        locked)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warnings.append(args))

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    before = (screen._settings.src, screen._settings.db_path)
    screen._open_source(src)

    assert len(warnings) == 1
    assert (screen._settings.src, screen._settings.db_path) == before
