"""The annotation writer keeps judgement (verdict) batches apart from labels.

Pinned behaviour of :class:`spacr.qt.annotate_engine.SaveWorker` for a
batch aimed at a column other than the annotation column (item 512's
``<column>_verdict``):

* a verdict batch queued on its own is written to its column and leaves
  the annotation column untouched;
* when a transaction fails, the verdict batch it carried is kept under its
  column, apart from the annotation batch, and nothing is written;
* a verdict batch submitted after a failure is kept under its column too,
  without counting as a pending annotation batch.

Real sqlite files, no Qt event loop needed.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr.qt import annotate_engine as engine
from spacr.qt.annotate_engine import SaveWorker


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "measurements.db"
    con = sqlite3.connect(path)
    try:
        con.execute(
            'CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
            'annotate INTEGER, annotate_verdict INTEGER)')
        con.executemany('INSERT INTO "png_list" VALUES (?, NULL, NULL)',
                        [("a.png",), ("b.png",)])
        con.commit()
    finally:
        con.close()
    return str(path)


def _rows(db):
    con = sqlite3.connect(db)
    try:
        return con.execute(
            'SELECT png_path, annotate, annotate_verdict FROM "png_list" '
            'ORDER BY png_path').fetchall()
    finally:
        con.close()


def test_a_verdict_batch_alone_is_written_to_its_own_column(db):
    worker = SaveWorker(db, "annotate")
    worker.start()
    try:
        worker.submit({"a.png": 1}, column="annotate_verdict")
    finally:
        worker.stop()

    assert worker.last_error is None
    assert worker.pending_batches == 0
    assert _rows(db) == [("a.png", None, 1), ("b.png", None, None)]


def test_a_failed_transaction_keeps_the_verdict_batch_under_its_column(
        db, monkeypatch):
    def _locked(_conn):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "transaction", _locked)
    worker = SaveWorker(db, "annotate")
    worker.start()
    try:
        worker.submit({"b.png": 2}, column="annotate_verdict")
    finally:
        worker.stop()

    assert "database is locked" in worker.last_error
    assert worker._failed_extra == {"annotate_verdict": {"b.png": 2}}
    assert worker._failed_batch == {}
    assert _rows(db) == [("a.png", None, None), ("b.png", None, None)]


def test_a_verdict_submitted_after_a_failure_is_kept_not_queued(db):
    worker = SaveWorker(db, "annotate")
    with worker._lock:
        worker._last_error = "the database is read-only"

    worker.submit({"a.png": 1}, column="annotate_verdict")
    worker.submit({"b.png": 0}, column="annotate_verdict")

    assert worker.pending_batches == 0
    assert worker._failed_batch is None
    assert worker._failed_extra == {
        "annotate_verdict": {"a.png": 1, "b.png": 0}}
    assert _rows(db) == [("a.png", None, None), ("b.png", None, None)]
