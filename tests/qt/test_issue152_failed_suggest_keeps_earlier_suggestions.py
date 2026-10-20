"""A Suggest run that fails or is cancelled puts the earlier suggestions back.

Issue #152: the run clears the pending suggestions as its first step, then
failed to fit, and the annotator was left with neither the old proposals
nor new ones. The run now restores them before it reports.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.suggest import SUGGESTION_OFFSET  # noqa: E402


def _db_with_pending_suggestions(tmp_path: Path) -> Path:
    db = tmp_path / "measurements.db"
    rows = [("/c/a.png", 1), ("/c/b.png", 2),
            ("/c/c.png", SUGGESTION_OFFSET + 1),
            ("/c/d.png", SUGGESTION_OFFSET + 2), ("/c/e.png", None)]
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE png_list (png_path TEXT, annotate INTEGER)")
        conn.executemany("INSERT INTO png_list VALUES (?, ?)", rows)
    return db


def _values(db: Path):
    with sqlite3.connect(db) as conn:
        return dict(conn.execute(
            "SELECT png_path, annotate FROM png_list").fetchall())


def test_a_failed_fit_restores_the_suggestions_it_cleared(tmp_path):
    from spacr.qt.screens import annotate as mod

    db = _db_with_pending_suggestions(tmp_path)
    before = _values(db)
    worker = mod._SuggestWorker(
        str(db), "annotate",
        {"features": pd.DataFrame({"x": [1.0]}, index=["/elsewhere.png"]),
         "round_index": 1, "save_model": False})
    seen_cleared = []
    worker.progress.connect(
        lambda step, _total, _stage: seen_cleared.append(
            _values(db)["/c/c.png"]) if step == 2 else None)
    failed = []
    worker.failed.connect(failed.append)
    worker.run()

    assert seen_cleared == [None]
    assert failed and "feature matrix" in failed[0]
    assert worker.earlier_kept is True
    assert _values(db) == before


def test_a_cancelled_run_restores_the_suggestions_it_cleared(tmp_path):
    from spacr.qt.screens import annotate as mod

    db = _db_with_pending_suggestions(tmp_path)
    before = _values(db)
    worker = mod._SuggestWorker(str(db), "annotate", {"round_index": 1})

    asked = []
    worker.isInterruptionRequested = lambda: bool(asked)

    def _cancel_after_clear(step, _total, _stage):
        if step == 1:
            asked.append(True)

    worker.progress.connect(_cancel_after_clear)
    cancelled = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.run()

    assert cancelled
    assert worker.earlier_kept is True
    assert _values(db) == before


def test_a_crop_annotated_during_the_run_keeps_its_annotation(tmp_path):
    from spacr.suggest import _restore_suggestions, _suggestion_snapshot

    db = _db_with_pending_suggestions(tmp_path)
    snapshot = _suggestion_snapshot(str(db), "annotate")
    assert snapshot == {"/c/c.png": SUGGESTION_OFFSET + 1,
                        "/c/d.png": SUGGESTION_OFFSET + 2}
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE png_list SET annotate = NULL "
                     "WHERE annotate > ?", (SUGGESTION_OFFSET,))
        conn.execute("UPDATE png_list SET annotate = 2 "
                     "WHERE png_path = '/c/c.png'")

    assert _restore_suggestions(str(db), "annotate", snapshot) == 1
    values = _values(db)
    assert values["/c/c.png"] == 2
    assert values["/c/d.png"] == SUGGESTION_OFFSET + 2
